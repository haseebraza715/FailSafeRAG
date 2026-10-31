"""The visual repair (``invoke_vlm``): the initial answer's text evidence plus images of runtime-chosen pages.

Design: ``docs/research/visual-repair-design.md``. This module holds the parts that decide what a repair request
contains: the page policy, the gate, the page renderer and the request builder. None of them reads the evaluation
manifest, clean reference text, gold evidence pages or any score. Their inputs are the frozen runtime manifest,
the no-recovery run's ``requests.jsonl`` and the locked PDF archive.

Page policy ``retrieved-then-gaps-v1`` ranks a document's pages and keeps the first ``budget``:

1. pages of the retrieved chunks, in retrieval rank order, each page once;
2. pages whose MinerU text is not ``ok`` (empty or missing), in page order;
3. every other page, in page order.

Gate ``abstain-or-ocr-gap-v1`` flags a question when the initial outcome is ``no_evidence``, when the initial
answer abstained (``NO_ANSWER``), or when any page of the document has MinerU text that is not ``ok``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer_prompt import PromptTemplate

POLICY_ID = "retrieved-then-gaps-v1"
GATE_ID = "abstain-or-ocr-gap-v1"
DEFAULT_BUDGET = 3

RENDER_DPI = 150
MODEL_LONG_SIDE = 2048
MODEL_JPEG_QUALITY = 90
IMAGE_DETAIL = "high"
# Upper bound per attached image, in input tokens; the same deliberate overstatement as the headroom runs.
IMAGE_TOKEN_BOUND = 8000

IMAGE_NOTE = (
    "\n\nThe user message also contains images of document pages. The images are evidence too. Read them together "
    "with any text blocks, and answer from what the page shows."
)
NO_TEXT_USER_PREFIX = "Question:\n{question}\n\nNo text evidence is available. The evidence is the page image(s) below.\n\n"

# Files the repair must never open. The run checks every path it reads against these.
FORBIDDEN_PATH_PARTS = ("evaluation_manifest", "retrieval_base/gt", "scores.jsonl", "score_summary")


class RepairError(Exception):
    """A repair input is missing, inconsistent or forbidden. Nothing was sent."""


def check_runtime_path(path: Path | str) -> None:
    text = Path(path).as_posix()
    for part in FORBIDDEN_PATH_PARTS:
        if part in text:
            raise RepairError(f"the repair may not read {text}: it is an evaluation-only file")


def select_pages(pages: Sequence[Mapping[str, Any]], retrieved_page_idxs: Iterable[int], budget: int) -> list[int]:
    """Page indexes chosen by ``retrieved-then-gaps-v1``, in the order the images are attached."""
    if budget < 1:
        raise RepairError(f"budget must be at least 1, got {budget}")
    known = [int(p["page_idx"]) for p in pages]
    known_set = set(known)
    order: list[int] = []
    for idx in retrieved_page_idxs:
        if idx not in known_set:
            raise RepairError(f"retrieved page {idx} is not a page of the document")
        if idx not in order:
            order.append(idx)
    for page in sorted(pages, key=lambda p: int(p["page_idx"])):
        idx = int(page["page_idx"])
        if page["ocr_status"] != "ok" and idx not in order:
            order.append(idx)
    for idx in sorted(known):
        if idx not in order:
            order.append(idx)
    return order[:budget]


def gate_flags(*, initial_status: str, abstained: bool, pages: Sequence[Mapping[str, Any]]) -> bool:
    """``abstain-or-ocr-gap-v1``: flag no-evidence, abstained or OCR-gap questions. Takes no threshold."""
    return initial_status == "no_evidence" or bool(abstained) or any(p["ocr_status"] != "ok" for p in pages)


@dataclass(frozen=True)
class PageImage:
    page_idx: int
    sha256: str
    size: tuple[int, int]
    data: bytes


def render_model_jpeg(pdf_bytes: bytes, page_idx: int) -> tuple[bytes, tuple[int, int]]:
    """Render one page as the model sees it: 150 DPI, long side at most 2048 px, JPEG quality 90.

    The settings are those of the headroom runs (``scripts/analysis/ohr_dev_v1_failures/render_pages.py``).
    """
    import pypdfium2

    pdf = pypdfium2.PdfDocument(pdf_bytes)
    try:
        image = pdf[page_idx].render(scale=RENDER_DPI / 72).to_pil()
    finally:
        pdf.close()
    scale = min(1.0, MODEL_LONG_SIDE / max(image.size))
    model_image = image.convert("RGB")
    if scale < 1.0:
        model_image = model_image.resize((round(image.width * scale), round(image.height * scale)))
    out = io.BytesIO()
    model_image.save(out, format="JPEG", quality=MODEL_JPEG_QUALITY)
    return out.getvalue(), model_image.size


class PageRenderer:
    """Renders pages from the locked PDF archive after checking each member against the runtime manifest hash."""

    def __init__(self, archive_path: Path, documents: Mapping[str, Mapping[str, Any]]) -> None:
        check_runtime_path(archive_path)
        self._archive_path = archive_path
        self._documents = documents
        self._pdfs: dict[str, bytes] = {}

    def pdf(self, doc_id: str) -> bytes:
        if doc_id not in self._pdfs:
            pdf = self._documents[doc_id]["pdf"]
            with zipfile.ZipFile(self._archive_path) as archive:
                data = archive.read(pdf["member"])
            if hashlib.sha256(data).hexdigest() != pdf["sha256"]:
                raise RepairError(f"{doc_id}: the archive member's hash differs from the runtime manifest")
            self._pdfs[doc_id] = data
        return self._pdfs[doc_id]

    def page(self, doc_id: str, page_idx: int) -> PageImage:
        data, size = render_model_jpeg(self.pdf(doc_id), page_idx)
        return PageImage(page_idx, hashlib.sha256(data).hexdigest(), (int(size[0]), int(size[1])), data)


def repair_messages(record: Mapping[str, Any], images: Sequence[PageImage]) -> tuple[dict[str, Any], ...]:
    """The chat messages of one repair request: the text message, then each page as ``Image of page N:`` and the image."""
    if [im.page_idx for im in images] != [im["page_idx"] for im in record["images"]]:
        raise RepairError(f"{record['question_id']}: the images differ from the prepared page list")
    for image, prepared in zip(images, record["images"], strict=True):
        if image.sha256 != prepared["sha256"]:
            raise RepairError(f"{record['question_id']}: page {image.page_idx} renders differently from prepare")
    content: list[dict[str, Any]] = [{"type": "text", "text": record["user"]}]
    for image in images:
        content.append({"type": "text", "text": f"Image of page {image.page_idx + 1}:"})
        url = "data:image/jpeg;base64," + base64.b64encode(image.data).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": url, "detail": IMAGE_DETAIL}})
    return ({"role": "system", "content": record["system"]}, {"role": "user", "content": content})


def repair_text(baseline_request: Mapping[str, Any], template: PromptTemplate) -> tuple[str, str, list[int]]:
    """System message, user message and retrieved page order for one question, from the no-recovery request.

    A question the no-recovery run sent keeps its exact user message. One it did not send (no text evidence) gets
    ``NO_TEXT_USER_PREFIX`` and the template's closing line, as in the headroom image-only condition.
    """
    if baseline_request["template_id"] != template.template_id:
        raise RepairError(
            f"{baseline_request['question_id']}: the no-recovery run used {baseline_request['template_id']}, not {template.template_id}"
        )
    system = template.system + IMAGE_NOTE
    if baseline_request["action"] == "send":
        if baseline_request["system"] != template.system:
            raise RepairError(f"{baseline_request['question_id']}: the no-recovery system message differs from the template")
        evidence = sorted(baseline_request["evidence"], key=lambda e: e["rank"])
        return system, baseline_request["user"], [int(e["page_idx"]) for e in evidence]
    user = (NO_TEXT_USER_PREFIX + template.closing_line).format(question=baseline_request["question"])
    return system, user, []
