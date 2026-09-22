from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .asset_paths import to_relative_project_path
from .asset_preparation import page_image_name, page_ocr_name, render_pdf_pages, sha256_file

VIDORE_EXPECTED_ROWS = 500
DEFAULT_DOWNLOAD_DELAY_SEC = 3.0
DEFAULT_ACCEPT_SCORE = 0.92
DEFAULT_ACCEPT_MARGIN = 0.05
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_IMAGE_NAME_RE = re.compile(
    r"^(?:.*/)?(?P<raw_id>.+)_(?P<figure_index>\d+)\.(?P<ext>jpe?g|png)$",
    re.IGNORECASE,
)
_MODERN_ID_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})(?:v\d+)?$")
_LEGACY_SLASH_RE = re.compile(
    r"^(?P<archive>[a-zA-Z][a-zA-Z0-9.-]+)/(?P<num>\d{7})(?:v\d+)?$"
)
_LEGACY_COMPACT_RE = re.compile(
    r"^(?P<archive>[a-zA-Z][a-zA-Z0-9.-]+?)(?P<num>\d{7})(?:v\d+)?$"
)


class ArXivQAPrepareError(ValueError):
    """Raised when ArXivQA full-paper preparation cannot proceed safely."""


@dataclass(frozen=True)
class ArxivIdentifiers:
    raw_id: str
    paper_id: str
    modern_id: str | None
    legacy_id: str | None
    figure_index: int
    image_filename: str


@dataclass
class StagedRow:
    qa_id: str
    question: str
    answer: str
    options: list[str]
    image_filename: str
    figure_index: int
    paper_id: str
    modern_id: str | None
    legacy_id: str | None
    official_id: str
    official_id_aliases: list[str]
    vidore_index: int
    rationale: str | None = None
    image_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceCandidate:
    qa_id: str
    paper_id: str
    figure_index: int
    status: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_evidence_page_ids: list[int] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def default_source_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "arxivqa_source_lock.json"


STAGED_ROW_FINGERPRINT_ALGORITHM = "sha256(canonical-json(sorted staged StagedRow dictionaries))"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint_staged_row_dicts(rows: Iterable[dict[str, Any]]) -> str:
    """Canonical staged-row fingerprint shared by prepare and remap."""
    normalized = [dict(row) for row in rows]
    normalized.sort(key=lambda item: (str(item.get("paper_id") or ""), int(item.get("figure_index") or 0), str(item.get("qa_id") or "")))
    return _sha256_bytes(_canonical_json_bytes(normalized))


def staged_row_fingerprint(staged: Iterable[StagedRow]) -> str:
    return fingerprint_staged_row_dicts(row.to_dict() for row in staged)


def staged_row_dict_from_qa_record(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the staged-row dictionary shape used by the shared fingerprint algorithm."""
    if not isinstance(row, dict):
        raise ArXivQAPrepareError("QA record must be a JSON object for fingerprint verification.")
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}

    def _pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in row and row[key] is not None:
                return row[key]
            if key in meta and meta[key] is not None:
                return meta[key]
        return default

    qa_id = str(_pick("qa_id", "id", "official_id", default="") or "").strip()
    question = str(_pick("question", "query", default="") or "").strip()
    answer = str(_pick("answer", "label", default="") or "").strip()
    image_filename = str(_pick("image_filename", "image", default="") or "").strip().replace("\\", "/")
    paper_id = str(_pick("paper_id", default="") or "").strip()
    if not qa_id or not question or not answer or not image_filename or not paper_id:
        raise ArXivQAPrepareError(
            "QA record is missing fields required to recompute the staged-row fingerprint "
            "(qa_id/id, question, answer, image_filename, paper_id)."
        )
    figure_index = _pick("figure_index", default=None)
    vidore_index = _pick("vidore_index", default=None)
    if figure_index is None or vidore_index is None:
        raise ArXivQAPrepareError(
            f"QA record {qa_id!r} is missing figure_index/vidore_index required for fingerprint verification."
        )
    official_id = str(_pick("official_id", default=qa_id) or qa_id).strip()
    aliases = _pick("official_id_aliases", default=[])
    if not isinstance(aliases, list):
        raise ArXivQAPrepareError(f"QA record {qa_id!r} has invalid official_id_aliases.")
    options = _pick("options", default=[])
    if not isinstance(options, list):
        raise ArXivQAPrepareError(f"QA record {qa_id!r} has invalid options.")
    rationale = _pick("rationale", default=None)
    image_sha256 = _pick("image_sha256", default=None)
    modern_id = _pick("modern_id", default=None)
    legacy_id = _pick("legacy_id", default=None)
    return {
        "qa_id": qa_id,
        "question": question,
        "answer": answer,
        "options": [str(item) for item in options],
        "image_filename": image_filename,
        "figure_index": int(figure_index),
        "paper_id": paper_id,
        "modern_id": (str(modern_id) if modern_id is not None else None),
        "legacy_id": (str(legacy_id) if legacy_id is not None else None),
        "official_id": official_id,
        "official_id_aliases": [str(item) for item in aliases],
        "vidore_index": int(vidore_index),
        "rationale": (str(rationale).strip() if rationale not in (None, "") else None),
        "image_sha256": (str(image_sha256) if image_sha256 is not None else None),
    }


def qa_rows_content_fingerprint(qa_rows: Iterable[dict[str, Any]]) -> str:
    """Recompute the locked staged-row fingerprint from QA source row content."""
    return fingerprint_staged_row_dicts(staged_row_dict_from_qa_record(row) for row in qa_rows)


_ALLOWED_PDF_PINNING = {"not_claimed", "documented_overrides_only"}
_PDF_VERSION_RE = re.compile(r"^v\d+$")


def _validate_pdf_override_schema(overrides: dict[str, Any]) -> None:
    for paper_id, override in overrides.items():
        if not isinstance(override, dict):
            raise ArXivQAPrepareError(f"ArXivQA pdf_override {paper_id!r} must be a JSON object.")
        if str(override.get("paper_id") or "") != paper_id:
            raise ArXivQAPrepareError(
                f"ArXivQA pdf_override key {paper_id!r} does not match its paper_id field."
            )
        version = str(override.get("version") or "")
        if not _PDF_VERSION_RE.fullmatch(version):
            raise ArXivQAPrepareError(f"ArXivQA pdf_override {paper_id!r} has invalid version {version!r}.")
        pdf_url = str(override.get("pdf_url") or "")
        expected_urls = {
            f"https://arxiv.org/pdf/{paper_id}{version}.pdf",
            f"https://arxiv.org/pdf/{paper_id}{version}",
        }
        if pdf_url not in expected_urls:
            raise ArXivQAPrepareError(
                f"ArXivQA pdf_override {paper_id!r} pdf_url {pdf_url!r} does not match "
                f"the pinned version {version!r}."
            )
        if not str(override.get("withdrawal_reason") or "").strip():
            raise ArXivQAPrepareError(f"ArXivQA pdf_override {paper_id!r} must record a withdrawal reason.")
        figure_sha = str(override.get("figure_sha256") or "")
        if not _SHA256_RE.fullmatch(figure_sha):
            raise ArXivQAPrepareError(f"ArXivQA pdf_override {paper_id!r} has an invalid figure hash.")
        matched_page_id = override.get("matched_page_id")
        if not isinstance(matched_page_id, int) or isinstance(matched_page_id, bool) or matched_page_id < 0:
            raise ArXivQAPrepareError(
                f"ArXivQA pdf_override {paper_id!r} has an invalid matched page id."
            )
        for score_key in ("v1_score", "v2_max_score"):
            score = override.get(score_key)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
                raise ArXivQAPrepareError(f"ArXivQA pdf_override {paper_id!r} has an invalid {score_key}.")


def load_source_lock(path: Path | None = None) -> dict[str, Any]:
    lock_path = (path or default_source_lock_path()).resolve()
    payload = read_json(lock_path)
    if not isinstance(payload, dict):
        raise ArXivQAPrepareError(f"ArXivQA source lock must be a JSON object: {lock_path}")
    official = payload.get("official_source")
    vidore = payload.get("vidore_source")
    policy = payload.get("project_role_policy")
    if not isinstance(official, dict) or not isinstance(vidore, dict) or not isinstance(policy, dict):
        raise ArXivQAPrepareError("ArXivQA source lock is missing official, ViDoRe, or project-role policy data.")
    for label, source in (("official", official), ("ViDoRe", vidore)):
        digest = str(source.get("sha256") or "")
        revision = str(source.get("revision") or "")
        filename = str(source.get("filename") or "")
        if not _SHA256_RE.fullmatch(digest) or not revision or not filename:
            raise ArXivQAPrepareError(f"ArXivQA source lock has incomplete {label} source evidence.")
    fingerprint = str(payload.get("staged_row_fingerprint_sha256") or "")
    if not _SHA256_RE.fullmatch(fingerprint):
        raise ArXivQAPrepareError("ArXivQA source lock has no valid staged-row fingerprint.")
    if payload.get("upstream_split") != "test" or official.get("upstream_split") != "test":
        raise ArXivQAPrepareError("ArXivQA source lock must preserve upstream_split='test'.")
    if payload.get("supervisor_approved_project_role") != "validation":
        raise ArXivQAPrepareError("ArXivQA source lock must explicitly approve project role 'validation'.")
    if policy.get("allows_upstream_test_as_project_validation") is not True or not str(policy.get("policy_id") or ""):
        raise ArXivQAPrepareError("ArXivQA source lock lacks an explicit upstream-test-to-project-validation policy.")
    if not isinstance(vidore.get("expected_rows"), int) or int(vidore["expected_rows"]) <= 0:
        raise ArXivQAPrepareError("ArXivQA source lock has an invalid expected ViDoRe row count.")
    version_policy = payload.get("arxiv_version_policy")
    if not isinstance(version_policy, dict) or not isinstance(version_policy.get("limitation"), str) or not str(
        version_policy.get("limitation") or ""
    ).strip():
        raise ArXivQAPrepareError("ArXivQA source lock lacks an explicit arxiv_version_policy limitation.")
    pinning = str(version_policy.get("pdf_version_pinning") or "")
    if pinning not in _ALLOWED_PDF_PINNING:
        raise ArXivQAPrepareError(f"ArXivQA source lock has unsupported pdf_version_pinning {pinning!r}.")
    overrides = payload.get("pdf_overrides")
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ArXivQAPrepareError("ArXivQA source lock pdf_overrides must be a JSON object.")
    _validate_pdf_override_schema(overrides)
    if overrides and pinning != "documented_overrides_only":
        raise ArXivQAPrepareError(
            "ArXivQA source lock records pdf_overrides but pdf_version_pinning does not document them."
        )
    if pinning == "documented_overrides_only" and not overrides:
        raise ArXivQAPrepareError(
            "ArXivQA source lock claims documented_overrides_only but records no pdf_overrides."
        )
    return payload


def verify_source_inputs(parquet_path: Path, jsonl_path: Path, lock: dict[str, Any]) -> None:
    official = lock["official_source"]
    vidore = lock["vidore_source"]
    checks = (
        ("official JSONL", jsonl_path, official),
        ("ViDoRe parquet", parquet_path, vidore),
    )
    for label, path, source in checks:
        if path.name != source["filename"]:
            raise ArXivQAPrepareError(
                f"{label} filename {path.name!r} does not match locked filename {source['filename']!r}."
            )
        if not path.is_file():
            raise ArXivQAPrepareError(f"Locked {label} is missing: {path}")
        actual = sha256_file(path)
        if actual != source["sha256"]:
            raise ArXivQAPrepareError(
                f"{label} SHA-256 does not match the committed source lock: {actual} != {source['sha256']}."
            )


def load_verified_staged(
    staging_manifest_path: Path,
    parquet_path: Path,
    jsonl_path: Path,
    *,
    source_lock_path: Path | None = None,
) -> list[StagedRow]:
    """Reload staging only after re-checking parquet/JSONL hashes and the row fingerprint."""
    lock = load_source_lock(source_lock_path)
    verify_source_inputs(parquet_path, jsonl_path, lock)
    payload = read_json(staging_manifest_path)
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ArXivQAPrepareError(f"Staging manifest has no usable rows: {staging_manifest_path}")
    try:
        staged = [StagedRow(**row) for row in rows]
    except TypeError as exc:
        raise ArXivQAPrepareError(f"Staging manifest rows are malformed: {exc}") from exc
    fingerprint = staged_row_fingerprint(staged)
    expected_rows = int(lock["vidore_source"]["expected_rows"])
    if len(staged) != expected_rows or fingerprint != lock["staged_row_fingerprint_sha256"]:
        raise ArXivQAPrepareError(
            "Reloaded staging rows do not match the committed source lock fingerprint/row count."
        )
    return staged


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArXivQAPrepareError(f"Could not read JSON {path}: {exc}") from exc


def _rel(path: Path, project_root: Path | None) -> str:
    if project_root is None:
        return path.as_posix()
    try:
        return to_relative_project_path(path, project_root)
    except OSError:
        return path.as_posix()


def _parse_options(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _is_option_placeholder(value: Any) -> bool:
    """True only for empty or '-' placeholders ViDoRe appends; never for real option text."""
    return str(value).strip() in {"", "-"}


def normalize_options(value: Any) -> list[str]:
    """Drop only empty/'-' placeholders; preserve every real option string unchanged."""
    return [str(item) for item in _parse_options(value) if not _is_option_placeholder(item)]


def _content_signature(options: list[str], label: str, rationale: str | None) -> tuple[Any, ...]:
    return (tuple(normalize_options(options)), str(label).strip(), (rationale or "").strip())


def _answers_match(vidore_answer: Any, official_label: Any) -> bool:
    return str(vidore_answer).strip() == str(official_label).strip()


def parse_arxiv_identifiers(image_filename: str) -> ArxivIdentifiers:
    """Derive modern/legacy arXiv IDs and figure index from an ArXivQA image path."""
    text = str(image_filename).strip().replace("\\", "/")
    match = _IMAGE_NAME_RE.match(text)
    if match is None:
        raise ArXivQAPrepareError(
            f"Image filename {image_filename!r} does not match images/<arxiv_id>_<figure>.<ext>."
        )
    raw_id = match.group("raw_id")
    figure_index = int(match.group("figure_index"))
    modern_id: str | None = None
    legacy_id: str | None = None

    modern = _MODERN_ID_RE.fullmatch(raw_id)
    legacy_slash = _LEGACY_SLASH_RE.fullmatch(raw_id)
    legacy_compact = _LEGACY_COMPACT_RE.fullmatch(raw_id)

    matches = [item for item in (modern, legacy_slash, legacy_compact) if item is not None]
    if len(matches) != 1:
        raise ArXivQAPrepareError(
            f"Ambiguous or unsupported arXiv id in image filename {image_filename!r}."
        )

    if modern is not None:
        modern_id = modern.group("id")
        paper_id = modern_id
    elif legacy_slash is not None:
        legacy_id = f"{legacy_slash.group('archive')}/{legacy_slash.group('num')}"
        paper_id = legacy_id
    else:
        assert legacy_compact is not None
        legacy_id = f"{legacy_compact.group('archive')}/{legacy_compact.group('num')}"
        paper_id = legacy_id

    return ArxivIdentifiers(
        raw_id=raw_id,
        paper_id=paper_id,
        modern_id=modern_id,
        legacy_id=legacy_id,
        figure_index=figure_index,
        image_filename=text,
    )


def arxiv_pdf_url(paper_id: str) -> str:
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def load_official_jsonl(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ArXivQAPrepareError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ArXivQAPrepareError(f"JSONL row {line_no} must be an object.")
                image = str(row.get("image") or "").strip()
                question = str(row.get("question") or "").strip()
                if not image or not question:
                    raise ArXivQAPrepareError(
                        f"JSONL row {line_no} needs non-empty image and question fields."
                    )
                index.setdefault((image, question), []).append(row)
    except OSError as exc:
        raise ArXivQAPrepareError(f"Could not read official JSONL {path}: {exc}") from exc
    if not index:
        raise ArXivQAPrepareError(f"Official JSONL {path} contained no usable rows.")
    return index


def load_vidore_parquet(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # pandas/pyarrow errors vary by backend
        raise ArXivQAPrepareError(f"Could not read ViDoRe parquet {path}: {exc}") from exc
    required = {"query", "image_filename", "answer", "image"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ArXivQAPrepareError(f"ViDoRe parquet missing columns: {', '.join(missing)}.")
    return frame


def stage_vidore_against_official(
    parquet_path: Path,
    jsonl_path: Path,
    *,
    source_lock_path: Path | None = None,
    expected_rows: int | None = None,
) -> list[StagedRow]:
    """Stage ViDoRe rows against official JSONL.

    Proof requires exact image filename + question, matching answer/label, and
    equal option lists after dropping only empty/'-' placeholders.
    """
    lock = load_source_lock(source_lock_path)
    verify_source_inputs(parquet_path, jsonl_path, lock)
    locked_rows = int(lock["vidore_source"]["expected_rows"])
    if expected_rows is not None and expected_rows != locked_rows:
        raise ArXivQAPrepareError(
            f"Requested expected row count {expected_rows} conflicts with locked count {locked_rows}."
        )
    frame = load_vidore_parquet(parquet_path)
    if len(frame) != locked_rows:
        raise ArXivQAPrepareError(
            f"ViDoRe parquet has {len(frame)} rows; expected exactly {locked_rows}."
        )
    official = load_official_jsonl(jsonl_path)
    seen_keys: set[tuple[str, str]] = set()
    staged: list[StagedRow] = []

    for vidore_index, (_, row) in enumerate(frame.iterrows()):
        image_filename = str(row["image_filename"]).strip().replace("\\", "/")
        question = str(row["query"]).strip()
        key = (image_filename, question)
        if key in seen_keys:
            raise ArXivQAPrepareError(
                f"Duplicate ViDoRe row for image+question {image_filename!r}."
            )
        seen_keys.add(key)

        matches = official.get(key) or []
        if not matches:
            raise ArXivQAPrepareError(
                f"No official JSONL match for image+question {image_filename!r}."
            )

        signatures = {
            _content_signature(_parse_options(item.get("options")), str(item.get("label") or ""), item.get("rationale"))
            for item in matches
        }
        if len(signatures) != 1:
            raise ArXivQAPrepareError(
                f"Ambiguous official content for image+question {image_filename!r}."
            )

        official_ids = sorted({str(item.get("id") or "").strip() for item in matches if str(item.get("id") or "").strip()})
        if not official_ids:
            raise ArXivQAPrepareError(
                f"Official match for {image_filename!r} has no usable id."
            )
        primary = matches[0]
        for item in matches:
            if str(item.get("id") or "").strip() == official_ids[0]:
                primary = item
                break

        vidore_answer = str(row["answer"]).strip()
        official_label = str(primary.get("label") or "").strip()
        if not _answers_match(vidore_answer, official_label):
            raise ArXivQAPrepareError(
                f"Answer/label mismatch for image+question {image_filename!r}: "
                f"vidore={vidore_answer!r} official={official_label!r}."
            )

        vidore_options = normalize_options(row.get("options") if "options" in frame.columns else [])
        official_options = normalize_options(primary.get("options"))
        if vidore_options != official_options:
            raise ArXivQAPrepareError(
                f"Option list mismatch for image+question {image_filename!r} "
                f"after placeholder normalization."
            )

        identifiers = parse_arxiv_identifiers(image_filename)
        image_obj = row["image"]
        image_bytes: bytes | None = None
        if isinstance(image_obj, dict) and isinstance(image_obj.get("bytes"), (bytes, bytearray)):
            image_bytes = bytes(image_obj["bytes"])
        elif isinstance(image_obj, (bytes, bytearray)):
            image_bytes = bytes(image_obj)

        staged.append(
            StagedRow(
                qa_id=official_ids[0],
                question=question,
                answer=vidore_answer,
                options=vidore_options,
                image_filename=image_filename,
                figure_index=identifiers.figure_index,
                paper_id=identifiers.paper_id,
                modern_id=identifiers.modern_id,
                legacy_id=identifiers.legacy_id,
                official_id=official_ids[0],
                official_id_aliases=official_ids[1:],
                vidore_index=vidore_index,
                rationale=(str(primary.get("rationale")).strip() if primary.get("rationale") else None),
                image_sha256=_sha256_bytes(image_bytes) if image_bytes else None,
            )
        )

    staged.sort(key=lambda item: (item.paper_id, item.figure_index, item.qa_id))
    actual_fingerprint = staged_row_fingerprint(staged)
    if actual_fingerprint != lock["staged_row_fingerprint_sha256"]:
        raise ArXivQAPrepareError(
            "Staged rows do not match the committed source lock fingerprint: "
            f"{actual_fingerprint} != {lock['staged_row_fingerprint_sha256']}."
        )
    return staged


def write_staging_manifest(
    staged: list[StagedRow],
    output: Path,
    *,
    parquet_path: Path,
    jsonl_path: Path,
    project_root: Path | None = None,
    source_lock_path: Path | None = None,
) -> dict[str, Any]:
    lock = load_source_lock(source_lock_path)
    verify_source_inputs(parquet_path, jsonl_path, lock)
    fingerprint = staged_row_fingerprint(staged)
    if fingerprint != lock["staged_row_fingerprint_sha256"]:
        raise ArXivQAPrepareError("Cannot write a staging manifest whose rows do not match the committed source lock.")
    papers = sorted({row.paper_id for row in staged})
    payload = {
        "dataset": "arxivqa",
        "subset": "vidore_arxivqa_test_subsampled",
        "row_count": len(staged),
        "paper_count": len(papers),
        "match_key": ["image_filename", "question", "answer", "normalized_options"],
        "provenance": {
            "method": "exact_image_question_answer_options",
            "option_normalization": "drop_empty_and_dash_placeholders_only",
            "vidore_parquet": _rel(parquet_path, project_root),
            "official_jsonl": _rel(jsonl_path, project_root),
            "vidore_parquet_sha256": sha256_file(parquet_path) if parquet_path.is_file() else None,
            "official_jsonl_sha256": sha256_file(jsonl_path) if jsonl_path.is_file() else None,
            "staged_row_fingerprint_sha256": fingerprint,
            "staged_row_fingerprint_algorithm": STAGED_ROW_FINGERPRINT_ALGORITHM,
            "source_lock": _rel(source_lock_path or default_source_lock_path(), project_root),
            "upstream_split": lock["upstream_split"],
            "supervisor_approved_project_role": lock["supervisor_approved_project_role"],
            "project_role_policy": lock["project_role_policy"],
            "arxiv_version_policy": lock["arxiv_version_policy"],
            "note": (
                "Fixed ViDoRe upstream test subset staged against MMInstruction/ArxivQA for a "
                "supervisor-approved project validation remap. It is not an upstream validation split."
            ),
            "created_at_utc": _utc_now(),
        },
        "rows": [row.to_dict() for row in staged],
    }
    write_json(output, payload)
    return payload


def extract_figures_from_parquet(
    parquet_path: Path,
    staged: list[StagedRow],
    figures_dir: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    frame = load_vidore_parquet(parquet_path)
    by_key = {(row.image_filename, row.question): row for row in staged}
    if len(by_key) != len(staged):
        raise ArXivQAPrepareError("Staged rows contain duplicate image_filename+question keys.")
    written: list[dict[str, Any]] = []
    covered: set[tuple[str, str]] = set()
    figures_dir.mkdir(parents=True, exist_ok=True)

    for index, row in frame.iterrows():
        image_filename = str(row["image_filename"]).strip().replace("\\", "/")
        question = str(row["query"]).strip()
        key = (image_filename, question)
        staged_row = by_key.get(key)
        if staged_row is None:
            raise ArXivQAPrepareError(
                f"Parquet row {image_filename!r} was not present in the staged subset."
            )
        if key in covered:
            raise ArXivQAPrepareError(
                f"Parquet repeats staged image+question {image_filename!r}; figure extraction must be 1:1."
            )
        image_obj = row["image"]
        if not isinstance(image_obj, dict) or not isinstance(image_obj.get("bytes"), (bytes, bytearray)):
            raise ArXivQAPrepareError(f"Parquet image bytes missing for {image_filename!r}.")
        payload = bytes(image_obj["bytes"])
        relative_name = Path(image_filename).name
        destination = figures_dir / staged_row.paper_id.replace("/", "__") / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or sha256_file(destination) != _sha256_bytes(payload):
            partial = destination.with_suffix(destination.suffix + ".partial")
            partial.write_bytes(payload)
            partial.replace(destination)
        covered.add(key)
        written.append(
            {
                "qa_id": staged_row.qa_id,
                "paper_id": staged_row.paper_id,
                "image_filename": image_filename,
                "question": question,
                "figure_path": _rel(destination, project_root),
                "sha256": _sha256_bytes(payload),
                "vidore_index": int(index) if isinstance(index, (int, np.integer)) else staged_row.vidore_index,
            }
        )

    missing = sorted(set(by_key) - covered)
    if missing or len(written) != len(staged):
        raise ArXivQAPrepareError(
            "Figure extraction must cover every staged row exactly; "
            f"missing={len(missing)} written={len(written)} staged={len(staged)}."
        )

    manifest = {
        "figure_count": len(written),
        "figures_dir": _rel(figures_dir, project_root),
        "figures": sorted(written, key=lambda item: (item["paper_id"], item["qa_id"])),
    }
    write_json(figures_dir / "figures_manifest.json", manifest)
    return manifest


def build_qa_source(
    staged: list[StagedRow],
    output: Path,
    *,
    parquet_path: Path,
    jsonl_path: Path,
    project_root: Path | None = None,
    source_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Write project-validation QA inputs while retaining upstream-test provenance."""
    lock = load_source_lock(source_lock_path)
    verify_source_inputs(parquet_path, jsonl_path, lock)
    fingerprint = staged_row_fingerprint(staged)
    if len(staged) != lock["vidore_source"]["expected_rows"] or fingerprint != lock["staged_row_fingerprint_sha256"]:
        raise ArXivQAPrepareError("Cannot create QA source from rows that do not match the committed source lock.")
    records = [
        {
            "id": row.qa_id,
            "qa_id": row.qa_id,
            "question": row.question,
            "answer": row.answer,
            "options": row.options,
            "image_filename": row.image_filename,
            "paper_id": row.paper_id,
            "figure_index": row.figure_index,
            "modern_id": row.modern_id,
            "legacy_id": row.legacy_id,
            "official_id": row.official_id,
            "official_id_aliases": row.official_id_aliases,
            "vidore_index": row.vidore_index,
            "rationale": row.rationale,
            "image_sha256": row.image_sha256,
            "metadata": {
                "official_id": row.official_id,
                "official_id_aliases": row.official_id_aliases,
                "modern_id": row.modern_id,
                "legacy_id": row.legacy_id,
                "figure_index": row.figure_index,
                "vidore_index": row.vidore_index,
                "rationale": row.rationale,
                "image_sha256": row.image_sha256,
            },
        }
        for row in staged
    ]
    recomputed = qa_rows_content_fingerprint(records)
    if recomputed != fingerprint:
        raise ArXivQAPrepareError("QA source content fingerprint diverged from the staged-row fingerprint.")
    payload: dict[str, Any] = {
        "remapping_inputs": {
            "role": "qa_source_preparation_only",
            "subset": "vidore_arxivqa_test_subsampled",
            "row_count": len(records),
            "staged_row_fingerprint_sha256": fingerprint,
            "staged_row_fingerprint_algorithm": STAGED_ROW_FINGERPRINT_ALGORITHM,
        },
        "split": "val",
        "upstream_split": lock["upstream_split"],
        "supervisor_approved_project_role": lock["supervisor_approved_project_role"],
        "project_role_policy": lock["project_role_policy"],
        "source_lock": {
            "official_source": lock["official_source"],
            "vidore_source": lock["vidore_source"],
            "staged_row_fingerprint_sha256": lock["staged_row_fingerprint_sha256"],
            "staged_row_fingerprint_algorithm": STAGED_ROW_FINGERPRINT_ALGORITHM,
            "arxiv_version_policy": lock["arxiv_version_policy"],
            **(
                {"pdf_overrides": dict(lock.get("pdf_overrides") or {})}
                if str(lock["arxiv_version_policy"].get("pdf_version_pinning") or "")
                == "documented_overrides_only"
                else {}
            ),
        },
        "provenance": {
            "source": "vidore_arxivqa_test_subsampled",
            "match_key": ["image_filename", "question", "answer", "normalized_options"],
            "option_normalization": "drop_empty_and_dash_placeholders_only",
            "vidore_parquet": _rel(parquet_path, project_root),
            "official_jsonl": _rel(jsonl_path, project_root),
            "vidore_parquet_sha256": sha256_file(parquet_path) if parquet_path.is_file() else None,
            "official_jsonl_sha256": sha256_file(jsonl_path) if jsonl_path.is_file() else None,
            "note": (
                "The upstream subset is test. Its use as project validation is allowed only by the "
                "recorded supervisor-approved project-role policy. This is preparation-only data, "
                "not a paper-ready manifest."
            ),
            "created_at_utc": _utc_now(),
        },
        "data": records,
    }
    payload["val"] = records
    write_json(output, payload)
    return payload


def build_pdf_download_plan(
    staged: list[StagedRow],
    output: Path,
    *,
    pdf_dir: Path,
    project_root: Path | None = None,
    source_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Write the deterministic PDF download plan, applying only explicit version overrides."""
    lock_path = (source_lock_path or default_source_lock_path()).resolve()
    lock = load_source_lock(lock_path)
    fingerprint = staged_row_fingerprint(staged)
    if fingerprint != lock["staged_row_fingerprint_sha256"]:
        raise ArXivQAPrepareError(
            "Cannot build a PDF download plan whose staged rows do not match the committed source lock."
        )
    overrides = dict(lock.get("pdf_overrides") or {})
    papers: dict[str, dict[str, Any]] = {}
    figure_hashes_by_paper: dict[str, set[str]] = {}
    for row in staged:
        entry = papers.setdefault(
            row.paper_id,
            {
                "paper_id": row.paper_id,
                "modern_id": row.modern_id,
                "legacy_id": row.legacy_id,
                "pdf_url": arxiv_pdf_url(row.paper_id),
                "destination": _rel(pdf_dir / f"{row.paper_id.replace('/', '__')}.pdf", project_root),
                "qa_ids": [],
            },
        )
        entry["qa_ids"].append(row.qa_id)
        if row.image_sha256:
            figure_hashes_by_paper.setdefault(row.paper_id, set()).add(str(row.image_sha256))
    stale = sorted(set(overrides) - set(papers))
    if stale:
        raise ArXivQAPrepareError(
            "Source lock pdf_overrides reference paper ids absent from the staged subset: "
            + ", ".join(stale)
        )
    plan_rows = sorted(papers.values(), key=lambda item: item["paper_id"])
    for item in plan_rows:
        item["qa_ids"] = sorted(set(item["qa_ids"]))
        override = overrides.get(item["paper_id"])
        if override is None:
            continue
        figure_sha = str(override["figure_sha256"])
        if figure_sha not in figure_hashes_by_paper.get(item["paper_id"], set()):
            raise ArXivQAPrepareError(
                f"Source lock pdf_override {item['paper_id']!r} figure_sha256 does not match "
                "any staged figure hash for that paper."
            )
        item["pdf_url"] = override["pdf_url"]
        item["pdf_override"] = dict(override)
    override_count = sum(1 for item in plan_rows if "pdf_override" in item)
    payload = {
        "paper_count": len(plan_rows),
        "download_delay_sec_default": DEFAULT_DOWNLOAD_DELAY_SEC,
        "pdf_version_pinning": lock["arxiv_version_policy"]["pdf_version_pinning"],
        "pdf_override_count": override_count,
        "provenance": {
            "created_at_utc": _utc_now(),
            "source_subset": "vidore_arxivqa_test_subsampled",
            "source_lock": _rel(lock_path, project_root),
            "source_lock_sha256": sha256_file(lock_path),
            "staged_row_fingerprint_sha256": fingerprint,
            "pdf_override_count": override_count,
        },
        "papers": plan_rows,
    }
    write_json(output, payload)
    return payload


def _pdf_override_evidence_matches(planned: Any, prior: Any) -> bool:
    planned_override = planned if isinstance(planned, dict) else None
    prior_override = prior if isinstance(prior, dict) else None
    if planned_override is None and prior_override is None:
        return True
    if planned_override is None or prior_override is None:
        return False
    return planned_override == prior_override


def _load_prior_download_results(pdf_dir: Path) -> dict[str, dict[str, Any]]:
    results_path = pdf_dir / "download_results.json"
    if not results_path.is_file():
        return {}
    try:
        payload = read_json(results_path)
    except ArXivQAPrepareError:
        return {}
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    prior: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        paper_id = str(item.get("paper_id") or "").strip()
        if paper_id:
            prior[paper_id] = item
    return prior


def _can_skip_existing_pdf(
    *,
    paper: dict[str, Any],
    url: str,
    destination: Path,
    prior: dict[str, Any] | None,
) -> tuple[bool, int | None, str | None]:
    """Skip only when prior results prove URL, on-disk SHA-256, and override evidence."""
    if prior is None or prior.get("status") not in {"downloaded", "skipped_existing"}:
        return False, None, None
    if str(prior.get("pdf_url") or "") != url:
        return False, None, None
    if not _pdf_override_evidence_matches(paper.get("pdf_override"), prior.get("pdf_override")):
        return False, None, None
    if not destination.is_file() or destination.stat().st_size <= 0:
        return False, None, None
    try:
        page_count = inspect_pdf(destination)
    except ArXivQAPrepareError:
        return False, None, None
    actual_sha = sha256_file(destination)
    if prior.get("pdf_sha256") != actual_sha:
        return False, None, None
    return True, page_count, actual_sha


def is_pdf_bytes(payload: bytes) -> bool:
    return payload.startswith(b"%PDF")


def inspect_pdf(path: Path) -> int:
    """Return the page count only for a readable, non-empty PDF."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise ArXivQAPrepareError(f"PDF is missing or empty: {path}")
    if not is_pdf_bytes(path.read_bytes()[:8]):
        raise ArXivQAPrepareError(f"PDF is missing the %PDF signature: {path}")
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        try:
            page_count = len(document)
        finally:
            document.close()
    except Exception as exc:
        raise ArXivQAPrepareError(f"PDF cannot be parsed: {path}: {exc}") from exc
    if page_count <= 0:
        raise ArXivQAPrepareError(f"PDF has zero pages: {path}")
    return page_count


def download_pdfs(
    plan: dict[str, Any],
    *,
    pdf_dir: Path,
    delay_sec: float = DEFAULT_DOWNLOAD_DELAY_SEC,
    fetcher: Callable[[str], bytes] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Resumable polite PDF download. Never marks success without a valid PDF signature."""
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _default_fetch
    prior_by_id = _load_prior_download_results(pdf_dir)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    papers = plan.get("papers")
    if not isinstance(papers, list) or not papers:
        raise ArXivQAPrepareError("PDF download plan has no papers.")

    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            raise ArXivQAPrepareError(f"PDF plan papers[{index}] must be an object.")
        paper_id = str(paper.get("paper_id") or "").strip()
        url = str(paper.get("pdf_url") or arxiv_pdf_url(paper_id))
        destination = pdf_dir / f"{paper_id.replace('/', '__')}.pdf"
        record: dict[str, Any] = {
            "paper_id": paper_id,
            "pdf_url": url,
            "destination": _rel(destination, project_root),
        }
        if isinstance(paper.get("pdf_override"), dict):
            record["pdf_override"] = dict(paper["pdf_override"])

        can_skip, page_count, pdf_sha = _can_skip_existing_pdf(
            paper=paper,
            url=url,
            destination=destination,
            prior=prior_by_id.get(paper_id),
        )
        if can_skip and page_count is not None and pdf_sha is not None:
            record.update(
                {
                    "status": "skipped_existing",
                    "pdf_sha256": pdf_sha,
                    "bytes": destination.stat().st_size,
                    "page_count": page_count,
                }
            )
            results.append(record)
            continue

        if destination.is_file():
            try:
                inspect_pdf(destination)
            except ArXivQAPrepareError:
                destination.unlink(missing_ok=True)

        try:
            if index > 0 and delay_sec > 0:
                sleep_fn(delay_sec)
            payload = fetch(url)
            if not isinstance(payload, (bytes, bytearray)) or not is_pdf_bytes(bytes(payload)):
                raise ArXivQAPrepareError("Downloaded bytes are not a PDF (missing %PDF signature).")
            partial = destination.with_suffix(destination.suffix + ".partial")
            partial.write_bytes(bytes(payload))
            page_count = inspect_pdf(partial)
            partial.replace(destination)
            record.update(
                {
                    "status": "downloaded",
                    "pdf_sha256": sha256_file(destination),
                    "bytes": destination.stat().st_size,
                    "page_count": page_count,
                }
            )
            results.append(record)
        except Exception as exc:  # network and validation failures are recorded, not fabricated
            failure = {
                **record,
                "status": "failed",
                "error": str(exc),
            }
            failures.append(failure)
            results.append(failure)

    summary = {
        "downloaded": sum(1 for item in results if item["status"] == "downloaded"),
        "skipped_existing": sum(1 for item in results if item["status"] == "skipped_existing"),
        "failed": len(failures),
        "results": results,
        "failures": failures,
        "updated_at_utc": _utc_now(),
    }
    write_json(pdf_dir / "download_results.json", summary)
    return summary


def _default_fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "faar-arxivqa-prepare/0.1 (research; polite retry-safe downloader)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise ArXivQAPrepareError(f"PDF download failed for {url}: {exc}") from exc


def render_all_pdf_pages(
    pdf_path: Path,
    images_dir: Path,
    *,
    paper_id: str,
    project_root: Path | None = None,
    render_fn: Callable[..., dict[str, Any]] = render_pdf_pages,
    scale: float = 2.0,
) -> dict[str, Any]:
    """Render every page of one PDF into the page-image inventory layout."""
    page_count = inspect_pdf(pdf_path)
    page_ids = list(range(page_count))

    doc_rel = paper_id.replace("/", "__")
    image_paths = {page_id: images_dir / page_image_name(doc_rel, page_id) for page_id in page_ids}
    metrics = render_fn(pdf_path, page_ids, image_paths, scale=scale)
    pages = []
    for page_id in page_ids:
        image_path = image_paths[page_id]
        if not image_path.is_file() or image_path.stat().st_size <= 0:
            raise ArXivQAPrepareError(f"Rendering failed to produce page image {page_id} for {paper_id!r}.")
        pages.append(
            {
                "page_id": page_id,
                "image_path": _rel(image_path, project_root),
                "image_sha256": sha256_file(image_path),
                "ocr_text_path": _rel(
                    images_dir.parent / "ocr" / page_ocr_name(doc_rel, page_id),
                    project_root,
                ),
            }
        )
    return {
        "paper_id": paper_id,
        "pdf_path": _rel(pdf_path, project_root),
        "pdf_sha256": sha256_file(pdf_path),
        "page_count": page_count,
        "pages": pages,
        "render_metrics": metrics,
    }


def build_paper_inventory(
    paper_renders: Iterable[dict[str, Any]],
    output: Path,
    *,
    project_root: Path | None = None,
    expected_paper_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    papers = sorted(list(paper_renders), key=lambda item: item["paper_id"])
    paper_ids = [str(paper.get("paper_id") or "") for paper in papers]
    if len(paper_ids) != len(set(paper_ids)) or any(not paper_id for paper_id in paper_ids):
        raise ArXivQAPrepareError("Paper inventory has missing or duplicate paper ids.")
    if expected_paper_ids is not None and set(paper_ids) != set(expected_paper_ids):
        missing = sorted(set(expected_paper_ids) - set(paper_ids))
        unexpected = sorted(set(paper_ids) - set(expected_paper_ids))
        raise ArXivQAPrepareError(
            f"Paper inventory is incomplete or unexpected (missing={missing}, unexpected={unexpected})."
        )

    normalized_papers: list[dict[str, Any]] = []
    for paper in papers:
        paper_id = str(paper["paper_id"])
        page_count = paper.get("page_count")
        if not isinstance(page_count, int) or page_count <= 0:
            raise ArXivQAPrepareError(f"Paper {paper_id!r} has no valid PDF page count.")
        pdf_path = Path(str(paper.get("pdf_path") or ""))
        resolved_pdf = pdf_path if pdf_path.is_absolute() else (project_root / pdf_path if project_root else pdf_path)
        actual_pdf_page_count = inspect_pdf(resolved_pdf)
        actual_pdf_sha = sha256_file(resolved_pdf)
        if page_count != actual_pdf_page_count or paper.get("pdf_sha256") != actual_pdf_sha:
            raise ArXivQAPrepareError(f"Paper {paper_id!r} PDF page count or SHA-256 does not match.")
        pages = paper.get("pages")
        if not isinstance(pages, list):
            raise ArXivQAPrepareError(f"Paper {paper_id!r} has no pages list.")
        ids = [page.get("page_id") for page in pages if isinstance(page, dict)]
        if ids != list(range(page_count)):
            raise ArXivQAPrepareError(
                f"Paper {paper_id!r} pages must be exactly 0..{page_count - 1}; found {ids}."
            )
        normalized_pages: list[dict[str, Any]] = []
        for page in pages:
            image_path = Path(str(page.get("image_path") or ""))
            resolved_image = image_path if image_path.is_absolute() else (project_root / image_path if project_root else image_path)
            if not resolved_image.is_file() or resolved_image.stat().st_size <= 0:
                raise ArXivQAPrepareError(f"Paper {paper_id!r} page {page['page_id']} image is missing.")
            actual_image_sha = sha256_file(resolved_image)
            if page.get("image_sha256") != actual_image_sha:
                raise ArXivQAPrepareError(f"Paper {paper_id!r} page {page['page_id']} image hash does not match.")
            normalized_pages.append(
                {
                    "page_id": page["page_id"],
                    "image_path": page["image_path"],
                    "image_sha256": actual_image_sha,
                    "ocr_text_path": page["ocr_text_path"],
                }
            )
        normalized_papers.append(
            {
                "paper_id": paper_id,
                "pdf_path": paper["pdf_path"],
                "pdf_sha256": actual_pdf_sha,
                "page_count": page_count,
                "pages": normalized_pages,
            }
        )
    payload = {
        "state": "preparation_only",
        "paper_ready": False,
        "ocr_readiness": {
            "state": "pending_pinned_got_ocr_provenance",
            "requirement": "Every page needs a pinned GOT-OCR file and matching per-page provenance before remapping.",
        },
        "papers": normalized_papers,
        "provenance": {
            "created_at_utc": _utc_now(),
            "note": (
                "Complete rendered PDF page inventory for retrieval preparation. This file is not paper-ready "
                "until every pinned GOT-OCR output and its provenance are verified."
            ),
            "project_root": str(project_root) if project_root is not None else None,
        },
    }
    write_json(output, payload)
    return payload


def finalize_inventory_ocr_readiness(
    inventory: dict[str, Any],
    *,
    project_root: Path,
    ocr_provenance_dir: Path,
) -> dict[str, Any]:
    """Verify every page's pinned GOT-OCR asset before allowing a full-paper remap."""
    if inventory.get("state") != "preparation_only":
        raise ArXivQAPrepareError("Only a preparation-only inventory can be finalized for OCR readiness.")
    try:
        locked = json.loads((project_root / "config/model_revisions.json").read_text(encoding="utf-8"))["models"]["got_ocr"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ArXivQAPrepareError("Could not load pinned GOT-OCR model evidence.") from exc
    papers = inventory.get("papers")
    if not isinstance(papers, list) or not papers:
        raise ArXivQAPrepareError("OCR readiness requires a non-empty paper inventory.")
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "")
        provenance_path = ocr_provenance_dir / f"{paper_id.replace('/', '__')}.json"
        provenance = read_json(provenance_path)
        if (
            provenance.get("paper_id") != paper_id
            or provenance.get("pdf_sha256") != paper.get("pdf_sha256")
            or provenance.get("got_ocr_repository") != locked.get("repository")
            or provenance.get("got_ocr_revision") != locked.get("revision")
        ):
            raise ArXivQAPrepareError(f"Pinned GOT-OCR provenance does not match paper {paper_id!r}.")
        provenance_pages = provenance.get("pages")
        if not isinstance(provenance_pages, dict):
            raise ArXivQAPrepareError(f"Pinned GOT-OCR provenance has no pages for {paper_id!r}.")
        for page in paper.get("pages") or []:
            page_id = str(page["page_id"])
            page_provenance = provenance_pages.get(page_id)
            ocr_path = project_root / str(page["ocr_text_path"])
            if not isinstance(page_provenance, dict) or not ocr_path.is_file() or ocr_path.stat().st_size <= 0:
                raise ArXivQAPrepareError(f"Pinned GOT-OCR output is missing for {paper_id!r} page {page_id}.")
            actual_ocr_sha = sha256_file(ocr_path)
            if page_provenance.get("ocr_sha256") != actual_ocr_sha:
                raise ArXivQAPrepareError(f"Pinned GOT-OCR hash does not match for {paper_id!r} page {page_id}.")
            page["ocr_sha256"] = actual_ocr_sha
        paper["got_ocr_provenance"] = _rel(provenance_path, project_root)
    inventory["state"] = "ocr_ready"
    inventory["paper_ready"] = False
    inventory["ocr_readiness"] = {
        "state": "verified_pinned_got_ocr",
        "repository": locked["repository"],
        "revision": locked["revision"],
    }
    return inventory


def _load_gray(path_or_bytes: Path | bytes) -> np.ndarray:
    from PIL import Image

    if isinstance(path_or_bytes, bytes):
        image = Image.open(io.BytesIO(path_or_bytes))
    else:
        image = Image.open(path_or_bytes)
    return np.asarray(image.convert("L"), dtype=np.float32)


def figure_page_similarity(figure_path: Path | bytes, page_path: Path | bytes) -> float:
    """Return [0, 1] confidence that the figure appears on the page via template match."""
    figure = _load_gray(figure_path)
    page = _load_gray(page_path)
    if figure.size == 0 or page.size == 0:
        return 0.0

    # Whole-page exact/near equality (offline fixtures and cropped page renders).
    if figure.shape == page.shape:
        f = figure.astype(np.float64)
        p = page.astype(np.float64)
        f_c = f - f.mean()
        p_c = p - p.mean()
        f_norm = float(np.linalg.norm(f_c))
        p_norm = float(np.linalg.norm(p_c))
        if f_norm <= 1e-8 and p_norm <= 1e-8:
            return 1.0 if abs(float(f.mean()) - float(p.mean())) <= 1.0 else 0.0
        if f_norm <= 1e-8 or p_norm <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(f_c.reshape(-1), p_c.reshape(-1)) / (f_norm * p_norm), 0.0, 1.0))

    # Figure is typically a crop inside a larger page.
    template = figure
    haystack = page
    if template.shape[0] > haystack.shape[0] or template.shape[1] > haystack.shape[1]:
        scale = min(haystack.shape[0] / template.shape[0], haystack.shape[1] / template.shape[1], 1.0)
        if scale < 1.0:
            from PIL import Image

            new_size = (
                max(1, int(template.shape[1] * scale)),
                max(1, int(template.shape[0] * scale)),
            )
            template = np.asarray(
                Image.fromarray(template.astype(np.uint8)).resize(new_size, Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
    if template.shape[0] > haystack.shape[0] or template.shape[1] > haystack.shape[1]:
        return 0.0

    try:
        import cv2

        result = cv2.matchTemplate(haystack, template, cv2.TM_CCOEFF_NORMED)
        return float(np.clip(result.max(), 0.0, 1.0))
    except Exception:
        from PIL import Image

        resized = np.asarray(
            Image.fromarray(template.astype(np.uint8)).resize(
                (haystack.shape[1], haystack.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float64,
        )
        page64 = haystack.astype(np.float64)
        r_c = resized - resized.mean()
        p_c = page64 - page64.mean()
        denom = float(np.linalg.norm(r_c) * np.linalg.norm(p_c))
        if denom <= 1e-8:
            return 0.0
        return float(np.clip(np.dot(r_c.reshape(-1), p_c.reshape(-1)) / denom, 0.0, 1.0))


def match_evidence_pages(
    staged: list[StagedRow],
    figures_manifest: dict[str, Any],
    paper_inventory: dict[str, Any],
    *,
    accept_score: float = DEFAULT_ACCEPT_SCORE,
    accept_margin: float = DEFAULT_ACCEPT_MARGIN,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Score figure-to-page candidates. Human review owns every final mapping."""
    figure_by_qa = {item["qa_id"]: item for item in figures_manifest.get("figures", [])}
    papers = {
        paper["paper_id"]: paper
        for paper in paper_inventory.get("papers", [])
        if isinstance(paper, dict)
    }
    reports: list[EvidenceCandidate] = []
    review_candidates: list[dict[str, Any]] = []

    for row in staged:
        figure = figure_by_qa.get(row.qa_id)
        paper = papers.get(row.paper_id)
        if figure is None or paper is None:
            reports.append(
                EvidenceCandidate(
                    qa_id=row.qa_id,
                    paper_id=row.paper_id,
                    figure_index=row.figure_index,
                    status="unresolved",
                    reason="missing_figure_or_paper_pages",
                )
            )
            continue

        figure_path = Path(figure["figure_path"])
        if project_root is not None and not figure_path.is_absolute():
            figure_path = project_root / figure_path
        try:
            figure_exists = figure_path.is_file()
        except OSError:
            figure_exists = False
        if not figure_exists:
            reports.append(
                EvidenceCandidate(
                    qa_id=row.qa_id,
                    paper_id=row.paper_id,
                    figure_index=row.figure_index,
                    status="unresolved",
                    reason="figure_missing",
                )
            )
            continue

        scored: list[dict[str, Any]] = []
        for page in paper.get("pages") or []:
            page_path = Path(page["image_path"])
            if project_root is not None and not page_path.is_absolute():
                page_path = project_root / page_path
            if not page_path.is_file():
                continue
            try:
                score = figure_page_similarity(figure_path, page_path)
            except Exception:
                continue
            scored.append({"page_id": int(page["page_id"]), "score": round(score, 6)})
        scored.sort(key=lambda item: (-item["score"], item["page_id"]))

        status = "unresolved"
        candidate_pages: list[int] | None = None
        reason = "no_page_scores"
        if scored:
            best = scored[0]
            second = scored[1]["score"] if len(scored) > 1 else -1.0
            if best["score"] >= accept_score and (best["score"] - second) >= accept_margin:
                status = "candidate_requires_human_confirmation"
                candidate_pages = [int(best["page_id"])]
                reason = "high_confidence_candidate_not_final_mapping"
            else:
                reason = "below_confidence_or_margin"

        candidate = EvidenceCandidate(
            qa_id=row.qa_id,
            paper_id=row.paper_id,
            figure_index=row.figure_index,
            status=status,
            candidates=scored[:5],
            candidate_evidence_page_ids=candidate_pages,
            reason=reason,
        )
        reports.append(candidate)
        if status == "candidate_requires_human_confirmation" and candidate_pages is not None:
            review_candidates.append(
                {
                    "qa_id": row.qa_id,
                    "paper_id": row.paper_id,
                    "candidate_evidence_page_ids": candidate_pages,
                    "figure_id": row.image_filename,
                    "confidence": scored[0]["score"] if scored else None,
                    "method": "figure_page_template_match",
                    "requires_human_confirmation": True,
                }
            )

    payload = {
        "accept_score": accept_score,
        "accept_margin": accept_margin,
        "candidate_count": len(review_candidates),
        "unresolved_count": sum(1 for item in reports if item.status == "unresolved"),
        "reports": [item.to_dict() for item in reports],
        "review_candidates": review_candidates,
        "provenance": {
            "created_at_utc": _utc_now(),
            "note": (
                "This output contains confidence-scored candidates only, never final mappings. "
                "A human must explicitly confirm every QA-to-paper/page mapping before remapping. "
                "Retrieval corpora come from complete paper inventories, never evidence-only pages."
            ),
        },
    }
    return payload


def run_prepare_pipeline(
    *,
    parquet_path: Path,
    jsonl_path: Path,
    out_root: Path,
    project_root: Path | None = None,
    source_lock_path: Path | None = None,
    expected_rows: int | None = None,
    download: bool = False,
    render: bool = False,
    match: bool = False,
    delay_sec: float = DEFAULT_DOWNLOAD_DELAY_SEC,
    fetcher: Callable[[str], bytes] | None = None,
    render_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage, extract, and plan. Optional download/render/match stay explicit."""
    out_root.mkdir(parents=True, exist_ok=True)
    staged = stage_vidore_against_official(
        parquet_path, jsonl_path, source_lock_path=source_lock_path, expected_rows=expected_rows
    )
    staging_path = out_root / "staging_manifest.json"
    write_staging_manifest(
        staged,
        staging_path,
        parquet_path=parquet_path,
        jsonl_path=jsonl_path,
        project_root=project_root,
        source_lock_path=source_lock_path,
    )
    figures_dir = out_root / "figures"
    figures_manifest = extract_figures_from_parquet(
        parquet_path, staged, figures_dir, project_root=project_root
    )
    qa_path = out_root / "qa_source.json"
    build_qa_source(
        staged,
        qa_path,
        parquet_path=parquet_path,
        jsonl_path=jsonl_path,
        project_root=project_root,
        source_lock_path=source_lock_path,
    )
    pdf_dir = out_root / "pdfs"
    plan_path = out_root / "pdf_download_plan.json"
    plan = build_pdf_download_plan(
        staged, plan_path, pdf_dir=pdf_dir, project_root=project_root, source_lock_path=source_lock_path
    )

    summary: dict[str, Any] = {
        "row_count": len(staged),
        "paper_count": plan["paper_count"],
        "staging_manifest": _rel(staging_path, project_root),
        "qa_source": _rel(qa_path, project_root),
        "pdf_download_plan": _rel(plan_path, project_root),
        "figures_manifest": _rel(figures_dir / "figures_manifest.json", project_root),
    }

    if download:
        summary["download"] = download_pdfs(
            plan,
            pdf_dir=pdf_dir,
            delay_sec=delay_sec,
            fetcher=fetcher,
            project_root=project_root,
        )

    if render:
        paper_renders: list[dict[str, Any]] = []
        for paper in plan["papers"]:
            paper_id = paper["paper_id"]
            pdf_path = pdf_dir / f"{paper_id.replace('/', '__')}.pdf"
            if not pdf_path.is_file():
                continue
            paper_renders.append(
                render_all_pdf_pages(
                    pdf_path,
                    out_root / "images",
                    paper_id=paper_id,
                    project_root=project_root,
                    render_fn=render_fn or render_pdf_pages,
                )
            )
        inventory = build_paper_inventory(
            paper_renders,
            out_root / "paper_inventory.json",
            project_root=project_root,
            expected_paper_ids=[str(paper["paper_id"]) for paper in plan["papers"]],
        )
        summary["paper_inventory"] = _rel(out_root / "paper_inventory.json", project_root)
        summary["rendered_papers"] = len(paper_renders)

        if match:
            evidence = match_evidence_pages(
                staged,
                figures_manifest,
                inventory,
                project_root=project_root,
            )
            evidence_path = out_root / "evidence_candidates.json"
            write_json(evidence_path, evidence)
            summary["evidence_candidates"] = _rel(evidence_path, project_root)
            summary["human_review_required"] = True
            summary["candidate_mapping_count"] = evidence["candidate_count"]
            summary["unresolved_mapping_count"] = evidence["unresolved_count"]

    write_json(out_root / "prepare_summary.json", summary)
    return summary
