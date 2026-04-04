from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

from .settings import AppSettings
from .types import Phase0Example


def _parse_pages(raw: str) -> list[int]:
    raw = raw.strip()
    if raw.startswith("["):
        return [int(v) for v in ast.literal_eval(raw)]
    return [int(raw)]


def _parse_ocr_pages(text: str) -> dict[int, str]:
    pattern = re.compile(r"===== PAGE (\d+) =====\n")
    matches = list(pattern.finditer(text))
    if not matches:
        return {0: text.strip()}
    pages: dict[int, str] = {}
    for idx, match in enumerate(matches):
        page_id = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        pages[page_id] = text[start:end].strip()
    return pages


class Phase0Repository:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._summary = self._load_summary(settings.phase0_summary)
        self._manifest = self._load_manifest(settings.phase0_manifest)

    @staticmethod
    def _load_summary(path: Path | None) -> dict[str, dict]:
        if path is None or not path.exists():
            return {}
        payload = json.loads(path.read_text())
        return {item["example_id"]: item for item in payload.get("results", [])}

    @staticmethod
    def _load_manifest(path: Path | None) -> dict[str, dict]:
        if path is None or not path.exists():
            raise FileNotFoundError(f"Phase 0 manifest not found: {path}")
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {row["example_id"]: row for row in rows}

    def list_example_ids(self) -> list[str]:
        return sorted(self._manifest)

    def get_example(self, example_id: str) -> Phase0Example:
        row = self._manifest[example_id]
        summary = self._summary.get(example_id, {})
        ocr_text_path = Path(summary.get("ocr_text_path", self.settings.phase0_ocr_dir / f"{example_id}.txt"))
        gt_text_path = Path(summary["gt_text_path"]) if summary.get("gt_text_path") else None
        image_paths = [Path(p) for p in summary.get("image_paths", [])]
        ocr_text = ocr_text_path.read_text(errors="ignore")
        return Phase0Example(
            example_id=example_id,
            doc_name=row["doc_name"],
            question=row["question"],
            correct_answer=row["correct_answer"],
            page_ids=_parse_pages(row["page_no"]),
            ocr_text=ocr_text,
            ocr_text_path=ocr_text_path,
            gt_text_path=gt_text_path,
            image_paths=image_paths,
            metadata={
                "doc_type": row.get("doc_type", ""),
                "evidence_source": row.get("evidence_source", ""),
                "page_texts": _parse_ocr_pages(ocr_text),
            },
        )
