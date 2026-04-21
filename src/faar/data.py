from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
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
        self._manual_labels = self._load_manual_labels(settings.phase0_manual_labels)

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
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.DictReader(handle))
        return {row["example_id"]: row for row in rows}

    @staticmethod
    def _load_manual_labels(path: Path | None) -> dict[str, dict]:
        if path is None or not path.exists():
            return {}
        with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.DictReader(handle))
        return {row["example_id"]: row for row in rows}

    def list_example_ids(self) -> list[str]:
        return sorted(self._manifest)

    def get_example_record(self, example_id: str) -> dict[str, str]:
        row = dict(self._manifest[example_id])
        manual = self._manual_labels.get(example_id, {})
        row["manual_failure_type"] = manual.get("failure_type", "").strip()
        row["manual_failure_notes"] = manual.get("notes", "").strip()
        return row

    def select_example_ids(
        self,
        *,
        max_examples: int | None = None,
        doc_types: list[str] | None = None,
        evidence_sources: list[str] | None = None,
        manual_failure_types: list[str] | None = None,
        stratify_by: str | None = None,
        examples_per_stratum: int | None = None,
    ) -> list[str]:
        allowed_strata = {"doc_type", "evidence_source", "manual_failure_type"}
        if stratify_by and stratify_by not in allowed_strata:
            raise ValueError(
                f"Unsupported stratify_by={stratify_by!r}. Expected one of: {', '.join(sorted(allowed_strata))}."
            )

        doc_type_filter = {value.strip() for value in doc_types or [] if value.strip()}
        evidence_filter = {value.strip() for value in evidence_sources or [] if value.strip()}
        failure_filter = {value.strip() for value in manual_failure_types or [] if value.strip()}

        filtered: list[str] = []
        for example_id in self.list_example_ids():
            row = self.get_example_record(example_id)
            if doc_type_filter and row.get("doc_type", "").strip() not in doc_type_filter:
                continue
            if evidence_filter and row.get("evidence_source", "").strip() not in evidence_filter:
                continue
            if failure_filter and row.get("manual_failure_type", "").strip() not in failure_filter:
                continue
            filtered.append(example_id)

        if stratify_by:
            grouped: dict[str, list[str]] = defaultdict(list)
            for example_id in filtered:
                row = self.get_example_record(example_id)
                stratum = row.get(stratify_by, "").strip() or "unknown"
                grouped[stratum].append(example_id)
            balanced: list[str] = []
            for stratum in sorted(grouped):
                group_ids = sorted(grouped[stratum])
                take = len(group_ids) if examples_per_stratum is None else max(examples_per_stratum, 0)
                balanced.extend(group_ids[:take])
            filtered = balanced

        if max_examples is not None:
            filtered = filtered[: max(max_examples, 0)]
        return filtered

    def get_example(self, example_id: str) -> Phase0Example:
        row = self.get_example_record(example_id)
        summary = self._summary.get(example_id, {})
        ocr_text_path = Path(summary.get("ocr_text_path", self.settings.phase0_ocr_dir / f"{example_id}.txt"))
        if not ocr_text_path.exists():
            ocr_text_path = self.settings.phase0_ocr_dir / f"{example_id}.txt"
        gt_text_path = Path(summary["gt_text_path"]) if summary.get("gt_text_path") else None
        image_paths = [Path(p) for p in summary.get("image_paths", []) if Path(p).exists()]
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
                "manual_failure_type": row.get("manual_failure_type", ""),
                "manual_failure_notes": row.get("manual_failure_notes", ""),
                "page_texts": _parse_ocr_pages(ocr_text),
            },
        )
