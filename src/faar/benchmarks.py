from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .asset_paths import AssetPathError, resolve_project_asset, to_relative_project_path
from .asset_preparation import sha256_file
from .chunking import build_page_chunks
from .data import DatasetUnavailableError
from .dataset_paths import SPLIT_RELATIVE_PATH
from .ohr_inventory import load_resolved_ohr_document_inventory
from .settings import RetrievalSettings
from .types import Phase0Example

_IMAGE_HASH_PROGRESS_THRESHOLD = 256
_IMAGE_HASH_PROGRESS_INTERVAL = 128


def _normalise_dataset(dataset: str) -> str:
    return dataset.lower().replace("-", "").replace("_", "")


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


class BenchmarkRepository:
    """Loads the complete, explicitly registered assets required for a paper run."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        corpus_pages: list[dict[str, Any]],
        project_root: Path,
        dataset: str,
        split: str,
        *,
        document_inventory: dict[str, list[int]] | None = None,
        manifest_sha256: str | None = None,
    ) -> None:
        self.project_root = project_root.expanduser().resolve()
        self.dataset = dataset
        self.split = split
        self.manifest_sha256 = manifest_sha256
        self._records = {str(record["example_id"]): record for record in records}
        self._corpus_pages = [self._normalise_corpus_page(page) for page in corpus_pages]
        self._verified_image_hashes: list[tuple[Path, str]] | None = None
        self._document_inventory = {
            str(doc_name): sorted({int(page_id) for page_id in page_ids})
            for doc_name, page_ids in (document_inventory or {}).items()
        }
        if not self._records:
            raise DatasetUnavailableError(f"No records were registered for {dataset} split {split}.")
        if not self._corpus_pages:
            raise DatasetUnavailableError(f"No shared corpus pages were registered for {dataset} split {split}.")
        self._validate_shared_corpus_integrity()
        self._validate_assets()

    def _normalise_corpus_page(self, page: dict[str, Any]) -> dict[str, Any]:
        normalised = dict(page)
        for key in ("ocr_text_path", "image_path"):
            value = normalised.get(key)
            if value:
                normalised[key] = str(resolve_project_asset(str(value), self.project_root))
        return normalised

    def _validate_shared_corpus_integrity(self) -> None:
        if not self._document_inventory:
            raise DatasetUnavailableError(
                f"{self.dataset} {self.split} cannot prove complete document pages. "
                "Manifest must include document_inventory separate from QA evidence pages."
            )
        pages_by_doc: dict[str, set[int]] = {}
        for page in self._corpus_pages:
            pages_by_doc.setdefault(str(page["doc_name"]), set()).add(int(page["page_id"]))

        for doc_name, expected_pages in self._document_inventory.items():
            actual = pages_by_doc.get(doc_name, set())
            expected = set(expected_pages)
            if actual != expected:
                raise DatasetUnavailableError(
                    f"{self.dataset} {self.split} corpus for {doc_name!r} is incomplete or evidence-biased. "
                    f"Expected pages {sorted(expected)}, found {sorted(actual)}."
                )

        for record in self._records.values():
            doc_name = str(record.get("doc_name", ""))
            evidence = {int(page_id) for page_id in record.get("page_ids", [])}
            inventory_pages = set(self._document_inventory.get(doc_name, []))
            if not evidence or not evidence.issubset(inventory_pages):
                raise DatasetUnavailableError(
                    f"Record {record.get('example_id')!r} evidence pages are not covered by document_inventory."
                )
            expected_corpus_ids = {f"{doc_name}:p{page_id}" for page_id in inventory_pages}
            if set(record.get("corpus_ids", [])) != expected_corpus_ids:
                raise DatasetUnavailableError(
                    f"Record {record.get('example_id')!r} corpus_ids must list every inventoried document page; "
                    "evidence pages must not control the retrieval corpus."
                )

    def _validate_assets(self) -> None:
        missing: list[str] = []
        for page in self._corpus_pages:
            page_id = str(page.get("corpus_id", "<missing corpus id>"))
            text_path = Path(page["ocr_text_path"]) if page.get("ocr_text_path") else None
            image_path = Path(page["image_path"]) if page.get("image_path") else None
            if not str(page.get("text", "")).strip() and (text_path is None or not text_path.exists()):
                missing.append(f"{page_id}: OCR text")
            if image_path is None or not image_path.exists():
                missing.append(f"{page_id}: page image")
        if missing:
            preview = "\n".join(missing[:10])
            raise DatasetUnavailableError(
                f"{self.dataset} {self.split} is not ready for a paper run. "
                "Register complete GOT-OCR text and page images for every selected document page.\n"
                f"Missing assets (first 10):\n{preview}"
            )

    def list_example_ids(self) -> list[str]:
        return sorted(self._records)

    def get_example(self, example_id: str) -> Phase0Example:
        record = self._records[example_id]
        return Phase0Example(
            example_id=example_id,
            doc_name=str(record.get("doc_name", "")),
            question=str(record["question"]),
            correct_answer=str(record["correct_answer"]),
            page_ids=[int(page_id) for page_id in record.get("page_ids", [])],
            ocr_text="",
            ocr_text_path=self.project_root / ".query-only",
            gt_text_path=None,
            image_paths=[],
            metadata=dict(record.get("metadata") or {}),
        )

    def get_corpus_chunks(self, settings: RetrievalSettings):
        chunks = []
        for page in self._corpus_pages:
            text = str(page.get("text", ""))
            if not text and page.get("ocr_text_path"):
                text = Path(page["ocr_text_path"]).read_text(errors="ignore")
            chunks.extend(
                build_page_chunks(
                    example_id=str(page["corpus_id"]),
                    doc_name=str(page["doc_name"]),
                    page_id=int(page["page_id"]),
                    page_text=text,
                    settings=settings,
                    image_path=str(page["image_path"]),
                )
            )
        if not chunks:
            raise DatasetUnavailableError(f"The registered {self.dataset} corpus produced no retrievable chunks.")
        return chunks

    def corpus_image_paths(self) -> list[Path]:
        return [Path(page["image_path"]) for page in self._corpus_pages]

    def _image_hash_mismatch_error(self, page: dict[str, Any], expected: str, actual: str) -> DatasetUnavailableError:
        page_id = str(page.get("corpus_id") or f"{page.get('doc_name')}:p{page.get('page_id')}")
        return DatasetUnavailableError(
            f"{self.dataset} {self.split} corpus image hash mismatch for {page_id} "
            f"({page['image_path']}): manifest records {expected}, file content hashes to {actual}. "
            "The image changed at the same path; re-render and update the locked asset manifest."
        )

    def corpus_image_hashes(self) -> list[tuple[Path, str]] | None:
        """Return the verified ordered per-page image SHA-256 mapping, or None to byte-hash.

        Recorded manifest hashes are claims, not facts. Every recorded hash is
        checked against the current image bytes by streaming each file before
        any retrieval or cache lookup. A mismatch (including a replaced image
        at the same path) fails closed so stale embeddings can never be reused.

        Policy: when the manifest records a hash for every corpus page, all of
        them are verified and the verified mapping is returned. If any page
        lacks an image_sha256 field, the manifest carries no authoritative
        identity and None is returned so callers fall back to hashing the
        bytes themselves (OHR and ArXivQA manifests). A present but malformed
        hash is a corrupt lock and fails closed.
        """
        if self._verified_image_hashes is not None:
            return self._verified_image_hashes
        recorded: list[tuple[dict[str, Any], Path, str]] = []
        for page in self._corpus_pages:
            raw = page.get("image_sha256")
            if raw is None:
                return None
            if not isinstance(raw, str):
                page_id = str(page.get("corpus_id") or f"{page.get('doc_name')}:p{page.get('page_id')}")
                raise DatasetUnavailableError(
                    f"{self.dataset} {self.split} corpus image {page_id} records a malformed "
                    f"image_sha256 ({raw!r}); expected a 64-character hexadecimal string."
                )
            sha = raw.strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                page_id = str(page.get("corpus_id") or f"{page.get('doc_name')}:p{page.get('page_id')}")
                raise DatasetUnavailableError(
                    f"{self.dataset} {self.split} corpus image {page_id} records a malformed "
                    f"image_sha256 ({raw!r}); expected a 64-character hexadecimal string."
                )
            recorded.append((page, Path(page["image_path"]), sha))
        total = len(recorded)
        if total >= _IMAGE_HASH_PROGRESS_THRESHOLD:
            print(
                f"[faar] verifying {total} recorded image hashes for {self.dataset} {self.split}...",
                flush=True,
            )
        verified: list[tuple[Path, str]] = []
        for index, (page, image_path, expected) in enumerate(recorded, start=1):
            try:
                actual = sha256_file(image_path)
            except OSError as exc:
                page_id = str(page.get("corpus_id") or f"{page.get('doc_name')}:p{page.get('page_id')}")
                raise DatasetUnavailableError(
                    f"{self.dataset} {self.split} corpus image {page_id} ({image_path}) cannot be "
                    f"hashed: {exc}. Expected manifest hash {expected}."
                ) from exc
            if actual != expected:
                raise self._image_hash_mismatch_error(page, expected, actual)
            verified.append((image_path, actual))
            if total >= _IMAGE_HASH_PROGRESS_THRESHOLD and (
                index % _IMAGE_HASH_PROGRESS_INTERVAL == 0 or index == total
            ):
                print(
                    f"[faar] verified image hashes {index}/{total} for {self.dataset} {self.split}",
                    flush=True,
                )
        self._verified_image_hashes = verified
        return verified

    def corpus_image_page_map(self) -> dict[Path, tuple[str, int]]:
        return {
            Path(page["image_path"]): (str(page["doc_name"]), int(page["page_id"]))
            for page in self._corpus_pages
        }


def load_benchmark_repository_from_manifest(
    project_root: Path,
    manifest_path: Path | str,
    *,
    dataset: str | None = None,
    split: str | None = None,
) -> BenchmarkRepository:
    """Load a BenchmarkRepository from an explicit manifest path (including smoke manifests)."""
    project_root = project_root.expanduser().resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise DatasetUnavailableError(f"Missing benchmark asset manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    corpus_pages = payload.get("corpus_pages") if isinstance(payload, dict) else None
    inventory = payload.get("document_inventory") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not isinstance(corpus_pages, list):
        raise DatasetUnavailableError(f"Asset manifest {manifest_path} must contain records and corpus_pages lists.")
    if isinstance(inventory, dict):
        document_inventory = {
            str(doc_name): [int(page_id) for page_id in page_ids]
            for doc_name, page_ids in inventory.items()
        }
    else:
        document_inventory = None
    dataset_key = _normalise_dataset(str(dataset or (payload.get("dataset") if isinstance(payload, dict) else None) or "unknown"))
    split_key = str(split or (payload.get("split") if isinstance(payload, dict) else None) or "unknown")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return BenchmarkRepository(
        records,
        corpus_pages,
        project_root,
        dataset_key,
        split_key,
        document_inventory=document_inventory,
        manifest_sha256=manifest_sha256,
    )


def load_benchmark_repository(project_root: Path, dataset: str, split: str) -> BenchmarkRepository:
    dataset_key = _normalise_dataset(dataset)
    project_root = project_root.expanduser().resolve()
    manifest_path = project_root / "data/benchmark_assets" / dataset_key / f"{split}.json"
    if not manifest_path.exists():
        raise DatasetUnavailableError(
            f"Missing benchmark asset manifest: {manifest_path}. "
            "Create it with scripts/data/register_benchmark_assets.py after GOT-OCR and page rendering are complete."
        )
    return load_benchmark_repository_from_manifest(
        project_root,
        manifest_path,
        dataset=dataset_key,
        split=split,
    )


def load_ohr_document_inventory(
    inventory_dir: Path,
    selected_documents: set[str],
) -> dict[str, list[int]]:
    """Load complete page ids per document from OHR retrieval_base-style JSON files."""
    inventory_dir = inventory_dir.expanduser().resolve()
    if not inventory_dir.is_dir():
        raise DatasetUnavailableError(f"OHR document inventory directory is missing: {inventory_dir}")

    inventory, _resolutions = load_resolved_ohr_document_inventory(inventory_dir, selected_documents)
    missing = sorted(selected_documents - set(inventory))
    if missing:
        preview = ", ".join(missing[:10])
        raise DatasetUnavailableError(
            "Complete document pages cannot be proven for OHR-Bench. Missing inventory files for: "
            f"{preview}"
        )
    return inventory


def build_ohr_asset_manifest(
    project_root: Path,
    split: str,
    ocr_dir: Path,
    image_dir: Path,
    *,
    document_inventory_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]] | dict[str, list[int]]]:
    project_root = project_root.expanduser().resolve()
    split_payload = json.loads((project_root / SPLIT_RELATIVE_PATH).read_text())
    if split not in split_payload["splits"]:
        raise ValueError(f"Unknown OHR-Bench split {split!r}.")
    selected_ids = set(split_payload["splits"][split])
    source_rows = json.loads((project_root / "OHR-Bench/data/qas_v2.json").read_text())
    source_by_id = {str(row["ID"]): row for row in source_rows}
    records: list[dict[str, Any]] = []
    for example_id in sorted(selected_ids):
        row = source_by_id[example_id]
        evidence_pages = [int(page) for page in _listify(row.get("evidence_page_no"))]
        records.append(
            {
                "example_id": example_id,
                "doc_name": row["doc_name"],
                "question": row["questions"],
                "correct_answer": row["answers"],
                "page_ids": evidence_pages,
                "corpus_ids": [],  # filled after inventory is known
                "metadata": {
                    "doc_type": row.get("doc_type", ""),
                    "evidence_source": row.get("evidence_source", ""),
                    "evidence_page_ids": evidence_pages,
                },
            }
        )
    selected_documents = {str(record["doc_name"]) for record in records}
    inventory_dir = document_inventory_dir or (project_root / "OHR-Bench/data/retrieval_base/gt")
    document_inventory = load_ohr_document_inventory(inventory_dir, selected_documents)
    corpus_pages = _load_ohr_corpus_pages(
        ocr_dir,
        image_dir,
        selected_documents,
        document_inventory,
        project_root,
    )
    for record in records:
        doc_name = str(record["doc_name"])
        page_ids = document_inventory[doc_name]
        evidence = {int(page_id) for page_id in record["page_ids"]}
        if not evidence.issubset(set(page_ids)):
            raise DatasetUnavailableError(
                f"OHR evidence pages for {record['example_id']} are outside the complete document inventory."
            )
        record["corpus_ids"] = [f"{doc_name}:p{page_id}" for page_id in page_ids]
    return {
        "records": records,
        "corpus_pages": corpus_pages,
        "document_inventory": document_inventory,
    }


def _load_ohr_corpus_pages(
    ocr_dir: Path,
    image_dir: Path,
    selected_documents: set[str],
    document_inventory: dict[str, list[int]],
    project_root: Path,
) -> list[dict[str, Any]]:
    ocr_dir = ocr_dir.expanduser().resolve()
    image_dir = image_dir.expanduser().resolve()
    pages_by_id: dict[str, dict[str, Any]] = {}

    for path in sorted(ocr_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("pages", payload.get("data", []))
        doc_name = str(path.relative_to(ocr_dir).with_suffix(""))
        if doc_name not in selected_documents:
            continue
        for row in rows:
            page_id = int(row.get("page_idx", row.get("page_no", row.get("page_id", 0))))
            text = str(row.get("text", ""))
            image_path = _find_page_image(image_dir, doc_name, page_id)
            corpus_id = f"{doc_name}:p{page_id}"
            try:
                image_rel = to_relative_project_path(image_path, project_root) if image_path else ""
            except AssetPathError as exc:
                raise DatasetUnavailableError(str(exc)) from exc
            pages_by_id[corpus_id] = {
                "corpus_id": corpus_id,
                "doc_name": doc_name,
                "page_id": page_id,
                "text": text,
                "image_path": image_rel,
            }

    for path in sorted(ocr_dir.rglob("*.txt")):
        relative = path.relative_to(ocr_dir)
        match = re.match(r"(.+?)(?:_page_|_p)(\d+)$", relative.stem)
        if not match:
            continue
        doc_name = str(relative.with_name(match.group(1)).with_suffix(""))
        if doc_name not in selected_documents:
            continue
        page_id = int(match.group(2))
        image_path = _find_page_image(image_dir, doc_name, page_id)
        corpus_id = f"{doc_name}:p{page_id}"
        try:
            ocr_rel = to_relative_project_path(path, project_root)
            image_rel = to_relative_project_path(image_path, project_root) if image_path else ""
        except AssetPathError as exc:
            raise DatasetUnavailableError(str(exc)) from exc
        page = {
            "corpus_id": corpus_id,
            "doc_name": doc_name,
            "page_id": page_id,
            "ocr_text_path": ocr_rel,
            "image_path": image_rel,
        }
        existing = pages_by_id.get(corpus_id)
        if existing is None or not str(existing.get("text", "")).strip():
            pages_by_id[corpus_id] = page

    # Evidence pages must never control membership: enforce the inventory exactly.
    missing: list[str] = []
    for doc_name, page_ids in document_inventory.items():
        for page_id in page_ids:
            corpus_id = f"{doc_name}:p{page_id}"
            page = pages_by_id.get(corpus_id)
            if page is None:
                missing.append(f"{corpus_id}: missing OCR/image assets")
                continue
            if not str(page.get("text", "")).strip() and not page.get("ocr_text_path"):
                missing.append(f"{corpus_id}: OCR text")
            if not page.get("image_path"):
                missing.append(f"{corpus_id}: page image")
    if missing:
        preview = "\n".join(missing[:10])
        raise DatasetUnavailableError(
            "OHR shared corpus is incomplete relative to the document inventory. "
            "Evidence-page assets alone are insufficient.\n"
            f"Missing (first 10):\n{preview}"
        )

    ordered: list[dict[str, Any]] = []
    for doc_name, page_ids in sorted(document_inventory.items()):
        for page_id in page_ids:
            ordered.append(pages_by_id[f"{doc_name}:p{page_id}"])
    return ordered


def _find_page_image(image_dir: Path, doc_name: str, page_id: int) -> Path | None:
    base = image_dir / doc_name
    candidates = [
        base.with_name(f"{base.name}_page_{page_id}.png"),
        base.with_name(f"{base.name}_page_{page_id}.jpg"),
        base.with_name(f"{base.name}_p{page_id}.png"),
        base.with_name(f"{base.name}_p{page_id}.jpg"),
        image_dir / f"{doc_name}_page_{page_id}.png",
        image_dir / f"{doc_name}_page_{page_id}.jpg",
        image_dir / f"{doc_name}_p{page_id}.png",
        image_dir / f"{doc_name}_p{page_id}.jpg",
    ]
    # Nested doc_name paths: images/academic/foo_page_1.png
    nested = image_dir / doc_name
    candidates.extend(
        [
            nested.parent / f"{nested.name}_page_{page_id}.png",
            nested.parent / f"{nested.name}_page_{page_id}.jpg",
            nested.parent / f"{nested.name}_p{page_id}.png",
            nested.parent / f"{nested.name}_p{page_id}.jpg",
        ]
    )
    return next((path for path in candidates if path.exists()), None)
