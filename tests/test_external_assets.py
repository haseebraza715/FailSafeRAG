from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from faar.arxivqa_prepare import (
    STAGED_ROW_FINGERPRINT_ALGORITHM,
    fingerprint_staged_row_dicts,
    staged_row_dict_from_qa_record,
)
from faar.arxivqa_remap import ArXivQARemapError, build_full_paper_source
from faar.asset_preparation import sha256_file
from faar.benchmarks import BenchmarkRepository
from faar.external_assets import ExternalAssetError, build_external_asset_manifest
from faar.settings import RetrievalSettings


def _asset_files(root: Path, pages: range = range(1, 3)) -> None:
    (root / "assets").mkdir(exist_ok=True)
    for page in pages:
        (root / "assets" / f"page-{page}.png").write_bytes(b"png")
        (root / "assets" / f"page-{page}.txt").write_text(f"OCR page {page}")


def _documents_inventory(doc_id: str, pages: range) -> list[dict]:
    return [
        {
            "doc_id": doc_id,
            "pages": [
                {
                    "page_id": page,
                    "image_path": f"assets/page-{page}.png",
                    "ocr_text_path": f"assets/page-{page}.txt",
                }
                for page in pages
            ],
        }
    ]


def _write_got_ocr_lock(project_root: Path, *, repository: str = "fixture/got-ocr", revision: str = "fixture-locked") -> None:
    config = project_root / "config"
    config.mkdir(exist_ok=True)
    config.joinpath("model_revisions.json").write_text(
        json.dumps({"models": {"got_ocr": {"repository": repository, "revision": revision}}}),
        encoding="utf-8",
    )


def _locked_qa_source(rows: list[dict], *, expected_rows: int) -> dict:
    enriched = []
    for index, row in enumerate(rows):
        paper_id = str(row.get("paper_id") or "paper-1")
        qa_id = str(row.get("id") or row.get("qa_id") or f"q{index}")
        enriched.append(
            {
                "id": qa_id,
                "qa_id": qa_id,
                "question": row["question"],
                "answer": row["answer"],
                "options": row.get("options") or [],
                "image_filename": row.get("image_filename") or f"images/{paper_id}_0.jpg",
                "paper_id": paper_id,
                "figure_index": int(row.get("figure_index", 0)),
                "modern_id": row.get("modern_id", paper_id if paper_id[0].isdigit() else None),
                "legacy_id": row.get("legacy_id"),
                "official_id": qa_id,
                "official_id_aliases": [],
                "vidore_index": int(row.get("vidore_index", index)),
                "rationale": row.get("rationale"),
                "image_sha256": row.get("image_sha256") or ("a" * 64),
                "metadata": {
                    "official_id": qa_id,
                    "official_id_aliases": [],
                    "modern_id": row.get("modern_id", paper_id if paper_id[0].isdigit() else None),
                    "legacy_id": row.get("legacy_id"),
                    "figure_index": int(row.get("figure_index", 0)),
                    "vidore_index": int(row.get("vidore_index", index)),
                    "rationale": row.get("rationale"),
                    "image_sha256": row.get("image_sha256") or ("a" * 64),
                },
            }
        )
    fingerprint = fingerprint_staged_row_dicts(staged_row_dict_from_qa_record(row) for row in enriched)
    return {
        "remapping_inputs": {
            "row_count": len(enriched),
            "staged_row_fingerprint_sha256": fingerprint,
            "staged_row_fingerprint_algorithm": STAGED_ROW_FINGERPRINT_ALGORITHM,
        },
        "split": "val",
        "upstream_split": "test",
        "supervisor_approved_project_role": "validation",
        "project_role_policy": {
            "policy_id": "fixture-policy",
            "allows_upstream_test_as_project_validation": True,
        },
        "source_lock": {
            "official_source": {
                "repository": "fixture/official",
                "revision": "fixture-revision",
                "filename": "arxivqa.jsonl",
                "sha256": "e" * 64,
                "upstream_split": "test",
            },
            "vidore_source": {
                "repository": "fixture/vidore",
                "revision": "fixture-revision",
                "filename": "test-00000-of-00001.parquet",
                "sha256": "d" * 64,
                "expected_rows": expected_rows,
            },
            "staged_row_fingerprint_sha256": fingerprint,
            "staged_row_fingerprint_algorithm": STAGED_ROW_FINGERPRINT_ALGORITHM,
            "arxiv_version_policy": {
                "source_filenames_include_versions": False,
                "pdf_version_pinning": "not_claimed",
                "limitation": "fixture",
            },
        },
        "val": enriched,
        "data": enriched,
    }


def _ocr_ready_inventory(pages: range, project_root: Path, *, paper_id: str = "paper-1") -> dict:
    _write_got_ocr_lock(project_root)
    pdf_path = project_root / "pdfs" / f"{paper_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    # Build a multi-page PDF by appending pages via Pillow save_all when needed.
    images = [Image.new("RGB", (32, 32), (255, 255, 255)) for _ in pages]
    if len(images) == 1:
        images[0].save(pdf_path, format="PDF")
    else:
        images[0].save(pdf_path, format="PDF", save_all=True, append_images=images[1:])
    page_entries = []
    provenance_pages = {}
    for page in pages:
        image_path = project_root / "assets" / f"page-{page}.png"
        ocr_path = project_root / "assets" / f"page-{page}.txt"
        if not image_path.is_file():
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")
        if not ocr_path.is_file():
            ocr_path.write_text(f"OCR page {page}", encoding="utf-8")
        page_entries.append(
            {
                "page_id": page,
                "image_path": f"assets/page-{page}.png",
                "image_sha256": sha256_file(image_path),
                "ocr_text_path": f"assets/page-{page}.txt",
                "ocr_sha256": sha256_file(ocr_path),
            }
        )
        provenance_pages[str(page)] = {"ocr_sha256": sha256_file(ocr_path)}
    provenance_dir = project_root / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    provenance_path = provenance_dir / f"{paper_id}.json"
    provenance_path.write_text(
        json.dumps(
            {
                "paper_id": paper_id,
                "pdf_sha256": sha256_file(pdf_path),
                "got_ocr_repository": "fixture/got-ocr",
                "got_ocr_revision": "fixture-locked",
                "pages": provenance_pages,
            }
        ),
        encoding="utf-8",
    )
    return {
        "state": "ocr_ready",
        "paper_ready": False,
        "ocr_readiness": {
            "state": "verified_pinned_got_ocr",
            "repository": "fixture/got-ocr",
            "revision": "fixture-locked",
        },
        "papers": [
            {
                "paper_id": paper_id,
                "pdf_path": f"pdfs/{paper_id}.pdf",
                "pdf_sha256": sha256_file(pdf_path),
                "page_count": len(page_entries),
                "pages": page_entries,
                "got_ocr_provenance": f"provenance/{paper_id}.json",
            }
        ],
    }


def _human_confirmed_mapping(qa_id: str, paper_id: str, evidence_page_ids: list[int]) -> dict:
    return {
        "data": [
            {
                "qa_id": qa_id,
                "paper_id": paper_id,
                "evidence_page_ids": evidence_page_ids,
                "human_confirmation": {
                    "confirmed_by": "reviewer-1",
                    "confirmed_at_utc": "2026-08-08T00:00:00Z",
                    "method": "human",
                },
            }
        ]
    }


def test_corpus_includes_all_document_pages_when_answer_is_on_page_two(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(1, 6))
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "documents": _documents_inventory("doc-a", range(1, 6)),
                "data": [
                    {
                        "question_id": "q1",
                        "query": "What is on page two?",
                        "answers": ["answer-page-2"],
                        "doc_id": "doc-a",
                        "evidence_page_no": 2,
                    }
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "MP-DocVQA", project_root=tmp_path)

    assert manifest["records"][0]["page_ids"] == [2]
    assert manifest["records"][0]["corpus_ids"] == [f"doc-a:p{page}" for page in range(1, 6)]
    assert [page["page_id"] for page in manifest["corpus_pages"]] == [1, 2, 3, 4, 5]
    assert all(not Path(page["image_path"]).is_absolute() for page in manifest["corpus_pages"])
    assert manifest["document_inventory"]["doc-a"] == [1, 2, 3, 4, 5]

    repository = BenchmarkRepository(
        manifest["records"],
        manifest["corpus_pages"],
        tmp_path,
        "mpdocvqa",
        "val",
        document_inventory=manifest["document_inventory"],
    )
    chunks = repository.get_corpus_chunks(RetrievalSettings(chunk_size_words=10, chunk_overlap_words=0))
    assert {chunk.page_id for chunk in chunks} == {1, 2, 3, 4, 5}


def test_rejects_evidence_only_source_without_document_inventory(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "data": [
                    {
                        "id": "q1",
                        "question": "What is the total?",
                        "answer": "42",
                        "document": "doc-a",
                        "page_no": 1,
                        "image_path": "assets/page-1.png",
                        "ocr_path": "assets/page-1.txt",
                    }
                ],
            }
        )
    )

    with pytest.raises(ExternalAssetError, match="Complete document pages cannot be proven"):
        build_external_asset_manifest(source, "MP-DocVQA", project_root=tmp_path)


def test_normalizes_aliases_with_complete_inventory(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "mpdocvqa.json"
    source.write_text(
        json.dumps(
            {
                "split": "val",
                "documents": _documents_inventory("doc-a", range(1, 3)),
                "data": [
                    {
                        "question_id": "q1",
                        "query": "What is the total?",
                        "answers": ["42", "forty two"],
                        "doc_id": "doc-a",
                        "evidence_page_no": 1,
                    },
                    {
                        "id": "q2",
                        "question": "Which document?",
                        "answer": "A",
                        "document_id": "doc-a",
                        "page": 1,
                    },
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "MP-DocVQA", project_root=tmp_path)

    assert manifest["dataset"] == "mpdocvqa"
    assert len(manifest["records"]) == 2
    assert manifest["records"][0]["correct_answer"] == "42"
    assert len(manifest["corpus_pages"]) == 2
    assert manifest["corpus_pages"][0]["image_path"] == "assets/page-1.png"


def test_normalizes_nested_arxiv_pages_with_inventory(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "arxivqa.json"
    source.write_text(
        json.dumps(
            {
                "val": [
                    {
                        "qa_id": "a1",
                        "questions": "Where is the result?",
                        "ground_truth": "Page two",
                        "document": {"id": "paper-7"},
                        "evidence_page_ids": [2],
                    }
                ],
                "documents": [
                    {
                        "doc_id": "paper-7",
                        "pages": [
                            {
                                "page_number": 1,
                                "page_image": "assets/page-1.png",
                                "ocr_file": "assets/page-1.txt",
                            },
                            {
                                "page_number": 2,
                                "page_image": "assets/page-2.png",
                                "ocr_file": "assets/page-2.txt",
                            },
                        ],
                    }
                ],
            }
        )
    )

    manifest = build_external_asset_manifest(source, "arxivqa", project_root=tmp_path)

    assert manifest["records"][0]["page_ids"] == [2]
    assert manifest["records"][0]["corpus_ids"] == ["paper-7:p1", "paper-7:p2"]
    assert len(manifest["corpus_pages"]) == 2


def test_full_paper_arxivqa_remap_keeps_all_pages_in_inventory(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(0, 3))
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            _locked_qa_source(
                [{"id": "q1", "question": "Where?", "answer": "page two", "paper_id": "paper-1"}],
                expected_rows=1,
            )
        )
    )
    inventory = tmp_path / "papers.json"
    inventory.write_text(json.dumps(_ocr_ready_inventory(range(0, 3), tmp_path)))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(_human_confirmed_mapping("q1", "paper-1", [1])))

    source = build_full_paper_source(qa, inventory, mapping, project_root=tmp_path)
    remapped = tmp_path / "remapped.json"
    remapped.write_text(json.dumps(source))
    manifest = build_external_asset_manifest(remapped, "arxivqa", project_root=tmp_path)
    assert source["remapping"]["method"] == "full_paper"
    assert source["upstream_split"] == "test"
    assert source["project_role_policy"]["allows_upstream_test_as_project_validation"] is True
    assert source["data"][0]["evidence_page_ids"] == [1]
    assert [page["page_id"] for page in source["documents"][0]["pages"]] == [0, 1, 2]
    assert manifest["records"][0]["page_ids"] == [1]
    assert [page["page_id"] for page in manifest["corpus_pages"]] == [0, 1, 2]


def test_full_paper_arxivqa_remap_rejects_missing_mapping(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(0, 1))
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            _locked_qa_source(
                [{"id": "q1", "question": "Where?", "answer": "x", "paper_id": "paper-1"}],
                expected_rows=1,
            )
        )
    )
    inventory = tmp_path / "papers.json"
    inventory.write_text(json.dumps(_ocr_ready_inventory(range(0, 1), tmp_path)))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"data": []}))
    with pytest.raises(ArXivQARemapError, match="figure mapping must contain"):
        build_full_paper_source(qa, inventory, mapping, project_root=tmp_path)


def test_full_paper_arxivqa_remap_rejects_forged_fingerprint_metadata(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(0, 1))
    payload = _locked_qa_source(
        [{"id": "q1", "question": "Where?", "answer": "x", "paper_id": "paper-1"}],
        expected_rows=1,
    )
    payload["data"][0]["question"] = "Tampered question?"
    payload["val"][0]["question"] = "Tampered question?"
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps(payload))
    inventory = tmp_path / "papers.json"
    inventory.write_text(json.dumps(_ocr_ready_inventory(range(0, 1), tmp_path)))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(_human_confirmed_mapping("q1", "paper-1", [0])))
    with pytest.raises(ArXivQARemapError, match="content fingerprint"):
        build_full_paper_source(qa, inventory, mapping, project_root=tmp_path)


def test_full_paper_arxivqa_remap_rejects_forged_ocr_ready_state(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(0, 1))
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps(
            _locked_qa_source(
                [{"id": "q1", "question": "Where?", "answer": "x", "paper_id": "paper-1"}],
                expected_rows=1,
            )
        )
    )
    inventory_payload = _ocr_ready_inventory(range(0, 1), tmp_path)
    inventory_payload["ocr_readiness"]["revision"] = "not-the-lock"
    inventory = tmp_path / "papers.json"
    inventory.write_text(json.dumps(inventory_payload))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps(_human_confirmed_mapping("q1", "paper-1", [0])))
    with pytest.raises(ArXivQARemapError, match="model_revisions"):
        build_full_paper_source(qa, inventory, mapping, project_root=tmp_path)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"split": "test", "documents": [], "data": [{"id": "q"}]}, "exactly the val split"),
        ({"data": [{"id": "q"}]}, "Could not verify the exact val split"),
    ],
)
def test_rejects_unverified_or_non_val_sources(tmp_path: Path, payload: dict, message: str) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload))
    with pytest.raises(ExternalAssetError, match=message):
        build_external_asset_manifest(source, "mpdocvqa", project_root=tmp_path)


def test_rejects_placeholders_and_missing_inventory_assets(tmp_path: Path) -> None:
    _asset_files(tmp_path)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "val": [
                    {
                        "id": "q1",
                        "question": "Question",
                        "answer": "Answer",
                        "document": "doc",
                        "evidence_page_no": 1,
                    }
                ],
                "documents": [
                    {
                        "doc_id": "doc",
                        "pages": [
                            {
                                "page_id": 1,
                                "image_path": "path/to/image.png",
                                "ocr_text_path": "assets/page-1.txt",
                            }
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(ExternalAssetError, match="placeholder image path"):
        build_external_asset_manifest(source, "mpdocvqa", project_root=tmp_path)


def test_cli_writes_portable_query_and_corpus_manifest(tmp_path: Path) -> None:
    _asset_files(tmp_path, range(1, 3))
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "val": [
                    {
                        "id": "q1",
                        "question": "Question",
                        "answer": "Answer",
                        "document": "doc",
                        "evidence_page_no": 1,
                    }
                ],
                "documents": _documents_inventory("doc", range(1, 3)),
            }
        )
    )
    out = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parents[1] / "scripts/data/register_external_assets.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset",
            "arxivqa",
            "--source",
            str(source),
            "--project-root",
            str(tmp_path),
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(out.read_text())
    assert payload["records"][0]["example_id"] == "q1"
    assert {page["corpus_id"] for page in payload["corpus_pages"]} == {"doc:p1", "doc:p2"}
    assert all(not Path(page["image_path"]).is_absolute() for page in payload["corpus_pages"])
