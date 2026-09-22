from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pandas as pd
import pytest
from PIL import Image

from faar.arxivqa_prepare import (
    ArXivQAPrepareError,
    StagedRow,
    build_paper_inventory,
    build_pdf_download_plan,
    build_qa_source,
    download_pdfs,
    extract_figures_from_parquet,
    finalize_inventory_ocr_readiness,
    inspect_pdf,
    load_source_lock,
    load_verified_staged,
    match_evidence_pages,
    qa_rows_content_fingerprint,
    render_all_pdf_pages,
    stage_vidore_against_official,
    staged_row_fingerprint,
    write_staging_manifest,
)
from faar.arxivqa_remap import ArXivQARemapError, build_full_paper_source
from faar.asset_preparation import sha256_file


def _image_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), (220, 20, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _pdf_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PDF")
    return buffer.getvalue()


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lock_payload(
    jsonl: Path,
    parquet: Path,
    row: StagedRow,
    *,
    pdf_overrides: dict | None = None,
    pdf_version_pinning: str = "not_claimed",
) -> dict:
    payload = {
        "schema_version": 1,
        "official_source": {
            "repository": "fixture/official",
            "revision": "fixture-revision",
            "filename": jsonl.name,
            "sha256": sha256_file(jsonl),
            "upstream_split": "test",
        },
        "vidore_source": {
            "repository": "fixture/vidore",
            "revision": "fixture-revision",
            "filename": parquet.name,
            "sha256": sha256_file(parquet),
            "expected_rows": 1,
        },
        "staged_row_fingerprint_sha256": staged_row_fingerprint([row]),
        "staged_row_fingerprint_algorithm": "sha256(canonical-json(sorted staged StagedRow dictionaries))",
        "upstream_split": "test",
        "supervisor_approved_project_role": "validation",
        "project_role_policy": {
            "policy_id": "fixture-policy",
            "allows_upstream_test_as_project_validation": True,
        },
        "arxiv_version_policy": {
            "source_filenames_include_versions": False,
            "pdf_version_pinning": pdf_version_pinning,
            "limitation": "fixture",
        },
    }
    if pdf_overrides:
        payload["pdf_overrides"] = pdf_overrides
    return payload


def _write_fixture(
    tmp_path: Path,
    *,
    pdf_overrides: dict | None = None,
    pdf_version_pinning: str = "not_claimed",
) -> tuple[Path, Path, Path, StagedRow]:
    image = _image_bytes()
    parquet = tmp_path / "test-00000-of-00001.parquet"
    pd.DataFrame(
        [
            {
                "query": "What color is the figure?",
                "image": {"bytes": image, "path": None},
                "image_filename": "images/1810.10511_0.jpg",
                "options": str(["A. Red", "B. Blue"]),
                "answer": "A",
            }
        ]
    ).to_parquet(parquet)
    jsonl = tmp_path / "arxivqa.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "id": "q1",
                "image": "images/1810.10511_0.jpg",
                "question": "What color is the figure?",
                "options": ["A. Red", "B. Blue"],
                "label": "A",
                "rationale": "The figure is red.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    row = StagedRow(
        qa_id="q1",
        question="What color is the figure?",
        answer="A",
        options=["A. Red", "B. Blue"],
        image_filename="images/1810.10511_0.jpg",
        figure_index=0,
        paper_id="1810.10511",
        modern_id="1810.10511",
        legacy_id=None,
        official_id="q1",
        official_id_aliases=[],
        vidore_index=0,
        rationale="The figure is red.",
        image_sha256=_sha_bytes(image),
    )
    lock = tmp_path / "source-lock.json"
    lock.write_text(
        json.dumps(_lock_payload(jsonl, parquet, row, pdf_overrides=pdf_overrides, pdf_version_pinning=pdf_version_pinning)),
        encoding="utf-8",
    )
    return parquet, jsonl, lock, row


def _write_model_lock(tmp_path: Path, *, repository: str = "got", revision: str = "locked") -> None:
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    config.joinpath("model_revisions.json").write_text(
        json.dumps({"models": {"got_ocr": {"repository": repository, "revision": revision}}}),
        encoding="utf-8",
    )


def _make_inventory(tmp_path: Path) -> tuple[dict, Path]:
    pdf_path = tmp_path / "pdfs" / "1810.10511.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(_pdf_bytes())
    image_path = tmp_path / "images" / "1810.10511_page_0.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (220, 20, 60)).save(image_path)
    inventory = build_paper_inventory(
        [
            {
                "paper_id": "1810.10511",
                "pdf_path": "pdfs/1810.10511.pdf",
                "pdf_sha256": sha256_file(pdf_path),
                "page_count": 1,
                "pages": [
                    {
                        "page_id": 0,
                        "image_path": "images/1810.10511_page_0.png",
                        "image_sha256": sha256_file(image_path),
                        "ocr_text_path": "ocr/1810.10511_page_0.txt",
                    }
                ],
            }
        ],
        tmp_path / "paper_inventory.json",
        project_root=tmp_path,
        expected_paper_ids=["1810.10511"],
    )
    return inventory, tmp_path / "paper_inventory.json"


def _finalize_inventory(tmp_path: Path, inventory: dict) -> dict:
    _write_model_lock(tmp_path)
    ocr_path = tmp_path / "ocr" / "1810.10511_page_0.txt"
    ocr_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_path.write_text("OCR text\n", encoding="utf-8")
    provenance_dir = tmp_path / "provenance"
    provenance_dir.mkdir(exist_ok=True)
    provenance_dir.joinpath("1810.10511.json").write_text(
        json.dumps(
            {
                "paper_id": "1810.10511",
                "pdf_sha256": inventory["papers"][0]["pdf_sha256"],
                "got_ocr_repository": "got",
                "got_ocr_revision": "locked",
                "pages": {"0": {"ocr_sha256": sha256_file(ocr_path)}},
            }
        ),
        encoding="utf-8",
    )
    return finalize_inventory_ocr_readiness(
        inventory, project_root=tmp_path, ocr_provenance_dir=provenance_dir
    )


def _human_mapping(*, paper_id: str = "1810.10511") -> dict:
    return {
        "data": [
            {
                "qa_id": "q1",
                "paper_id": paper_id,
                "evidence_page_ids": [0],
                "human_confirmation": {
                    "confirmed_by": "reviewer-1",
                    "confirmed_at_utc": "2026-08-08T00:00:00Z",
                    "method": "human",
                },
            }
        ]
    }


def test_staging_requires_locked_input_hashes_and_row_fingerprint(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    assert len(staged) == 1

    jsonl.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ArXivQAPrepareError, match="SHA-256"):
        stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)


def test_committed_source_lock_contains_the_approved_evidence() -> None:
    lock_path = Path(__file__).resolve().parents[1] / "config" / "arxivqa_source_lock.json"
    lock = load_source_lock(lock_path)
    assert lock["official_source"]["repository"] == "MMInstruction/ArxivQA"
    assert lock["official_source"]["revision"] == "85a6dca0e2bdc6f0268ae519be8913f83a83cafd"
    assert lock["vidore_source"]["revision"] == "b8a106812c8682bab08935cf5d1b4566c82562de"
    assert lock["staged_row_fingerprint_sha256"] == "ccb60459021d211628c399682437841158e1ca238799a3cd81eef63026a7fa9a"
    assert lock["arxiv_version_policy"]["pdf_version_pinning"] == "documented_overrides_only"
    override = lock["pdf_overrides"]["2304.04203"]
    assert override["pdf_url"] == "https://arxiv.org/pdf/2304.04203v1"
    assert override["version"] == "v1"
    assert override["figure_sha256"] == "e6a559bed0f6cccd2dbeea5e3062474591b5ee47557cf3ee9904ee32db1f93bf"
    assert override["matched_page_id"] == 4
    assert override["v1_score"] == 0.7699745893478394
    assert override["v2_max_score"] == 0.2556701898574829
    assert override["withdrawal_reason"]


def test_pdf_download_plan_applies_only_explicit_overrides(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    _override = {
        "1810.10511": {
            "paper_id": "1810.10511",
            "pdf_url": "https://arxiv.org/pdf/1810.10511v2.pdf",
            "version": "v2",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": staged[0].image_sha256,
            "matched_page_id": 1,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    lock.write_text(
        json.dumps(
            _lock_payload(
                jsonl,
                parquet,
                staged[0],
                pdf_overrides=_override,
                pdf_version_pinning="documented_overrides_only",
            )
        ),
        encoding="utf-8",
    )
    plan = build_pdf_download_plan(
        staged,
        tmp_path / "pdf_download_plan.json",
        pdf_dir=tmp_path / "pdfs",
        project_root=tmp_path,
        source_lock_path=lock,
    )
    assert plan["pdf_version_pinning"] == "documented_overrides_only"
    assert plan["pdf_override_count"] == 1
    assert plan["provenance"]["source_lock"] == "source-lock.json"
    assert plan["provenance"]["source_lock_sha256"] == sha256_file(lock)
    assert plan["provenance"]["staged_row_fingerprint_sha256"] == staged_row_fingerprint(staged)
    assert plan["provenance"]["pdf_override_count"] == 1
    plan_row = plan["papers"][0]
    assert plan_row["paper_id"] == "1810.10511"
    assert plan_row["pdf_url"] == "https://arxiv.org/pdf/1810.10511v2.pdf"
    assert plan_row["pdf_override"]["version"] == "v2"
    assert plan_row["pdf_override"]["figure_sha256"] == staged[0].image_sha256
    assert plan_row["pdf_override"]["matched_page_id"] == 1

    no_override_lock = tmp_path / "no-override-lock.json"
    no_override_lock.write_text(json.dumps(_lock_payload(jsonl, parquet, staged[0])), encoding="utf-8")
    plan = build_pdf_download_plan(
        staged,
        tmp_path / "pdf_download_plan_plain.json",
        pdf_dir=tmp_path / "pdfs",
        project_root=tmp_path,
        source_lock_path=no_override_lock,
    )
    assert plan["pdf_override_count"] == 0
    assert plan["papers"][0]["pdf_url"] == "https://arxiv.org/pdf/1810.10511.pdf"
    assert "pdf_override" not in plan["papers"][0]


def test_pdf_download_plan_rejects_stale_override_paper_ids(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    _override = {
        "9999.99999": {
            "paper_id": "9999.99999",
            "pdf_url": "https://arxiv.org/pdf/9999.99999v1.pdf",
            "version": "v1",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": row.image_sha256,
            "matched_page_id": 0,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    lock.write_text(
        json.dumps(
            _lock_payload(
                jsonl,
                parquet,
                row,
                pdf_overrides=_override,
                pdf_version_pinning="documented_overrides_only",
            )
        ),
        encoding="utf-8",
    )
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    with pytest.raises(ArXivQAPrepareError, match="absent from the staged subset"):
        build_pdf_download_plan(
            staged,
            tmp_path / "pdf_download_plan.json",
            pdf_dir=tmp_path / "pdfs",
            project_root=tmp_path,
            source_lock_path=lock,
        )


def test_pdf_download_plan_rejects_figure_hash_drift(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    drifted = {
        "1810.10511": {
            "paper_id": "1810.10511",
            "pdf_url": "https://arxiv.org/pdf/1810.10511v1.pdf",
            "version": "v1",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": "f" * 64,
            "matched_page_id": 0,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    lock.write_text(
        json.dumps(
            _lock_payload(
                jsonl,
                parquet,
                row,
                pdf_overrides=drifted,
                pdf_version_pinning="documented_overrides_only",
            )
        ),
        encoding="utf-8",
    )
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    with pytest.raises(ArXivQAPrepareError, match="figure_sha256 does not match"):
        build_pdf_download_plan(
            staged,
            tmp_path / "pdf_download_plan.json",
            pdf_dir=tmp_path / "pdfs",
            project_root=tmp_path,
            source_lock_path=lock,
        )


def test_qa_source_retains_pdf_overrides_under_documented_policy(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    override = {
        "1810.10511": {
            "paper_id": "1810.10511",
            "pdf_url": "https://arxiv.org/pdf/1810.10511v1.pdf",
            "version": "v1",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": staged[0].image_sha256,
            "matched_page_id": 0,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    lock.write_text(
        json.dumps(
            _lock_payload(
                jsonl,
                parquet,
                staged[0],
                pdf_overrides=override,
                pdf_version_pinning="documented_overrides_only",
            )
        ),
        encoding="utf-8",
    )
    source = build_qa_source(
        staged,
        tmp_path / "qa_source.json",
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    assert source["source_lock"]["arxiv_version_policy"]["pdf_version_pinning"] == "documented_overrides_only"
    assert source["source_lock"]["pdf_overrides"] == override


def test_download_resume_requires_prior_url_hash_and_override_evidence(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    destination = pdf_dir / "1810.10511.pdf"
    first_pdf = _pdf_bytes()
    second_pdf = _pdf_bytes()
    # Force two distinct PDF payloads so a URL/version change is observable.
    if second_pdf == first_pdf:
        image = Image.new("RGB", (32, 32), (10, 20, 30))
        buffer = io.BytesIO()
        image.save(buffer, format="PDF")
        second_pdf = buffer.getvalue()
    assert first_pdf != second_pdf
    destination.write_bytes(first_pdf)
    override_v1 = {
        "paper_id": "1810.10511",
        "pdf_url": "https://arxiv.org/pdf/1810.10511v1.pdf",
        "version": "v1",
        "withdrawal_reason": "fixture withdrawal reason",
        "figure_sha256": "a" * 64,
        "matched_page_id": 0,
        "v1_score": 0.9,
        "v2_max_score": 0.5,
    }
    pdf_dir.joinpath("download_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "paper_id": "1810.10511",
                        "pdf_url": override_v1["pdf_url"],
                        "status": "downloaded",
                        "pdf_sha256": sha256_file(destination),
                        "bytes": destination.stat().st_size,
                        "page_count": 1,
                        "pdf_override": override_v1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    fetched: list[str] = []

    def _fetch(url: str) -> bytes:
        fetched.append(url)
        return second_pdf

    skip_summary = download_pdfs(
        {
            "papers": [
                {
                    "paper_id": "1810.10511",
                    "pdf_url": override_v1["pdf_url"],
                    "pdf_override": override_v1,
                }
            ]
        },
        pdf_dir=pdf_dir,
        delay_sec=0,
        fetcher=_fetch,
    )
    assert skip_summary["skipped_existing"] == 1
    assert skip_summary["downloaded"] == 0
    assert fetched == []
    assert skip_summary["results"][0]["pdf_override"] == override_v1
    assert skip_summary["results"][0]["pdf_sha256"] == sha256_file(destination)

    override_v2 = dict(override_v1)
    override_v2["pdf_url"] = "https://arxiv.org/pdf/1810.10511v2.pdf"
    override_v2["version"] = "v2"
    redownload = download_pdfs(
        {
            "papers": [
                {
                    "paper_id": "1810.10511",
                    "pdf_url": override_v2["pdf_url"],
                    "pdf_override": override_v2,
                }
            ]
        },
        pdf_dir=pdf_dir,
        delay_sec=0,
        fetcher=_fetch,
    )
    assert redownload["downloaded"] == 1
    assert redownload["skipped_existing"] == 0
    assert fetched == [override_v2["pdf_url"]]
    assert redownload["results"][0]["pdf_override"] == override_v2
    assert redownload["results"][0]["pdf_sha256"] == sha256_file(destination)
    assert sha256_file(destination) == _sha_bytes(second_pdf)


def test_download_result_retains_override_evidence_on_fresh_download(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    override = {
        "paper_id": "1810.10511",
        "pdf_url": "https://arxiv.org/pdf/1810.10511v1.pdf",
        "version": "v1",
        "withdrawal_reason": "fixture withdrawal reason",
        "figure_sha256": "a" * 64,
        "matched_page_id": 0,
        "v1_score": 0.9,
        "v2_max_score": 0.5,
    }
    summary = download_pdfs(
        {
            "papers": [
                {
                    "paper_id": "1810.10511",
                    "pdf_url": override["pdf_url"],
                    "pdf_override": override,
                }
            ]
        },
        pdf_dir=pdf_dir,
        delay_sec=0,
        fetcher=lambda _url: _pdf_bytes(),
    )
    result = summary["results"][0]
    assert result["status"] == "downloaded"
    assert result["pdf_override"] == override
    persisted = json.loads((pdf_dir / "download_results.json").read_text(encoding="utf-8"))
    assert persisted["results"][0]["pdf_override"] == override


def test_download_pdfs_cli_regenerates_stale_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import prepare_arxivqa_full_papers as cli

    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    write_staging_manifest(
        staged,
        tmp_path / "staging_manifest.json",
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    override = {
        "1810.10511": {
            "paper_id": "1810.10511",
            "pdf_url": "https://arxiv.org/pdf/1810.10511v1.pdf",
            "version": "v1",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": staged[0].image_sha256,
            "matched_page_id": 0,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    lock.write_text(
        json.dumps(
            _lock_payload(
                jsonl,
                parquet,
                staged[0],
                pdf_overrides=override,
                pdf_version_pinning="documented_overrides_only",
            )
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config"
    config.mkdir()
    locked = config / "arxivqa_source_lock.json"
    locked.write_text(lock.read_text(encoding="utf-8"), encoding="utf-8")

    stale_plan = {
        "paper_count": 1,
        "pdf_override_count": 0,
        "provenance": {"created_at_utc": "2000-01-01T00:00:00+00:00", "source_subset": "stale"},
        "papers": [
            {
                "paper_id": "1810.10511",
                "pdf_url": "https://arxiv.org/pdf/1810.10511.pdf",
                "destination": "pdfs/1810.10511.pdf",
                "qa_ids": ["q1"],
            }
        ],
    }
    (tmp_path / "pdf_download_plan.json").write_text(json.dumps(stale_plan), encoding="utf-8")

    fetched: list[str] = []

    def _fake_download(plan: dict, **kwargs: object) -> dict:
        fetched.append(plan["papers"][0]["pdf_url"])
        assert plan["papers"][0]["pdf_override"]["version"] == "v1"
        assert plan["provenance"]["source_lock_sha256"] == sha256_file(locked)
        return {"downloaded": 1, "skipped_existing": 0, "failed": 0, "results": [], "failures": []}

    monkeypatch.setattr(cli, "download_pdfs", _fake_download)
    monkeypatch.chdir(tmp_path)
    cli.main(
        [
            "download-pdfs",
            "--parquet",
            str(parquet),
            "--jsonl",
            str(jsonl),
            "--out-root",
            str(tmp_path),
            "--delay-sec",
            "0",
        ]
    )
    refreshed = json.loads((tmp_path / "pdf_download_plan.json").read_text(encoding="utf-8"))
    assert refreshed["papers"][0]["pdf_url"] == "https://arxiv.org/pdf/1810.10511v1.pdf"
    assert refreshed["pdf_override_count"] == 1
    assert refreshed["provenance"]["source_lock_sha256"] == sha256_file(locked)
    assert fetched == ["https://arxiv.org/pdf/1810.10511v1.pdf"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda override: override.__setitem__("pdf_url", "https://arxiv.org/pdf/1810.10511v9.pdf"),
            "does not match",
        ),
        (
            lambda override: override.__setitem__("figure_sha256", "not-a-hash"),
            "invalid figure hash",
        ),
        (
            lambda override: override.__setitem__("matched_page_id", -1),
            "invalid matched page id",
        ),
        (
            lambda override: override.__setitem__("v1_score", 1.5),
            "invalid v1_score",
        ),
        (
            lambda override: override.__setitem__("withdrawal_reason", " "),
            "withdrawal reason",
        ),
    ],
)
def test_source_lock_rejects_invalid_pdf_override_schema(
    tmp_path: Path, mutate: Callable[[dict], None], message: str
) -> None:
    _override = {
        "1810.10511": {
            "paper_id": "1810.10511",
            "pdf_url": "https://arxiv.org/pdf/1810.10511v2.pdf",
            "version": "v2",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": "a" * 64,
            "matched_page_id": 1,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    mutate(_override["1810.10511"])
    parquet, jsonl, lock, row = _write_fixture(tmp_path, pdf_overrides=_override)
    lock.write_text(
        json.dumps(
            _lock_payload(
                jsonl,
                parquet,
                row,
                pdf_overrides=_override,
                pdf_version_pinning="documented_overrides_only",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArXivQAPrepareError, match=message):
        load_source_lock(lock)


def test_source_lock_rejects_overrides_without_documented_pinning(tmp_path: Path) -> None:
    _override = {
        "1810.10511": {
            "paper_id": "1810.10511",
            "pdf_url": "https://arxiv.org/pdf/1810.10511v2.pdf",
            "version": "v2",
            "withdrawal_reason": "fixture withdrawal reason",
            "figure_sha256": "a" * 64,
            "matched_page_id": 1,
            "v1_score": 0.9,
            "v2_max_score": 0.5,
        }
    }
    parquet, jsonl, lock, row = _write_fixture(tmp_path, pdf_overrides=_override)
    lock.write_text(
        json.dumps(_lock_payload(jsonl, parquet, row, pdf_overrides=_override)), encoding="utf-8"
    )
    with pytest.raises(ArXivQAPrepareError, match="pdf_version_pinning does not document them"):
        load_source_lock(lock)


def test_source_lock_rejects_documented_pinning_without_overrides(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path, pdf_version_pinning="documented_overrides_only")
    lock.write_text(
        json.dumps(_lock_payload(jsonl, parquet, row, pdf_version_pinning="documented_overrides_only")),
        encoding="utf-8",
    )
    with pytest.raises(ArXivQAPrepareError, match="records no pdf_overrides"):
        load_source_lock(lock)


def test_qa_source_retains_upstream_test_and_explicit_project_policy(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    source = build_qa_source(
        staged,
        tmp_path / "qa_source.json",
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    assert source["upstream_split"] == "test"
    assert source["supervisor_approved_project_role"] == "validation"
    assert source["project_role_policy"]["allows_upstream_test_as_project_validation"] is True
    assert qa_rows_content_fingerprint(source["data"]) == source["source_lock"]["staged_row_fingerprint_sha256"]


def test_download_rejects_truncated_existing_pdf_and_records_pdf_hash(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_dir.joinpath("1810.10511.pdf").write_bytes(b"%PDF truncated")
    summary = download_pdfs(
        {"papers": [{"paper_id": "1810.10511"}]},
        pdf_dir=pdf_dir,
        delay_sec=0,
        fetcher=lambda _url: _pdf_bytes(),
    )
    result = summary["results"][0]
    assert result["status"] == "downloaded"
    assert result["pdf_sha256"] == sha256_file(pdf_dir / "1810.10511.pdf")
    assert result["page_count"] == 1


def test_zero_page_pdf_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class EmptyPdf:
        def __init__(self, _path: str) -> None:
            pass

        def __len__(self) -> int:
            return 0

        def close(self) -> None:
            pass

    path = tmp_path / "empty.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setitem(sys.modules, "pypdfium2", SimpleNamespace(PdfDocument=EmptyPdf))
    with pytest.raises(ArXivQAPrepareError, match="zero pages"):
        inspect_pdf(path)


def test_render_and_inventory_reject_missing_or_partial_papers(tmp_path: Path) -> None:
    with pytest.raises(ArXivQAPrepareError, match="missing or empty"):
        render_all_pdf_pages(tmp_path / "missing.pdf", tmp_path / "images", paper_id="1810.10511")

    inventory, _path = _make_inventory(tmp_path)
    assert inventory["state"] == "preparation_only"
    assert inventory["papers"][0]["page_count"] == 1
    assert inventory["papers"][0]["pages"][0]["image_sha256"]
    broken = dict(inventory["papers"][0])
    broken["pages"] = []
    with pytest.raises(ArXivQAPrepareError, match="exactly 0..0"):
        build_paper_inventory(
            [broken],
            tmp_path / "broken.json",
            project_root=tmp_path,
            expected_paper_ids=["1810.10511"],
        )
    with pytest.raises(ArXivQAPrepareError, match="incomplete"):
        build_paper_inventory([], tmp_path / "missing-paper.json", expected_paper_ids=["1810.10511"])


def test_remap_requires_ocr_readiness_and_human_confirmation(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    qa_path = tmp_path / "qa_source.json"
    build_qa_source(
        staged,
        qa_path,
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    inventory, inventory_path = _make_inventory(tmp_path)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(_human_mapping()), encoding="utf-8")
    with pytest.raises(ArXivQARemapError, match="preparation-only"):
        build_full_paper_source(qa_path, inventory_path, mapping_path, project_root=tmp_path)

    ready = _finalize_inventory(tmp_path, inventory)
    ready_path = tmp_path / "paper_inventory_ocr_ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    mapping = _human_mapping()
    del mapping["data"][0]["human_confirmation"]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    with pytest.raises(ArXivQARemapError, match="human_confirmation"):
        build_full_paper_source(qa_path, ready_path, mapping_path, project_root=tmp_path)

    mapping_path.write_text(json.dumps(_human_mapping()), encoding="utf-8")
    remapped = build_full_paper_source(qa_path, ready_path, mapping_path, project_root=tmp_path)
    assert remapped["upstream_split"] == "test"
    assert remapped["data"][0]["metadata"]["human_confirmation"]["method"] == "human"


def test_similarity_output_is_only_a_candidate_not_a_final_mapping(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    figure = tmp_path / "figure.jpg"
    figure.write_bytes(_image_bytes())
    page = tmp_path / "page.png"
    Image.open(figure).save(page)
    evidence = match_evidence_pages(
        staged,
        {"figures": [{"qa_id": "q1", "figure_path": "figure.jpg"}]},
        {
            "papers": [
                {
                    "paper_id": "1810.10511",
                    "pages": [{"page_id": 0, "image_path": "page.png"}],
                }
            ]
        },
        project_root=tmp_path,
    )
    assert evidence["candidate_count"] == 1
    assert "accepted_figure_mappings" not in evidence
    assert evidence["review_candidates"][0]["requires_human_confirmation"] is True


def test_remap_rejects_forged_qa_content_despite_copied_fingerprint(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    qa_path = tmp_path / "qa_source.json"
    source = build_qa_source(
        staged,
        qa_path,
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    inventory, _inventory_path = _make_inventory(tmp_path)
    ready = _finalize_inventory(tmp_path, inventory)
    ready_path = tmp_path / "paper_inventory_ocr_ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(_human_mapping()), encoding="utf-8")

    forged = json.loads(qa_path.read_text(encoding="utf-8"))
    forged["data"][0]["answer"] = "FORGED"
    forged["val"][0]["answer"] = "FORGED"
    qa_path.write_text(json.dumps(forged), encoding="utf-8")
    assert forged["source_lock"]["staged_row_fingerprint_sha256"] == source["source_lock"]["staged_row_fingerprint_sha256"]
    with pytest.raises(ArXivQARemapError, match="content fingerprint"):
        build_full_paper_source(qa_path, ready_path, mapping_path, project_root=tmp_path)


def test_remap_rejects_forged_ocr_ready_inventory(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    qa_path = tmp_path / "qa_source.json"
    build_qa_source(
        staged,
        qa_path,
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    inventory, _inventory_path = _make_inventory(tmp_path)
    ready = _finalize_inventory(tmp_path, inventory)
    ready["papers"][0]["pages"][0]["ocr_sha256"] = "a" * 64
    ready_path = tmp_path / "forged_inventory.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(_human_mapping()), encoding="utf-8")
    with pytest.raises(ArXivQARemapError, match="OCR hash"):
        build_full_paper_source(qa_path, ready_path, mapping_path, project_root=tmp_path)


def test_remap_rejects_mapping_paper_mismatch(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    qa_path = tmp_path / "qa_source.json"
    build_qa_source(
        staged,
        qa_path,
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    inventory, _inventory_path = _make_inventory(tmp_path)
    ready = _finalize_inventory(tmp_path, inventory)
    ready_path = tmp_path / "paper_inventory_ocr_ready.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(_human_mapping(paper_id="9999.99999")), encoding="utf-8")
    with pytest.raises(ArXivQARemapError, match="does not equal staged QA paper_id"):
        build_full_paper_source(qa_path, ready_path, mapping_path, project_root=tmp_path)


def test_remap_rejects_partial_page_inventory(tmp_path: Path) -> None:
    parquet, jsonl, lock, _row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    qa_path = tmp_path / "qa_source.json"
    build_qa_source(
        staged,
        qa_path,
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    inventory, _inventory_path = _make_inventory(tmp_path)
    ready = _finalize_inventory(tmp_path, inventory)
    ready["papers"][0]["page_count"] = 2
    ready_path = tmp_path / "partial_pages.json"
    ready_path.write_text(json.dumps(ready), encoding="utf-8")
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps(_human_mapping()), encoding="utf-8")
    with pytest.raises(ArXivQARemapError, match="exactly 0..1|PDF page count"):
        build_full_paper_source(qa_path, ready_path, mapping_path, project_root=tmp_path)


def test_figure_extraction_rejects_incomplete_coverage(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    extra = StagedRow(
        qa_id="q-missing",
        question="Missing figure question?",
        answer="A",
        options=["A. Red", "B. Blue"],
        image_filename="images/1810.10511_99.jpg",
        figure_index=99,
        paper_id="1810.10511",
        modern_id="1810.10511",
        legacy_id=None,
        official_id="q-missing",
        official_id_aliases=[],
        vidore_index=1,
        rationale=None,
        image_sha256="b" * 64,
    )
    with pytest.raises(ArXivQAPrepareError, match="cover every staged row exactly"):
        extract_figures_from_parquet(parquet, staged + [extra], tmp_path / "figures", project_root=tmp_path)


def test_cli_reload_rejects_staging_when_parquet_hash_drifts(tmp_path: Path) -> None:
    parquet, jsonl, lock, row = _write_fixture(tmp_path)
    staged = stage_vidore_against_official(parquet, jsonl, source_lock_path=lock)
    write_staging_manifest(
        staged,
        tmp_path / "staging_manifest.json",
        parquet_path=parquet,
        jsonl_path=jsonl,
        project_root=tmp_path,
        source_lock_path=lock,
    )
    parquet.write_bytes(parquet.read_bytes() + b"tamper")
    with pytest.raises(ArXivQAPrepareError, match="SHA-256"):
        load_verified_staged(
            tmp_path / "staging_manifest.json",
            parquet,
            jsonl,
            source_lock_path=lock,
        )
