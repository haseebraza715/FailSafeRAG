from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from faar.benchmarks import BenchmarkRepository, build_ohr_asset_manifest
from faar.data import DatasetUnavailableError
from faar.settings import RetrievalSettings


def _image_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_fixture_repository(
    tmp_path: Path,
    *,
    hashes: dict[int, str] | None = None,
    uppercase: bool = False,
) -> BenchmarkRepository:
    """Repository whose corpus pages optionally record image_sha256 claims."""
    image_dir = tmp_path / "images"
    ocr_dir = tmp_path / "ocr"
    image_dir.mkdir()
    ocr_dir.mkdir()
    corpus_pages = []
    for page_id in (1, 2):
        image_path = image_dir / f"doc_page_{page_id}.png"
        image_path.write_bytes(f"image-{page_id}".encode())
        (ocr_dir / f"doc_page_{page_id}.txt").write_text(f"OCR {page_id}")
        page = {
            "corpus_id": f"doc:p{page_id}",
            "doc_name": "doc",
            "page_id": page_id,
            "ocr_text_path": str(ocr_dir / f"doc_page_{page_id}.txt"),
            "image_path": str(image_path),
        }
        if hashes is not None and page_id in hashes:
            page["image_sha256"] = hashes[page_id].upper() if uppercase else hashes[page_id]
        corpus_pages.append(page)
    records = [
        {
            "example_id": "e1",
            "doc_name": "doc",
            "question": "Q",
            "correct_answer": "A",
            "page_ids": [1],
            "corpus_ids": ["doc:p1", "doc:p2"],
        }
    ]
    return BenchmarkRepository(
        records,
        corpus_pages,
        tmp_path,
        "ohrbench",
        "test",
        document_inventory={"doc": [1, 2]},
    )


def _write_ohr_fixture(tmp_path: Path, *, pages: range, evidence_page: int = 2) -> None:
    (tmp_path / "OHR-Bench/data/retrieval_base/gt").mkdir(parents=True)
    (tmp_path / "ocr").mkdir()
    (tmp_path / "images").mkdir()
    (tmp_path / "config/datasets").mkdir(parents=True)
    (tmp_path / "config/datasets/ohr_split.json").write_text(json.dumps({"splits": {"test": ["e1"], "val": [], "train": []}}))
    (tmp_path / "OHR-Bench/data/qas_v2.json").write_text(
        json.dumps(
            [
                {
                    "ID": "e1",
                    "doc_name": "doc",
                    "questions": "Q",
                    "answers": "A",
                    "evidence_page_no": evidence_page,
                }
            ]
        )
    )
    inventory_rows = [{"text": f"gt {page}", "page_idx": page} for page in pages]
    (tmp_path / "OHR-Bench/data/retrieval_base/gt/doc.json").write_text(json.dumps(inventory_rows))
    for page in pages:
        (tmp_path / "ocr" / f"doc_page_{page}.txt").write_text(f"OCR {page}")
        (tmp_path / "images" / f"doc_page_{page}.png").write_bytes(b"image")


def test_benchmark_repository_requires_complete_ocr_and_images(tmp_path: Path) -> None:
    record = {
        "example_id": "e1",
        "doc_name": "doc",
        "question": "Q",
        "correct_answer": "A",
        "page_ids": [1],
        "corpus_ids": ["doc:p1"],
    }
    corpus_pages = [
        {
            "corpus_id": "doc:p1",
            "doc_name": "doc",
            "page_id": 1,
            "ocr_text_path": "missing.txt",
            "image_path": "",
        }
    ]
    with pytest.raises(DatasetUnavailableError, match="not ready for a paper run|incomplete"):
        BenchmarkRepository(
            [record],
            corpus_pages,
            tmp_path,
            "ohrbench",
            "test",
            document_inventory={"doc": [1]},
        )


def test_ohr_manifest_keeps_evidence_metadata_but_indexes_all_document_pages(tmp_path: Path) -> None:
    _write_ohr_fixture(tmp_path, pages=range(1, 6), evidence_page=2)

    manifest = build_ohr_asset_manifest(tmp_path, "test", tmp_path / "ocr", tmp_path / "images")

    assert manifest["records"][0]["example_id"] == "e1"
    assert manifest["records"][0]["page_ids"] == [2]
    assert manifest["records"][0]["metadata"]["evidence_page_ids"] == [2]
    assert manifest["records"][0]["corpus_ids"] == [f"doc:p{page}" for page in range(1, 6)]
    assert [page["page_id"] for page in manifest["corpus_pages"]] == [1, 2, 3, 4, 5]
    assert all(not Path(page["image_path"]).is_absolute() for page in manifest["corpus_pages"])
    assert all(not Path(page["ocr_text_path"]).is_absolute() for page in manifest["corpus_pages"])
    assert manifest["document_inventory"]["doc"] == [1, 2, 3, 4, 5]

    repository = BenchmarkRepository(
        manifest["records"],
        manifest["corpus_pages"],
        tmp_path,
        "ohrbench",
        "test",
        document_inventory=manifest["document_inventory"],
    )
    chunks = repository.get_corpus_chunks(RetrievalSettings(chunk_size_words=8, chunk_overlap_words=0))
    assert {chunk.page_id for chunk in chunks} == {1, 2, 3, 4, 5}


def test_ohr_manifest_rejects_evidence_only_assets(tmp_path: Path) -> None:
    _write_ohr_fixture(tmp_path, pages=range(1, 6), evidence_page=2)
    # Keep inventory for pages 1-5, but delete non-evidence OCR/image assets.
    for page in (1, 3, 4, 5):
        (tmp_path / "ocr" / f"doc_page_{page}.txt").unlink()
        (tmp_path / "images" / f"doc_page_{page}.png").unlink()

    with pytest.raises(DatasetUnavailableError, match="incomplete relative to the document inventory"):
        build_ohr_asset_manifest(tmp_path, "test", tmp_path / "ocr", tmp_path / "images")


def test_load_rejects_path_traversal_in_manifest(tmp_path: Path) -> None:
    asset_dir = tmp_path / "data/benchmark_assets/ohrbench"
    asset_dir.mkdir(parents=True)
    (tmp_path / "safe.png").write_bytes(b"png")
    (tmp_path / "safe.txt").write_text("OCR")
    payload = {
        "records": [
            {
                "example_id": "e1",
                "doc_name": "doc",
                "question": "Q",
                "correct_answer": "A",
                "page_ids": [1],
                "corpus_ids": ["doc:p1"],
            }
        ],
        "corpus_pages": [
            {
                "corpus_id": "doc:p1",
                "doc_name": "doc",
                "page_id": 1,
                "ocr_text_path": "../outside.txt",
                "image_path": "safe.png",
            }
        ],
        "document_inventory": {"doc": [1]},
    }
    (asset_dir / "test.json").write_text(json.dumps(payload))

    from faar.asset_paths import AssetPathError
    from faar.benchmarks import load_benchmark_repository

    with pytest.raises((DatasetUnavailableError, AssetPathError, ValueError), match="\\.\\.|escapes|not ready"):
        load_benchmark_repository(tmp_path, "ohrbench", "test")


def test_corpus_image_hashes_verifies_valid_recorded_hashes(tmp_path: Path) -> None:
    hashes = {
        1: _image_sha256(b"image-1"),
        2: _image_sha256(b"image-2"),
    }
    repository = _hash_fixture_repository(tmp_path, hashes=hashes)
    verified = repository.corpus_image_hashes()
    assert verified is not None
    assert [path for path, _ in verified] == repository.corpus_image_paths()
    assert [sha for _, sha in verified] == [hashes[1], hashes[2]]


def test_corpus_image_hashes_normalizes_uppercase_hashes(tmp_path: Path) -> None:
    hashes = {
        1: _image_sha256(b"image-1").upper(),
        2: _image_sha256(b"image-2").upper(),
    }
    repository = _hash_fixture_repository(tmp_path, hashes=hashes, uppercase=True)
    verified = repository.corpus_image_hashes()
    assert verified is not None
    assert [sha for _, sha in verified] == [
        _image_sha256(b"image-1"),
        _image_sha256(b"image-2"),
    ]


@pytest.mark.parametrize("bad", ["xyz", "0" * 63, "g" * 64, "", 42])
def test_corpus_image_hashes_rejects_malformed_hash(tmp_path: Path, bad) -> None:
    repository = _hash_fixture_repository(tmp_path, hashes={1: bad, 2: _image_sha256(b"image-2")})
    with pytest.raises(DatasetUnavailableError, match="malformed.*image_sha256"):
        repository.corpus_image_hashes()


def test_corpus_image_hashes_fails_on_missing_image_file(tmp_path: Path) -> None:
    hashes = {1: _image_sha256(b"image-1"), 2: _image_sha256(b"image-2")}
    repository = _hash_fixture_repository(tmp_path, hashes=hashes)
    repository.corpus_image_paths()[1].unlink()
    with pytest.raises(DatasetUnavailableError, match="cannot be hashed"):
        repository.corpus_image_hashes()


def test_corpus_image_hashes_fails_on_replaced_bytes_with_stale_recorded_hash(
    tmp_path: Path,
) -> None:
    hashes = {1: _image_sha256(b"image-1"), 2: _image_sha256(b"image-2")}
    repository = _hash_fixture_repository(tmp_path, hashes=hashes)
    replaced = repository.corpus_image_paths()[0]
    replaced.write_bytes(b"re-rendered content")
    with pytest.raises(DatasetUnavailableError) as excinfo:
        repository.corpus_image_hashes()
    message = str(excinfo.value)
    assert "ohrbench test" in message
    assert "doc:p1" in message
    assert str(replaced) in message
    assert hashes[1] in message
    assert _image_sha256(b"re-rendered content") in message


def test_corpus_image_hashes_mixed_present_missing_falls_back_to_none(tmp_path: Path) -> None:
    repository = _hash_fixture_repository(tmp_path, hashes={1: _image_sha256(b"image-1")})
    assert repository.corpus_image_hashes() is None


def test_corpus_image_hashes_does_not_call_read_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_read_bytes(self, *args, **kwargs):
        raise AssertionError("read_bytes must not be used for streaming image hashing")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    hashes = {1: _image_sha256(b"image-1"), 2: _image_sha256(b"image-2")}
    repository = _hash_fixture_repository(tmp_path, hashes=hashes)
    assert repository.corpus_image_hashes() is not None


def test_corpus_image_hashes_hashes_each_image_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import faar.benchmarks as benchmarks_module

    calls: list[Path] = []
    real_sha256_file = benchmarks_module.sha256_file

    def counting_sha256_file(path: Path, **kwargs) -> str:
        calls.append(path)
        return real_sha256_file(path, **kwargs)

    monkeypatch.setattr(benchmarks_module, "sha256_file", counting_sha256_file)
    hashes = {1: _image_sha256(b"image-1"), 2: _image_sha256(b"image-2")}
    repository = _hash_fixture_repository(tmp_path, hashes=hashes)
    repository.corpus_image_hashes()
    first_pass = list(calls)
    assert len(first_pass) == 2
    repository.corpus_image_hashes()
    assert calls == first_pass


def test_register_cli_writes_manifest_atomically(tmp_path: Path, monkeypatch) -> None:
    import register_benchmark_assets as register

    monkeypatch.setattr(
        register,
        "build_ohr_asset_manifest",
        lambda root, split, ocr_dir, image_dir, document_inventory_dir=None: {
            "records": [],
            "corpus_pages": [],
            "document_inventory": {},
        },
    )
    out = tmp_path / "ohrbench" / "val.json"
    monkeypatch.chdir(tmp_path)
    code = register.main(
        [
            "--dataset",
            "ohrbench",
            "--split",
            "val",
            "--ocr-dir",
            str(tmp_path / "ocr"),
            "--image-dir",
            str(tmp_path / "images"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["dataset"] == "ohrbench"
    assert not list(out.parent.glob(".val.json.*.tmp"))
