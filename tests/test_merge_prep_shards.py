from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MERGE_PATH = Path(__file__).resolve().parents[1] / "cluster/merge_prep_shards.py"
MERGE_SPEC = spec_from_file_location("merge_prep_shards", MERGE_PATH)
assert MERGE_SPEC and MERGE_SPEC.loader
merge = module_from_spec(MERGE_SPEC)
sys.modules["merge_prep_shards"] = merge
MERGE_SPEC.loader.exec_module(merge)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_prepare_assets import DOCS, build_project  # noqa: E402


def shard_manifest(tmp_path: Path, index: int, num_shards: int, docs: list[str], split: str = "val") -> Path:
    payload = {
        "schema_version": 1,
        "kind": "ohr-asset-shard-manifest",
        "dataset": "ohrbench",
        "split": split,
        "shard_index": index,
        "num_shards": num_shards,
        "created_at_utc": "t",
        "finished_at_utc": "t",
        "resumed": False,
        "bounds": {"max_documents": None, "max_pages": None, "bounds_hit": False},
        "documents_assigned": docs,
        "documents_completed": docs,
        "documents_failed": {},
        "retried_documents": [],
        "pages_assigned": len(docs) * 2,
        "pages_completed": len(docs) * 2,
        "runtime_sec": 1.0,
        "checkpoint": "c",
        "out_root": "o",
    }
    path = tmp_path / f"shard{index}.json"
    path.write_text(json.dumps(payload))
    return path


def test_merge_cli_validates_and_writes_atomically(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    m0 = shard_manifest(tmp_path, 0, 2, DOCS[:2])
    m1 = shard_manifest(tmp_path, 1, 2, DOCS[2:])
    out = tmp_path / "merged.json"
    code = merge.main(["--project-root", str(project), "--split", "val", "--out", str(out), str(m0), str(m1)])
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["kind"] == "merged-asset-preparation"
    assert payload["documents"] == DOCS
    assert payload["num_shards"] == 2
    assert payload["pages_completed"] == 6
    assert not list(tmp_path.glob(".merged.json.*.tmp"))


def test_merge_cli_rejects_out_of_range_and_incomplete(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    m0 = shard_manifest(tmp_path, 0, 2, DOCS[:2])
    m1 = shard_manifest(tmp_path, 1, 2, DOCS[2:])
    out_of_range = shard_manifest(tmp_path, 2, 2, [DOCS[2]])
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit, match="expected an index in"):
        merge.main(["--project-root", str(project), "--out", str(out), str(m0), str(m1), str(out_of_range)])
    with pytest.raises(SystemExit, match="missing shard indices"):
        merge.main(["--project-root", str(project), "--out", str(out), str(m0)])


def test_merge_cli_verifies_split_checksum(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    (project / "config/split_checksums.json").unlink()
    m0 = shard_manifest(tmp_path, 0, 1, DOCS)
    with pytest.raises(SystemExit, match="split checksum lock"):
        merge.main(["--project-root", str(project), "--out", str(tmp_path / "m.json"), str(m0)])


def test_merge_cli_rejects_expected_document_mismatch(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    m0 = shard_manifest(tmp_path, 0, 1, [DOCS[0]])
    with pytest.raises(SystemExit, match="do not match the expected document set"):
        merge.main(["--project-root", str(project), "--out", str(tmp_path / "m.json"), str(m0)])


def test_merge_cli_rejects_split_label_mismatch(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    test_manifest = shard_manifest(tmp_path, 0, 1, DOCS, split="test")
    with pytest.raises(SystemExit, match="mismatched merge"):
        merge.main(
            [
                "--project-root",
                str(project),
                "--split",
                "val",
                "--out",
                str(tmp_path / "merged.json"),
                str(test_manifest),
            ]
        )
    assert not (tmp_path / "merged.json").exists()


def test_merge_cli_subprocess_end_to_end(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    m0 = shard_manifest(tmp_path, 0, 2, DOCS[:2])
    m1 = shard_manifest(tmp_path, 1, 2, DOCS[2:])
    out = tmp_path / "merged.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MERGE_PATH),
            "--project-root",
            str(project),
            "--split",
            "val",
            "--out",
            str(out),
            str(m0),
            str(m1),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=project,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text())["documents"] == DOCS
