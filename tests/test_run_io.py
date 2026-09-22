from __future__ import annotations

import json
from pathlib import Path

import pytest

from faar.run_io import (
    atomic_write_json,
    atomic_write_text,
    canonical_digest,
    run_fingerprint,
    safe_checkpoint_stem,
    select_shard,
)
from faar.settings import AppSettings


def test_atomic_write_text_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.txt"
    atomic_write_text(target, "hello \u00fc\u4e2d\u6587\n")
    assert target.read_text(encoding="utf-8") == "hello \u00fc\u4e2d\u6587\n"
    assert not list(tmp_path.rglob(".*.tmp"))


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_json(target, {"a": 1, "nested": {"b": [True, None]}})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "nested": {"b": [True, None]}}
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_write_overwrites_existing(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}


@pytest.mark.parametrize("example_id", ["folder/item", "../item", "\u8ad6\u6587/42", ".."])
def test_checkpoint_stem_cannot_escape_output_directory(example_id: str) -> None:
    stem = safe_checkpoint_stem(example_id)
    assert stem not in {".", ".."}
    assert "/" not in stem
    assert "\\" not in stem


def test_checkpoint_stem_preserves_simple_ids() -> None:
    assert safe_checkpoint_stem("ohr-123_4.5") == "ohr-123_4.5"


def test_canonical_digest_is_order_independent() -> None:
    left = canonical_digest({"b": 2, "a": [1, {"z": True}], "c": None})
    right = canonical_digest({"c": None, "a": [1, {"z": True}], "b": 2})
    assert left == right


def test_canonical_digest_changes_with_value() -> None:
    assert canonical_digest({"profile": "faar_full"}) != canonical_digest({"profile": "faar_no_vlm"})


def test_run_fingerprint_stable_for_equal_settings(tmp_path: Path) -> None:
    first = AppSettings(project_root=tmp_path)
    second = AppSettings(project_root=tmp_path)
    assert run_fingerprint(first, profile="faar_full", dataset="ohrbench", split="test") == run_fingerprint(
        second, profile="faar_full", dataset="ohrbench", split="test"
    )


def test_run_fingerprint_sensitive_to_config_and_identity(tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    baseline = run_fingerprint(settings, profile="faar_full", dataset="ohrbench", split="test")
    changed_model = AppSettings(project_root=tmp_path)
    changed_model.retrieval.embedding_model = "other/embedder"
    assert run_fingerprint(changed_model, profile="faar_full", dataset="ohrbench", split="test") != baseline
    assert run_fingerprint(settings, profile="faar_no_vlm", dataset="ohrbench", split="test") != baseline
    assert run_fingerprint(settings, profile="faar_full", dataset="ohrbench", split="val") != baseline


def test_select_shard_partitions_and_is_deterministic() -> None:
    ids = ["e3", "e1", "e4", "e2", "e5", "e7", "e6"]
    shards = [select_shard(ids, index, 3) for index in range(3)]
    union = sorted(item for shard in shards for item in shard)
    assert union == sorted(ids)
    assert all(len(set(shard)) == len(shard) for shard in shards)
    assert select_shard(ids, 0, 3) == select_shard(ids, 0, 3)


def test_select_shard_validates_bounds() -> None:
    with pytest.raises(ValueError, match="num_shards"):
        select_shard(["a"], 0, 0)
    with pytest.raises(ValueError, match="num_shards"):
        select_shard(["a"], 0, -1)
    with pytest.raises(ValueError, match="out of range"):
        select_shard(["a"], 1, 1)
    with pytest.raises(ValueError, match="out of range"):
        select_shard(["a"], -1, 2)
    with pytest.raises(ValueError, match="num_shards"):
        select_shard(["a"], 0, "2")
    with pytest.raises(ValueError, match="shard_index"):
        select_shard(["a"], "0", 2)
    with pytest.raises(ValueError, match="duplicates"):
        select_shard(["a", "a"], 0, 2)
