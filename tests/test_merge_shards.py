from __future__ import annotations

import hashlib
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
SPEC = spec_from_file_location("cluster_merge_shards", CLUSTER / "merge_shards.py")
assert SPEC and SPEC.loader
merge_shards = module_from_spec(SPEC)
sys.modules["cluster_merge_shards"] = merge_shards
SPEC.loader.exec_module(merge_shards)


def _row(example_id: str) -> dict:
    return {
        "example_id": example_id,
        "question": f"Q {example_id}",
        "gold_answer": "gold",
        "predicted_answer": "gold",
        "failure_type": "pass",
        "policy_action": "answer_direct",
        "action_outcome": {"action": "answer_direct", "status": "succeeded"},
        "metrics": {"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0},
        "api_usage": {"api_requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
        "run_metadata": {"run_id": "r1"},
    }


def _shard(path: Path, rows: list[dict], *, shard_index: int = 0, num_shards: int = 2) -> dict:
    payload = {
        "label": "Random recovery",
        "profile": "faar_no_diagnosis",
        "run_spec": {
            "profile": "faar_no_diagnosis",
            "dataset": "ohrbench",
            "split": "test",
            "seed": 42,
            "max_examples": None,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "embedding_model": "nvidia/NV-Embed-v2",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "ocr_engine": "got-ocr-2",
            "model_provenance": {"embedding": {"repository": "nvidia/NV-Embed-v2"}},
            "manifest_sha256": "a" * 64,
        },
        "summary": {
            "EM": 1.0,
            "F1": 1.0,
            "vlm_rate": 0.0,
            "harm_rate": 0.0,
            "api_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "runtime_sec": 10.0,
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload))
    return payload


def _manifest(path: Path, example_ids: list[str], *, dataset: str = "ohrbench", split: str = "test") -> str:
    payload = {
        "dataset": dataset,
        "split": split,
        "records": [{"example_id": example_id} for example_id in example_ids],
        "corpus_pages": [],
        "document_inventory": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_manifest_hash(paths: list[Path], manifest_hash: str, *, profile: str = "naive_rag") -> None:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["profile"] = profile
        payload["run_spec"]["profile"] = profile
        if profile == "naive_rag":
            payload["label"] = "Text-only RAG"
        payload["run_spec"]["manifest_sha256"] = manifest_hash
        path.write_text(json.dumps(payload), encoding="utf-8")


def _make_unsharded_b0(path: Path, rows: list[dict]) -> dict:
    payload = _shard(path, rows, shard_index=0)
    payload["profile"] = "naive_rag"
    payload["run_spec"]["profile"] = "naive_rag"
    payload["run_spec"]["shard_index"] = None
    payload["run_spec"]["num_shards"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_merge_combines_disjoint_shards_in_order(tmp_path: Path) -> None:
    first_path = tmp_path / "b2_shard1of2.json"
    second_path = tmp_path / "b2_shard2of2.json"
    _shard(first_path, [_row("e1")], shard_index=0)
    _shard(second_path, [_row("e2")], shard_index=1)
    merged = merge_shards.merge_shards([first_path, second_path])
    assert [row["example_id"] for row in merged["rows"]] == ["e1", "e2"]
    assert merged["run_spec"]["shard_index"] is None
    assert merged["run_spec"]["num_shards"] is None
    assert merged["summary"]["EM"] == 1.0
    assert merged["summary"]["api_requests"] == 0


def test_merge_rejects_duplicate_example_ids(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _shard(first, [_row("e1")], shard_index=0)
    _shard(second, [_row("e1")], shard_index=1)
    with pytest.raises(SystemExit, match="duplicate example_id"):
        merge_shards.merge_shards([first, second])


def test_merge_rejects_rows_with_unscoreable_metrics(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    missing = _row("e1")
    del missing["metrics"]["em"]
    _shard(first, [missing], shard_index=0)
    _shard(second, [_row("e2")], shard_index=1)
    with pytest.raises(SystemExit, match="has no metrics.em"):
        merge_shards.merge_shards([first, second])

    nonfinite = _row("e3")
    nonfinite["metrics"]["f1"] = {"score": 0.9}
    _shard(first, [nonfinite], shard_index=0)
    with pytest.raises(SystemExit, match="invalid metrics.f1"):
        merge_shards.merge_shards([first, second])

    out_of_range = _row("e4")
    out_of_range["metrics"]["f1"] = 1.5
    _shard(first, [out_of_range], shard_index=0)
    with pytest.raises(SystemExit, match="outside \\[0, 1\\]"):
        merge_shards.merge_shards([first, second])


def test_merge_rejects_failed_rows_with_metrics(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    failed = _row("e1")
    failed["action_outcome"] = {"action": "failed", "status": "failed"}
    failed["error"] = "FileNotFoundError: missing OCR text"
    _shard(first, [failed], shard_index=0)
    _shard(second, [_row("e2")], shard_index=1)

    with pytest.raises(SystemExit, match="failed row 'e1'"):
        merge_shards.merge_shards([first, second])


def test_merge_rejects_mismatched_run_specs(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _shard(first, [_row("e1")], shard_index=0)
    payload = _shard(second, [_row("e2")], shard_index=1)
    payload["run_spec"]["seed"] = 7
    second.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="does not match"):
        merge_shards.merge_shards([first, second])


def test_merge_rejects_inconsistent_num_shards(tmp_path: Path) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    _shard(first, [_row("e1")], shard_index=0, num_shards=2)
    _shard(second, [_row("e2")], shard_index=1, num_shards=4)
    with pytest.raises(SystemExit, match="num_shards"):
        merge_shards.merge_shards([first, second])


def test_merge_cli_computes_harm_against_baseline(tmp_path: Path) -> None:
    first = tmp_path / "b2_shard1of2.json"
    second = tmp_path / "b2_shard2of2.json"
    baseline = tmp_path / "b0.json"
    _shard(first, [_row("e1")], shard_index=0)
    _shard(second, [_row("e2")], shard_index=1)
    _make_unsharded_b0(
        baseline,
        [
            dict(_row("e1"), metrics={"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0}),
            dict(_row("e2"), metrics={"ndcg@5": 1.0, "recall@5": 1.0, "em": 1.0, "f1": 1.0}),
        ],
    )
    out = tmp_path / "b2_merged.json"
    assert merge_shards.main(["--out", str(out), "--baseline", str(baseline), str(first), str(second)]) == 0
    payload = json.loads(out.read_text())
    assert payload["summary"]["harm_rate"] == 0.0
    assert payload["rows"][0]["example_id"] == "e1"


def test_merge_rejects_one_of_eight_shards(tmp_path: Path) -> None:
    only = tmp_path / "b2_shard1of8.json"
    _shard(only, [_row("e1")], shard_index=0, num_shards=8)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([only])
    assert "incomplete shard set for num_shards=8" in str(excinfo.value)
    assert "missing shard indices [1, 2, 3, 4, 5, 6, 7]" in str(excinfo.value)


def test_merge_rejects_missing_middle_shard(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    two = tmp_path / "shard2.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=3)
    _shard(two, [_row("e3")], shard_index=2, num_shards=3)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([zero, two])
    assert "missing shard indices [1]" in str(excinfo.value)


def test_merge_rejects_duplicate_shard_index_with_disjoint_ids(tmp_path: Path) -> None:
    first = tmp_path / "dup_a.json"
    second = tmp_path / "dup_b.json"
    _shard(first, [_row("e1")], shard_index=0, num_shards=2)
    _shard(second, [_row("e2")], shard_index=0, num_shards=2)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([first, second])
    assert "shard index 0 declared by" in str(excinfo.value)


def test_merge_rejects_out_of_range_and_negative_indices(tmp_path: Path) -> None:
    out_of_range = tmp_path / "oor.json"
    _shard(out_of_range, [_row("e1")], shard_index=2, num_shards=2)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([out_of_range])
    assert "shard_index=2" in str(excinfo.value)
    assert "[0, 2)" in str(excinfo.value)

    negative = tmp_path / "neg.json"
    _shard(negative, [_row("e1")], shard_index=-1, num_shards=2)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([negative])
    assert "shard_index=-1" in str(excinfo.value)


def test_merge_rejects_boolean_index_and_count(tmp_path: Path) -> None:
    bool_index = tmp_path / "bool_index.json"
    _shard(bool_index, [_row("e1")], shard_index=True, num_shards=2)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([bool_index])
    assert "non-boolean integer" in str(excinfo.value)
    assert "shard_index" in str(excinfo.value)

    bool_count = tmp_path / "bool_count.json"
    _shard(bool_count, [_row("e1")], shard_index=0, num_shards=True)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([bool_count])
    assert "non-boolean integer" in str(excinfo.value)
    assert "num_shards" in str(excinfo.value)

    zero_count = tmp_path / "zero_count.json"
    _shard(zero_count, [_row("e1")], shard_index=0, num_shards=0)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([zero_count])
    assert "positive integer" in str(excinfo.value)


def test_merge_accepts_complete_valid_shard_set(tmp_path: Path) -> None:
    paths = []
    for index in range(8):
        path = tmp_path / f"b2_shard{index}of8.json"
        _shard(path, [_row(f"e{index}")], shard_index=index, num_shards=8)
        paths.append(path)
    merged = merge_shards.merge_shards(paths)
    assert [row["example_id"] for row in merged["rows"]] == [f"e{i}" for i in range(8)]
    assert merged["run_spec"]["shard_index"] is None
    assert merged["run_spec"]["num_shards"] is None


def test_merge_accepts_valid_empty_shards_when_count_exceeds_examples(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    two = tmp_path / "shard2.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=3)
    _shard(one, [], shard_index=1, num_shards=3)
    _shard(two, [], shard_index=2, num_shards=3)
    merged = merge_shards.merge_shards([zero, one, two])
    assert [row["example_id"] for row in merged["rows"]] == ["e1"]
    assert merged["summary"]["EM"] == 1.0


def test_merge_rejects_rows_missing_from_baseline_expected_ids(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    baseline = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=2)
    _shard(one, [], shard_index=1, num_shards=2)
    _shard(baseline, [_row("e1"), _row("e2")], shard_index=0)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([zero, one], expected_example_ids=["e1", "e2"])
    assert "do not match the expected example ids" in str(excinfo.value)


def test_merge_rejects_entirely_empty_shard_set(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    _shard(zero, [], shard_index=0, num_shards=2)
    _shard(one, [], shard_index=1, num_shards=2)
    with pytest.raises(SystemExit) as excinfo:
        merge_shards.merge_shards([zero, one])
    assert "no rows" in str(excinfo.value)
    assert "entirely empty result" in str(excinfo.value)


def test_merge_all_empty_does_not_create_output(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    _shard(zero, [], shard_index=0, num_shards=2)
    _shard(one, [], shard_index=1, num_shards=2)
    out = tmp_path / "b2_merged.json"
    with pytest.raises(SystemExit, match="requires --baseline"):
        merge_shards.main(["--out", str(out), str(zero), str(one)])
    assert not out.exists()


def test_merge_all_empty_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    _shard(zero, [], shard_index=0, num_shards=2)
    _shard(one, [], shard_index=1, num_shards=2)
    out = tmp_path / "b2_merged.json"
    out.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit, match="requires --baseline"):
        merge_shards.main(["--out", str(out), str(zero), str(one)])
    assert out.read_text(encoding="utf-8") == "existing"


def test_merge_allows_empty_merge_only_with_explicit_empty_expected_ids(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    _shard(zero, [], shard_index=0, num_shards=2)
    _shard(one, [], shard_index=1, num_shards=2)
    merged = merge_shards.merge_shards([zero, one], expected_example_ids=[])
    assert merged["rows"] == []
    assert merged["summary"]["EM"] == 0.0
    assert merged["run_spec"]["shard_index"] is None


def test_merge_rejects_expected_ids_with_duplicates(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=2)
    _shard(one, [_row("e2")], shard_index=1, num_shards=2)
    with pytest.raises(SystemExit, match="expected example ids contain duplicates"):
        merge_shards.merge_shards([zero, one], expected_example_ids=["e1", "e1"])
    with pytest.raises(SystemExit, match="expected example id at index 0 is empty"):
        merge_shards.merge_shards([zero, one], expected_example_ids=["", "e2"])


def test_merge_rejects_baseline_with_duplicate_ids(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    baseline = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=2)
    _shard(one, [_row("e2")], shard_index=1, num_shards=2)
    _shard(baseline, [_row("e1"), _row("e1")], shard_index=0)
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit, match="duplicate example_ids"):
        merge_shards.main(["--out", str(out), "--baseline", str(baseline), str(zero), str(one)])
    assert not out.exists()


def test_merge_rejects_baseline_with_missing_or_empty_ids(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    baseline = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=2)
    _shard(one, [_row("e2")], shard_index=1, num_shards=2)
    empty_id = _row("e1")
    empty_id["example_id"] = ""
    _shard(baseline, [empty_id], shard_index=0)
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit, match="missing or empty example_id"):
        merge_shards.main(["--out", str(out), "--baseline", str(baseline), str(zero), str(one)])
    assert not out.exists()


def test_merge_rejects_baseline_that_is_not_an_object(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    baseline = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0, num_shards=2)
    _shard(one, [_row("e2")], shard_index=1, num_shards=2)
    baseline.write_text("[1, 2]", encoding="utf-8")
    out = tmp_path / "merged.json"
    with pytest.raises(SystemExit, match="must be a JSON object"):
        merge_shards.main(["--out", str(out), "--baseline", str(baseline), str(zero), str(one)])
    assert not out.exists()


def test_merge_rejects_baseline_with_mismatched_provenance(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    baseline = tmp_path / "b0.json"
    out = tmp_path / "merged.json"
    _shard(zero, [_row("e1")], shard_index=0)
    _shard(one, [_row("e2")], shard_index=1)
    payload = _make_unsharded_b0(baseline, [_row("e1"), _row("e2")])
    payload["run_spec"]["manifest_sha256"] = "b" * 64
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="manifest_sha256.*does not match"):
        merge_shards.main(
            ["--out", str(out), "--baseline", str(baseline), str(zero), str(one)]
        )
    assert not out.exists()


def test_merge_rejects_sharded_or_non_b0_baseline(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    baseline = tmp_path / "b0.json"
    out = tmp_path / "merged.json"
    _shard(zero, [_row("e1")], shard_index=0)
    _shard(one, [_row("e2")], shard_index=1)
    _shard(baseline, [_row("e1"), _row("e2")], shard_index=0)

    with pytest.raises(SystemExit, match="naive_rag B0"):
        merge_shards.main(
            ["--out", str(out), "--baseline", str(baseline), str(zero), str(one)]
        )
    payload = _make_unsharded_b0(baseline, [_row("e1"), _row("e2")])
    payload["run_spec"]["shard_index"] = 0
    payload["run_spec"]["num_shards"] = 2
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="unsharded merged B0"):
        merge_shards.main(
            ["--out", str(out), "--baseline", str(baseline), str(zero), str(one)]
        )
    assert not out.exists()


def test_b0_cli_requires_manifest_and_rejects_partial_rows(tmp_path: Path) -> None:
    zero = tmp_path / "b0_shard0.json"
    one = tmp_path / "b0_shard1.json"
    manifest = tmp_path / "test.json"
    out = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0)
    _shard(one, [], shard_index=1)
    manifest_hash = _manifest(manifest, ["e1", "e2"])
    _set_manifest_hash([zero, one], manifest_hash)

    with pytest.raises(SystemExit, match="do not match the expected example ids"):
        merge_shards.main(
            ["--out", str(out), "--manifest", str(manifest), str(zero), str(one)]
        )
    assert not out.exists()


def test_b0_cli_accepts_exact_manifest_selection(tmp_path: Path) -> None:
    zero = tmp_path / "b0_shard0.json"
    one = tmp_path / "b0_shard1.json"
    manifest = tmp_path / "test.json"
    out = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0)
    _shard(one, [_row("e2")], shard_index=1)
    manifest_hash = _manifest(manifest, ["e2", "e1", "e3"])
    _set_manifest_hash([zero, one], manifest_hash)
    for path in (zero, one):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_spec"]["max_examples"] = 2
        path.write_text(json.dumps(payload), encoding="utf-8")

    assert merge_shards.main(
        ["--out", str(out), "--manifest", str(manifest), str(zero), str(one)]
    ) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [row["example_id"] for row in payload["rows"]] == ["e1", "e2"]
    assert payload["summary"]["harm_rate"] == 0.0


def test_b0_cli_rejects_manifest_hash_dataset_and_split_mismatch(tmp_path: Path) -> None:
    zero = tmp_path / "b0_shard0.json"
    one = tmp_path / "b0_shard1.json"
    manifest = tmp_path / "test.json"
    out = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0)
    _shard(one, [_row("e2")], shard_index=1)
    manifest_hash = _manifest(manifest, ["e1", "e2"])
    _set_manifest_hash([zero, one], manifest_hash)

    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 does not match"):
        merge_shards.main(
            ["--out", str(out), "--manifest", str(manifest), str(zero), str(one)]
        )

    manifest_hash = _manifest(manifest, ["e1", "e2"], dataset="arxivqa")
    _set_manifest_hash([zero, one], manifest_hash)
    with pytest.raises(SystemExit, match="manifest dataset"):
        merge_shards.main(
            ["--out", str(out), "--manifest", str(manifest), str(zero), str(one)]
        )

    manifest_hash = _manifest(manifest, ["e1", "e2"], split="val")
    _set_manifest_hash([zero, one], manifest_hash)
    with pytest.raises(SystemExit, match="manifest split"):
        merge_shards.main(
            ["--out", str(out), "--manifest", str(manifest), str(zero), str(one)]
        )
    assert not out.exists()


def test_b0_cli_rejects_malformed_manifest_ids(tmp_path: Path) -> None:
    zero = tmp_path / "b0_shard0.json"
    one = tmp_path / "b0_shard1.json"
    manifest = tmp_path / "test.json"
    out = tmp_path / "b0.json"
    _shard(zero, [_row("e1")], shard_index=0)
    _shard(one, [_row("e2")], shard_index=1)
    manifest_hash = _manifest(manifest, ["e1", "e1"])
    _set_manifest_hash([zero, one], manifest_hash)

    with pytest.raises(SystemExit, match="duplicate example_ids"):
        merge_shards.main(
            ["--out", str(out), "--manifest", str(manifest), str(zero), str(one)]
        )
    assert not out.exists()


def test_merge_rejects_top_level_profile_or_label_mismatch(tmp_path: Path) -> None:
    zero = tmp_path / "shard0.json"
    one = tmp_path / "shard1.json"
    _shard(zero, [_row("e1")], shard_index=0)
    payload = _shard(one, [_row("e2")], shard_index=1)
    payload["profile"] = "naive_rag"
    one.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="top-level profile"):
        merge_shards.merge_shards([zero, one])

    payload["profile"] = payload["run_spec"]["profile"]
    payload["label"] = "wrong label"
    one.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match.*label"):
        merge_shards.merge_shards([zero, one])
