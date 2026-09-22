import json
from pathlib import Path

import pytest

from faar.final_analysis import _harm_rate, bootstrap_ci, pareto_frontier, summarize_analysis

DEFAULT_API_USAGE = {
    "api_requests": 1,
    "prompt_tokens": 10,
    "completion_tokens": 5,
    "cost_usd": 0.0001,
}


def _run_spec(**overrides) -> dict:
    spec = {
        "dataset": "ohrbench",
        "split": "test",
        "seed": 42,
        "max_examples": None,
        "shard_index": None,
        "num_shards": None,
        "embedding_model": "nvidia/NV-Embed-v2",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "ocr_engine": "got-ocr-2",
        "model_provenance": {
            "embedding": {"repository": "nvidia/NV-Embed-v2", "revision": None},
            "reranker": {"repository": "BAAI/bge-reranker-v2-m3", "revision": None},
        },
        "manifest_sha256": "a" * 64,
        "vlm_backend": "openai",
        "vlm_model": "gpt-4o-2024-11-20",
        "vlm_cost_rates": {
            "provider": "openai",
            "currency": "USD",
            "input_usd_per_million_tokens": 2.5,
            "output_usd_per_million_tokens": 10.0,
        },
    }
    spec.update(overrides)
    return spec


def _row(example_id: str, f1: float, failure_type: str = "semantic") -> dict:
    return {
        "example_id": example_id,
        "failure_type": failure_type,
        "action_outcome": {"action": "invoke_vlm"},
        "metrics": {"em": f1, "f1": f1},
        "api_usage": dict(DEFAULT_API_USAGE),
    }


def _consistent_summary(rows: list[dict], baseline_rows: list[dict] | None = None) -> dict:
    count = len(rows)
    em = round(sum(float((row.get("metrics") or {}).get("em", 0.0)) for row in rows) / count, 4)
    f1 = round(sum(float((row.get("metrics") or {}).get("f1", 0.0)) for row in rows) / count, 4)
    vlm_rate = round(
        sum(
            1.0 if (row.get("action_outcome") or {}).get("action") == "invoke_vlm" else 0.0
            for row in rows
        )
        / count,
        4,
    )
    return {
        "EM": em,
        "F1": f1,
        "vlm_rate": vlm_rate,
        "harm_rate": round(_harm_rate(rows, baseline_rows), 4) if baseline_rows else 0.0,
        "api_requests": sum(int((row.get("api_usage") or {}).get("api_requests", 0)) for row in rows),
        "prompt_tokens": sum(int((row.get("api_usage") or {}).get("prompt_tokens", 0)) for row in rows),
        "completion_tokens": sum(
            int((row.get("api_usage") or {}).get("completion_tokens", 0)) for row in rows
        ),
        "cost_usd": round(
            sum(float((row.get("api_usage") or {}).get("cost_usd", 0.0)) for row in rows), 6
        ),
        "runtime_sec": 1.0,
    }


def _result(
    path: Path,
    label: str,
    rows: list[dict],
    *,
    profile: str = "faar_full",
    run_spec: dict | None = None,
    summary: dict | None = None,
    baseline_rows: list[dict] | None = None,
) -> None:
    if summary is None:
        summary = _consistent_summary(rows, baseline_rows)
    payload = {
        "label": label,
        "summary": summary,
        "rows": rows,
    }
    if profile is not None:
        payload["profile"] = profile
    if run_spec is not None:
        payload["run_spec"] = run_spec
    path.write_text(json.dumps(payload))


def test_analysis_uses_rows_for_bootstrap_harm_and_breakdown(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0), _row("e2", 0.5)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [_row("e1", 0.0), _row("e2", 1.0, "structural")],
        run_spec=spec,
        baseline_rows=baseline_rows,
    )
    analysis = summarize_analysis([faar], baseline)
    result = analysis["results"]["FAAR"]
    assert result["harm_rate"] == 0.5
    assert result["failure_type_breakdown"]["semantic"]["count"] == 1
    assert result["bootstrap_95_ci"]["F1"]["mean"] == 0.5
    assert result["summary"]["F1"] == 0.5


def test_analysis_passes_when_summary_matches_rows(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0), _row("e2", 0.5)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [_row("e1", 0.0), _row("e2", 0.75)],
        run_spec=spec,
        baseline_rows=baseline_rows,
    )
    analysis = summarize_analysis([faar], baseline)
    summary = analysis["results"]["FAAR"]["summary"]
    assert summary["EM"] == 0.375
    assert summary["F1"] == 0.375
    assert summary["vlm_rate"] == 1.0
    assert summary["harm_rate"] == 0.5
    assert summary["api_requests"] == 2
    assert summary["prompt_tokens"] == 20
    assert summary["completion_tokens"] == 10
    assert summary["cost_usd"] == 0.0002
    assert summary["runtime_sec"] == 1.0


def test_analysis_rejects_fabricated_summary_f1(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    fabricated = _consistent_summary([_row("e1", 0.0)], baseline_rows)
    fabricated["F1"] = 1.0
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=spec, summary=fabricated)
    with pytest.raises(ValueError, match="summary.F1"):
        summarize_analysis([faar], baseline)


def test_bootstrap_ci_is_seeded_and_pareto_discards_dominated_points() -> None:
    assert bootstrap_ci([0.0, 1.0], samples=100, seed=42) == bootstrap_ci([0.0, 1.0], samples=100, seed=42)
    frontier = pareto_frontier(
        [
            {"label": "cheap", "cost_usd": 0.0, "F1": 0.4},
            {"label": "better", "cost_usd": 0.1, "F1": 0.8},
            {"label": "dominated", "cost_usd": 0.2, "F1": 0.7},
        ]
    )
    assert [point["label"] for point in frontier] == ["cheap", "better"]


def test_analysis_rejects_baseline_that_is_not_naive_rag(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="faar_full", run_spec=spec)
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=spec, baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match="naive_rag"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_missing_baseline_run_spec(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=None)
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=_run_spec(), baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match="run_spec"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_missing_result_run_spec(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=_run_spec())
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=None, baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match="run_spec"):
        summarize_analysis([faar], baseline)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dataset", "paperQA"),
        ("split", "val"),
        ("seed", 7),
        ("max_examples", 50),
        ("shard_index", 1),
        ("num_shards", 2),
        ("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        ("reranker", "BAAI/bge-reranker-base"),
        ("ocr_engine", "tesseract"),
        ("model_provenance", {"embedding": {"repository": "other", "revision": None}}),
        ("manifest_sha256", "0" * 64),
    ],
)
def test_analysis_rejects_run_spec_key_mismatch(tmp_path: Path, key: str, value: object) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [_row("e1", 0.0)],
        run_spec=_run_spec(**{key: value}),
        baseline_rows=baseline_rows,
    )
    with pytest.raises(ValueError, match=f"run_spec.{key}"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_run_spec_missing_match_key(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    partial = dict(spec)
    del partial["shard_index"]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=partial, baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match="shard_index"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_duplicate_result_example_ids(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [_row("e1", 0.0), _row("e1", 0.3)],
        run_spec=spec,
        baseline_rows=baseline_rows,
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_failed_rows_even_when_metrics_exist(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    failed = _row("e1", 0.0)
    failed["action_outcome"] = {"action": "failed", "status": "failed"}
    failed["error"] = "FileNotFoundError: missing OCR text"
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", [failed], run_spec=spec, baseline_rows=baseline_rows)

    with pytest.raises(ValueError, match="failed row 'e1'"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_duplicate_baseline_example_ids(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    _result(
        baseline,
        "B0",
        [_row("e1", 1.0), _row("e1", 0.8)],
        profile="naive_rag",
        run_spec=spec,
    )
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=spec, baseline_rows=[_row("e1", 1.0)])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_missing_example_id(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [_row("e1", 0.0), _row("", 0.3)],
        run_spec=spec,
        summary=_consistent_summary([_row("e1", 0.0)], baseline_rows),
    )
    with pytest.raises(ValueError, match="missing example_id"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_row_not_covered_by_baseline(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [_row("e1", 0.0), _row("e9", 0.5)],
        run_spec=spec,
        summary=_consistent_summary([_row("e1", 0.0)], baseline_rows),
    )
    with pytest.raises(ValueError, match="not covered"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_result_that_omits_baseline_row(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0), _row("e2", 0.5)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=spec, baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match="omits baseline B0 rows"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_empty_result_rows(tmp_path: Path) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(
        faar,
        "FAAR",
        [],
        run_spec=spec,
        summary={"EM": 0.0, "F1": 0.0, "vlm_rate": 0.0, "harm_rate": 0.0, "cost_usd": 0.0},
    )
    with pytest.raises(ValueError, match="no result rows"):
        summarize_analysis([faar], baseline)


def test_analysis_rejects_duplicate_result_labels(tmp_path: Path) -> None:
    baseline = tmp_path / "b0.json"
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(first, "FAAR", [_row("e1", 0.0)], run_spec=spec, baseline_rows=baseline_rows)
    _result(second, "FAAR", [_row("e1", 1.0)], run_spec=spec, baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match="duplicate result label"):
        summarize_analysis([first, second], baseline)


def test_load_result_rejects_malformed_rows(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"summary": {}, "rows": {"e1": _row("e1", 1.0)}}))
    with pytest.raises(ValueError, match="complete FAAR result"):
        summarize_analysis([malformed], malformed)


def test_harm_rate_rejects_uncovered_row_directly() -> None:
    baseline_rows = [_row("e1", 1.0)]
    with pytest.raises(ValueError, match="not covered"):
        _harm_rate([_row("e1", 0.0), _row("e9", 0.5)], baseline_rows)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("vlm_backend", "claude-sonnet-4-5"),
        ("vlm_model", "claude-sonnet-4-5"),
        (
            "vlm_cost_rates",
            {
                "provider": "anthropic",
                "currency": "USD",
                "input_usd_per_million_tokens": 3.0,
                "output_usd_per_million_tokens": 15.0,
            },
        ),
    ],
)
def test_analysis_rejects_mixed_vlm_provenance_across_results(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    baseline, first, second = tmp_path / "b0.json", tmp_path / "first.json", tmp_path / "second.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(first, "FAAR", [_row("e1", 0.0)], run_spec=spec, baseline_rows=baseline_rows)
    _result(
        second,
        "FAAR-2",
        [_row("e1", 1.0)],
        run_spec=_run_spec(**{key: value}),
        baseline_rows=baseline_rows,
    )
    with pytest.raises(ValueError, match="vlm_backend, vlm_model, and vlm_cost_rates"):
        summarize_analysis([first, second], baseline)


@pytest.mark.parametrize("key", ["vlm_backend", "vlm_model", "vlm_cost_rates"])
def test_analysis_rejects_missing_vlm_provenance_key(tmp_path: Path, key: str) -> None:
    baseline, faar = tmp_path / "b0.json", tmp_path / "faar.json"
    spec = _run_spec()
    baseline_rows = [_row("e1", 1.0)]
    partial = dict(spec)
    del partial[key]
    _result(baseline, "B0", baseline_rows, profile="naive_rag", run_spec=spec)
    _result(faar, "FAAR", [_row("e1", 0.0)], run_spec=partial, baseline_rows=baseline_rows)
    with pytest.raises(ValueError, match=key):
        summarize_analysis([faar], baseline)
