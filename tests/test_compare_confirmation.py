"""Failure list:

1. A changed frozen setting, code hash or provider hash is accepted.
2. A different pilot or split is accepted, or an existing output is overwritten.
3. The confirmation report omits headroom or makes a decision on an invalid run.
"""

import hashlib
import importlib.util
import shutil
from pathlib import Path

import pytest
from test_compare_round import change_json, jsonl, read_jsonl, sha
from test_compare_round import sample as round_sample  # noqa: F401

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/faar_confirmation/compare_confirmation.py"
spec = importlib.util.spec_from_file_location("compare_confirmation", SCRIPT)
confirmation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation)


@pytest.fixture
def frozen(request, monkeypatch):
    sample = request.getfixturevalue("round_sample")
    root = sample["root"]
    pilot = root / "results/pilots" / confirmation.PILOT
    shutil.move(root / "results/pilots/pilot", pilot)
    for filename in ("selection_record.json", "runtime_manifest.json", "evaluation_manifest.json"):
        change_json(pilot / filename, lambda x: x.update(pilot_id=confirmation.PILOT))
    change_json(pilot / "selection_record.json", lambda x: x["config"]["selection"].update(question_splits=["validation"]))
    runtime_sha = sha(pilot / "runtime_manifest.json")
    dirs = {}
    for name in confirmation.RUNS:
        old = sample["baseline_dir"] if name == "baseline" else sample["diagnosis_dir"] if name == "diagnosis" else sample["repair_dirs"][name]
        new = root / "results/development" / confirmation.RUNS[name]
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(old, new)
        dirs[name] = new
    provider = root / ".local/work/provider-config.gpt-6.1-terra.json"
    provider.parent.mkdir(parents=True)
    shutil.copy(sample["provider_config"], provider)
    monkeypatch.setattr(confirmation, "PROVIDER_SHA", sha(provider))
    base = dirs["baseline"]
    change_json(base / "run_config.json", lambda x: x["identity"].update(runtime_manifest_sha256=runtime_sha))
    change_json(base / "run_summary.json", lambda x: x.update(pilot_id=confirmation.PILOT))
    code_files = sorted((confirmation.ROOT / "src/faar").glob("*.py"))
    digest = hashlib.sha256()
    for path in code_files:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    change_json(base / "run_config.json", lambda x: x["identity"].update(code={"measurement_code_digest": digest.hexdigest(), "cli_script_sha256": sha(confirmation.ROOT / "scripts/experiments/run_pilot_live.py")}))
    for name in ("diagnosis", *confirmation.shared.REPAIRS):
        path = dirs[name] / "run_config.json"
        change_json(path, lambda x: x.update(pilot_id=confirmation.PILOT, baseline_run=base.name, runtime_manifest_sha256=runtime_sha,
                                             code={p: sha(confirmation.ROOT / p) for p in confirmation.CODE_FILES[name]}))
        if name != "diagnosis":
            change_json(path, lambda x: x.update(provider_config_sha256=sha(provider), resend_rule=confirmation.RESEND))
    change_json(dirs["diagnosis"] / "run_config.json", lambda x: x.update(gate_rule={"feature": "query_token_coverage", "threshold": 0.6923076923076923}, structural_threshold=1, weird_char_threshold=0.10, image_budget=3))
    change_json(dirs["invoke_vlm"] / "run_config.json", lambda x: x.update(image_budget=3, page_policy="retrieved-then-gaps-v1", render={"dpi": 150, "format": "JPEG", "quality": 90, "long_side_max": 2048, "detail": "high"}))
    change_json(dirs["retry_retrieval"] / "run_config.json", lambda x: x.update(method={"name": "keep-top3-add-next2-v1"}))
    change_json(dirs["correct_text"] / "run_config.json", lambda x: x.update(method={"name": "symspell-en-82765"}))
    return root, dirs, pilot, provider


def test_happy_path_and_headroom(frozen):
    root, _, _, _ = frozen
    result = confirmation.compare(root)
    assert result["decision"]["label"] == "Confirmation result (pre-registered, section 5)"
    assert result["checkout"]["commit"]
    text = (root / confirmation.OUTPUT / "report.md").read_text()
    assert "Descriptive headroom (not a claim)" in text
    assert "not deployable" in text
    assert result["input_and_code_sha256"]
    with pytest.raises(confirmation.shared.ComparisonError, match="output directory exists"):
        confirmation.compare(root)


@pytest.mark.parametrize("name,key,value", [
    ("diagnosis", "gate_rule", {"feature": "wrong", "threshold": 0.6923076923076923}),
    ("diagnosis", "gate_rule", {"feature": "query_token_coverage", "threshold": 0.7}),
    ("diagnosis", "structural_threshold", 2), ("diagnosis", "weird_char_threshold", 0.2),
    ("diagnosis", "image_budget", 2), ("invoke_vlm", "image_budget", 2),
    ("invoke_vlm", "page_policy", "wrong"),
    *[("invoke_vlm", "render", {"dpi": 150, "format": "JPEG", "quality": 90, "long_side_max": 2048, "detail": "high", k: v}) for k, v in [("dpi", 72), ("format", "PNG"), ("quality", 80), ("long_side_max", 1024), ("detail", "low")]],
    ("retry_retrieval", "method", {"name": "wrong"}), ("correct_text", "method", {"name": "wrong"}),
    *[(r, "template_id", "wrong") for r in confirmation.shared.REPAIRS],
    *[(r, "resend_rule", "wrong") for r in confirmation.shared.REPAIRS],
])
def test_frozen_setting_refused(frozen, name, key, value):
    root, dirs, _, _ = frozen
    change_json(dirs[name] / "run_config.json", lambda x: x.update({key: value}))
    with pytest.raises(confirmation.shared.ComparisonError):
        confirmation.compare(root)
    assert not (root / confirmation.OUTPUT).exists()


@pytest.mark.parametrize("kind", ["split", "pilot", "code", "provider"])
def test_provenance_refused(frozen, kind):
    root, dirs, pilot, provider = frozen
    if kind == "split":
        change_json(pilot / "selection_record.json", lambda x: x["config"]["selection"].update(question_splits=["train"]))
    elif kind == "pilot":
        change_json(pilot / "selection_record.json", lambda x: x.update(pilot_id="wrong"))
    elif kind == "code":
        change_json(dirs["diagnosis"] / "run_config.json", lambda x: x["code"].update({"src/faar/diagnosis.py": "wrong"}))
    else:
        provider.write_text("{}")
    with pytest.raises(confirmation.shared.ComparisonError):
        confirmation.compare(root)
    assert not (root / confirmation.OUTPUT).exists()


def test_invalid_run_has_no_decision(frozen):
    root, dirs, _, _ = frozen
    repair = dirs["retry_retrieval"]
    rows = read_jsonl(repair / "predictions.jsonl")
    rows[0].update(status="execution_failed", blocked=True, answer=None, usage=None)
    jsonl(repair / "predictions.jsonl", rows)
    events = read_jsonl(repair / "attempts.jsonl")
    event = next(e for e in events if e.get("attempt_id") == "retry_retrieval-q0" and e["event"] == "saved")
    event.update(event="failed", outcome="unknown", kind="timeout")
    event.pop("measured_cost")
    jsonl(repair / "attempts.jsonl", events)
    change_json(repair / "run_summary.json", lambda x: x.update(complete=False, blocked=["q0"], measured_cost_usd=0.4, reserved_cost_usd=0.5))
    result = confirmation.compare(root)
    assert result["decision"]["met"] is None
    report = (root / confirmation.OUTPUT / "report.md").read_text()
    assert "Invalid comparison under amendment 2. No decision." in report
    assert "Rule met:" not in report


@pytest.mark.parametrize("kind", ["directory", "broken_symlink"])
def test_existing_output_refused_before_reading_inputs(tmp_path, kind):
    out = tmp_path / confirmation.OUTPUT
    out.parent.mkdir(parents=True)
    if kind == "directory":
        out.mkdir()
    else:
        out.symlink_to(tmp_path / "missing")
    with pytest.raises(confirmation.shared.ComparisonError, match="output directory exists"):
        confirmation.compare(tmp_path)


def test_shared_parameters_preserve_confirmation_bytes(frozen):
    import subprocess

    root, _, _, _ = frozen
    source = subprocess.check_output(["git", "show", "76c9c21:scripts/analysis/faar_confirmation/compare_confirmation.py"], text=True)
    old = {"__file__": str(SCRIPT), "__name__": "old_confirmation"}
    exec(compile(source, str(SCRIPT), "exec"), old)
    old["PROVIDER_SHA"] = confirmation.PROVIDER_SHA
    old["compare"](root)
    out = root / confirmation.OUTPUT
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    shutil.rmtree(out)
    confirmation.compare(root)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
