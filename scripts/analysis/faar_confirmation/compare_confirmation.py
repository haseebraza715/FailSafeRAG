"""Compare the frozen confirmation runs without sending requests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/analysis/faar_long_dev"))
import compare_round as shared  # noqa: E402

PILOT = "ohr_val_multipage_v1"
PROVIDER_SHA = "35ea0799e1931fe0167224b5b05483f8881c91b3d1d0f3055c72910906b3ae6e"
PREFIX = "2026-11-01-ohr-val-multipage-v1-"
RUNS = {"baseline": PREFIX + "baseline-answer-v2-terra", "diagnosis": PREFIX + "diagnosis",
        "invoke_vlm": PREFIX + "visual-terra", "retry_retrieval": PREFIX + "retry-keep-top3-terra",
        "correct_text": PREFIX + "correct-symspell-terra"}
OUTPUT = "results/analysis/2026-11-01-ohr-val-multipage-v1-confirmation"
RESEND = "one-resend-connection-lost-v1"
CODE_FILES = {
    "diagnosis": {"scripts/experiments/diagnose_run.py", *[f"src/faar/{p}.py" for p in ("diagnosis", "quality", "text_repair")]},
    "invoke_vlm": {"scripts/experiments/run_visual_repair.py", *[f"src/faar/{p}.py" for p in ("visual_repair", "answer_prompt", "answer_providers", "request_budget")]},
    "retry_retrieval": {"scripts/experiments/run_text_repair.py", "scripts/experiments/run_visual_repair.py", *[f"src/faar/{p}.py" for p in ("text_repair", "answer_prompt", "answer_providers", "request_budget", "pilot_runner", "retrieval", "symspell")]},
}
CODE_FILES["correct_text"] = CODE_FILES["retry_retrieval"]


def require(value: Any, expected: Any, field: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise shared.ComparisonError(f"{field} mismatch")


def check_code(reader: shared.Reader, configs: dict[str, dict[str, Any]]) -> None:
    baseline_code = configs["baseline"].get("identity", {}).get("code", {})
    files = sorted((ROOT / "src/faar").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + b"\0" + reader.bytes(path))
    require(baseline_code.get("measurement_code_digest"), digest.hexdigest(), "measurement_code_digest")
    cli = ROOT / "scripts/experiments/run_pilot_live.py"
    require(baseline_code.get("cli_script_sha256"), hashlib.sha256(reader.bytes(cli)).hexdigest(), "cli_script_sha256")
    for name in ("diagnosis", *shared.REPAIRS):
        code = configs[name].get("code")
        if not isinstance(code, dict) or set(code) != CODE_FILES[name]:
            raise shared.ComparisonError(f"{name} code missing")
        for relative, expected in code.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                raise shared.ComparisonError(f"{name} code path {relative} invalid")
            require(hashlib.sha256(reader.bytes(path)).hexdigest(), expected, f"{name} code {relative}")


def check_settings(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    diag, visual = configs["diagnosis"], configs["invoke_vlm"]
    checks = [
        (diag.get("gate_rule", {}).get("feature"), "query_token_coverage", "gate_rule.feature"),
        (diag.get("gate_rule", {}).get("threshold"), 0.6923076923076923, "gate_rule.threshold"),
        (diag.get("structural_threshold"), 1, "structural_threshold"),
        (diag.get("weird_char_threshold"), 0.10, "weird_char_threshold"),
        (diag.get("image_budget"), 3, "diagnosis.image_budget"),
        (visual.get("image_budget"), 3, "image_budget"),
        (visual.get("page_policy"), "retrieved-then-gaps-v1", "page_policy"),
        *[(visual.get("render", {}).get(k), v, f"render.{k}") for k, v in
          {"dpi": 150, "format": "JPEG", "quality": 90, "long_side_max": 2048, "detail": "high"}.items()],
        (configs["retry_retrieval"].get("method", {}).get("name"), "keep-top3-add-next2-v1", "retry rule"),
        (configs["correct_text"].get("method", {}).get("name"), "symspell-en-82765", "corrector"),
        (configs["baseline"].get("identity", {}).get("prompt", {}).get("template_id"), "faar-answer-v2", "template_id"),
        *[(configs[r].get("template_id"), "faar-answer-v2", f"{r}.template_id") for r in shared.REPAIRS],
        *[(configs[r].get("resend_rule"), RESEND, f"{r}.resend_rule") for r in shared.REPAIRS],
        (list(shared.RANDOM_SEEDS), [20261101, 20261102, 20261103], "random seeds"),
    ]
    for actual, expected, field in checks:
        require(actual, expected, field)
    return {field: expected for _, expected, field in checks}


def compare(root: Path) -> dict[str, Any]:
    out = root / OUTPUT
    if out.exists() or out.is_symlink():
        raise shared.ComparisonError("output directory exists")
    reader = shared.Reader()
    provider = root / ".local/work/provider-config.gpt-6.1-terra.json"
    require(hashlib.sha256(reader.bytes(provider)).hexdigest(), PROVIDER_SHA, "provider config sha256")
    dirs = {name: root / "results/development" / run for name, run in RUNS.items()}
    configs = {name: reader.json(path / "run_config.json") for name, path in dirs.items()}
    settings = check_settings(configs)
    check_code(reader, configs)
    data = shared.read_inputs(root, PILOT, dirs["baseline"], dirs["diagnosis"],
                              {name: dirs[name] for name in shared.REPAIRS}, reader, provider, allowed_split="validation")
    rows = shared.cases(data)
    result = shared.analyse(data, rows, PILOT, 1)
    result["decision"]["label"] = "Confirmation result (pre-registered, section 5)"
    if not result["validity"]["valid"]:
        result["decision"]["met"] = None
    result["frozen_settings_checked"] = settings
    result["provider_config_sha256"] = data["provider_config_sha256"]
    result["provider_fields_checked"] = data["provider_fields_checked"]
    reader.bytes(Path(__file__))
    result["input_and_code_sha256"] = reader.hashes
    result["checkout"] = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                          "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}
    report = shared.report(result)
    report = report.replace(f"# {PILOT} round 1 policy comparison", f"# {PILOT} confirmation policy comparison")
    report = report.replace("## Development decision", "## Confirmation decision")
    if not result["validity"]["valid"]:
        report = report.replace(result["decision"]["rule"] + "\nRule met: None. Valid comparison: False.",
                                "Invalid comparison under amendment 2. No decision.")
    oracle, visual = result["policies"]["oracle"], result["policies"]["gated_visual"]
    report += ("\n## Descriptive headroom (not a claim)\n\n"
               "The oracle selects the best of the initial answer and repairs in hindsight using gold answers and is not deployable; this is descriptive only.\n\n"
               "| Policy | EM | Total USD per question |\n| --- | ---: | ---: |\n"
               f"| Oracle | {oracle['em']:.6f} | {oracle['cost']['per_question_mean']['total']:.6f} |\n"
               f"| Gated visual repair | {visual['em']:.6f} | {visual['cost']['per_question_mean']['total']:.6f} |\n")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "case_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        compare(args.project_root.resolve())
    except (shared.ComparisonError, OSError, KeyError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"comparison refused: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
