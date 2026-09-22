from __future__ import annotations

import json
from pathlib import Path

from .final_analysis import load_result


def _format(value: float) -> str:
    return f"{float(value):.4f}"


def _latex_table(columns: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    alignment = "l" + "r" * (len(columns) - 1)
    header = " & ".join(columns) + " \\\\"
    body = "\n".join(" & ".join(row) + " \\\\" for row in rows)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{alignment}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def build_main_results_table(paths: list[Path]) -> str:
    if len(paths) != 3:
        raise ValueError("Table 1 requires exactly the three FAAR main-result JSON files.")
    rows: list[list[str]] = []
    for path in paths:
        result = load_result(path)
        spec = result.get("run_spec") or {}
        if not spec.get("dataset"):
            raise ValueError(f"Main result {path} lacks dataset provenance.")
        summary = result["summary"]
        rows.append(
            [
                str(spec["dataset"]),
                _format(summary["EM"]),
                _format(summary["F1"]),
                _format(summary["vlm_rate"]),
                _format(summary["cost_usd"]),
                _format(summary["runtime_sec"]),
            ]
        )
    return _latex_table(
        ["Dataset", "EM", "F1", "VLM rate", "Cost USD", "Runtime s"],
        rows,
        "Main FAAR results from the pinned experimental stack.",
        "tab:main-results",
    )


def build_ablation_table(paths: list[Path]) -> str:
    if len(paths) != 4:
        raise ValueError("Table 2 requires exactly the four ablation result JSON files.")
    rows: list[list[str]] = []
    for path in paths:
        result = load_result(path)
        summary = result["summary"]
        rows.append(
            [
                str(result.get("label") or path.stem),
                _format(summary["EM"]),
                _format(summary["F1"]),
                _format(summary["vlm_rate"]),
                _format(summary["harm_rate"]),
            ]
        )
    return _latex_table(
        ["Ablation", "EM", "F1", "VLM rate", "Harm rate"],
        rows,
        "FAAR ablation results.",
        "tab:ablations",
    )


def build_failure_type_table(analysis_path: Path, result_label: str) -> str:
    payload = json.loads(analysis_path.read_text())
    result = (payload.get("results") or {}).get(result_label)
    if not result:
        raise ValueError(f"Phase 6 analysis does not contain result label {result_label!r}.")
    breakdown = result.get("failure_type_breakdown") or {}
    if not breakdown:
        raise ValueError("Phase 6 analysis has no per-failure-type breakdown.")
    rows = [
        [
            failure_type,
            str(values["count"]),
            _format(values["EM"]),
            _format(values["F1"]),
            _format(values["harm_rate"]),
        ]
        for failure_type, values in sorted(breakdown.items())
    ]
    return _latex_table(
        ["Failure type", "N", "EM", "F1", "Harm rate"],
        rows,
        "FAAR performance by diagnosed failure type.",
        "tab:failure-types",
    )


def write_paper_artifacts(
    main_paths: list[Path],
    ablation_paths: list[Path],
    analysis_path: Path,
    pareto_path: Path,
    result_label: str,
    out_dir: Path,
) -> dict[str, str]:
    if not pareto_path.exists():
        raise FileNotFoundError(f"Figure 1 is missing: {pareto_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table1": out_dir / "table1_main_results.tex",
        "table2": out_dir / "table2_ablations.tex",
        "table3": out_dir / "table3_failure_types.tex",
        "figure1": out_dir / "figure1_cost_accuracy_pareto.pdf",
    }
    outputs["table1"].write_text(build_main_results_table(main_paths))
    outputs["table2"].write_text(build_ablation_table(ablation_paths))
    outputs["table3"].write_text(build_failure_type_table(analysis_path, result_label))
    outputs["figure1"].write_bytes(pareto_path.read_bytes())
    return {key: str(value) for key, value in outputs.items()}
