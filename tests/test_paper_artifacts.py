import json
from pathlib import Path

from faar.paper_artifacts import write_paper_artifacts


def _result(path: Path, label: str, dataset: str) -> None:
    path.write_text(
        json.dumps(
            {
                "label": label,
                "run_spec": {"dataset": dataset},
                "summary": {"EM": 0.5, "F1": 0.6, "vlm_rate": 0.2, "harm_rate": 0.1, "cost_usd": 0.03, "runtime_sec": 2.0},
                "rows": [],
            }
        )
    )


def test_generates_all_tables_and_copies_real_pareto_pdf(tmp_path: Path) -> None:
    main_paths = [tmp_path / f"main_{index}.json" for index in range(3)]
    ablation_paths = [tmp_path / f"a{index}.json" for index in range(4)]
    for index, path in enumerate(main_paths):
        _result(path, f"main-{index}", f"dataset-{index}")
    for index, path in enumerate(ablation_paths):
        _result(path, f"a{index + 1}", "ohrbench")
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "results": {
                    "FAAR ohrbench test": {
                        "failure_type_breakdown": {"semantic": {"count": 2, "EM": 0.5, "F1": 0.6, "harm_rate": 0.1}}
                    }
                }
            }
        )
    )
    pareto = tmp_path / "pareto.pdf"
    pareto.write_bytes(b"real pdf bytes")
    outputs = write_paper_artifacts(main_paths, ablation_paths, analysis, pareto, "FAAR ohrbench test", tmp_path / "out")
    assert "dataset-0" in Path(outputs["table1"]).read_text()
    assert "a1" in Path(outputs["table2"]).read_text()
    assert Path(outputs["figure1"]).read_bytes() == b"real pdf bytes"
