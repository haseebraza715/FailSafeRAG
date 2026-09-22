from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .metrics import exact_match

LABELS = ("semantic", "word_level", "structural", "other")


def _is_failure(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics") or {}
    if "em" in metrics:
        return float(metrics["em"]) < 1.0
    return exact_match(
        str(row.get("predicted_answer", row.get("answer", ""))),
        str(row.get("gold_answer", row.get("correct_answer", ""))),
    ) < 1.0


def sample_failures(baseline_path: Path, n: int, seed: int = 42) -> list[dict[str, Any]]:
    payload = json.loads(baseline_path.read_text())
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    failures = [row for row in rows if _is_failure(row)]
    if len(failures) < n:
        raise ValueError(f"Requested {n} failures but {baseline_path} contains only {len(failures)} incorrect B0 examples.")
    selected = random.Random(seed).sample(failures, n)
    samples: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item.get("example_id", ""))):
        assets = row.get("source_assets") or {}
        samples.append(
            {
                "example_id": row.get("example_id"),
                "question": row.get("question", ""),
                "gold_answer": row.get("gold_answer", row.get("correct_answer", "")),
                "baseline_answer": row.get("predicted_answer", row.get("answer", "")),
                "top_reranker_score": row.get("top_reranker_score", (row.get("gate") or {}).get("top_reranker_score")),
                "image_paths": assets.get("image_paths", row.get("image_paths", [])),
                "source_ocr_path": assets.get("ocr_text_path"),
            }
        )
    return samples


def label_studio_config() -> str:
    return """<View>\n  <Header value=\"$question\"/>\n  <Text name=\"ocr_text\" value=\"$ocr_text\"/>\n  <Choices name=\"failure_type\" toName=\"ocr_text\" choice=\"single\" required=\"true\">\n    <Choice value=\"semantic\"/>\n    <Choice value=\"word_level\"/>\n    <Choice value=\"structural\"/>\n    <Choice value=\"other\"/>\n  </Choices>\n</View>\n"""


def write_label_studio_tasks(samples: list[dict[str, Any]], ocr_dir: Path, out_path: Path) -> None:
    tasks: list[dict[str, Any]] = []
    for sample in samples:
        example_id = str(sample["example_id"])
        ocr_path = ocr_dir / f"{example_id}.txt"
        if not ocr_path.exists():
            raise FileNotFoundError(f"Missing GOT-OCR text for Label Studio task: {ocr_path}")
        tasks.append(
            {
                "data": {
                    "example_id": example_id,
                    "question": sample["question"],
                    "gold_answer": sample["gold_answer"],
                    "baseline_answer": sample["baseline_answer"],
                    "ocr_text": ocr_path.read_text(errors="ignore"),
                }
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2) + "\n")


def load_label_studio_labels(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    labels: dict[str, str] = {}
    for task in payload:
        example_id = str((task.get("data") or {}).get("example_id", ""))
        annotations = task.get("annotations") or []
        if not example_id or len(annotations) != 1:
            raise ValueError(f"Expected exactly one annotation for task {example_id or '<missing id>'}.")
        choices = [
            choice
            for result in annotations[0].get("result", [])
            for choice in (result.get("value") or {}).get("choices", [])
        ]
        if len(choices) != 1 or choices[0] not in LABELS:
            raise ValueError(f"Task {example_id} must have exactly one label from: {', '.join(LABELS)}.")
        labels[example_id] = choices[0]
    return labels


def cohens_kappa(first: dict[str, str], second: dict[str, str]) -> dict[str, Any]:
    if set(first) != set(second):
        raise ValueError("The two annotation exports must cover the exact same example IDs.")
    if not first:
        raise ValueError("No shared annotation labels were provided.")
    ids = sorted(first)
    observed = sum(first[example_id] == second[example_id] for example_id in ids) / len(ids)
    first_counts = Counter(first.values())
    second_counts = Counter(second.values())
    expected = sum((first_counts[label] / len(ids)) * (second_counts[label] / len(ids)) for label in LABELS)
    kappa = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "count": len(ids),
        "observed_agreement": round(observed, 4),
        "expected_agreement": round(expected, 4),
        "cohens_kappa": round(kappa, 4),
        "threshold": 0.65,
        "passed": kappa >= 0.65,
    }
