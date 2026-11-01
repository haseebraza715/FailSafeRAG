"""Tests for the FAAR gate and diagnosis, written from their failure list.

D1. ``missing_text`` does not fire for a question with no text evidence, or for an empty page the policy shows.
D2. The order is wrong: a later diagnosis wins over an earlier one.
D3. The gate drops a reason, flags without one, or applies the confidence condition when the feature is missing.
D4. A half-specified or unknown confidence condition is accepted.
D5. A confidence feature is computed wrongly (coverage, margin with one chunk, empty retrieval).
D6. A sent question without chunks is diagnosed silently.
D7. The flagged list differs from the diagnosis rows, their order, or its recorded hash and count.
"""

from __future__ import annotations

import pytest

from faar.diagnosis import (
    GATE_BASE_ID,
    DiagnosisError,
    GateRule,
    RetrievedChunk,
    confidence_features,
    diagnose,
    gate,
)

CLEAN = "The agreement was signed by the parties on 4 May 2005 in Seattle."
TABLE = "a | b | c | d | e | f | g | h | i | j"  # pipe_heavy
NOISY = "Th3 agr33m3nt w4s s1gn3d by th3 p4rt13s 0n 4 M4y"


def pages(*statuses: str) -> list[dict]:
    return [{"page_idx": i, "ocr_status": s} for i, s in enumerate(statuses)]


def chunk(rank: int, page: int, text: str = CLEAN, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(rank, page, score, text)


def test_missing_text_fires_without_text_evidence_and_on_a_shown_empty_page() -> None:
    assert diagnose(initial_sent=False, chunks=[], pages=pages("empty"))["diagnosis"] == "missing_text"
    shown_gap = diagnose(initial_sent=True, chunks=[chunk(1, 0)], pages=pages("ok", "empty"))
    assert shown_gap["diagnosis"] == "missing_text" and shown_gap["route"] == "invoke_vlm"
    assert shown_gap["signals"]["shown_pages_with_ocr_gap"] == [1]
    # an empty page the budget does not reach does not fire
    far = diagnose(initial_sent=True, chunks=[chunk(1, 0), chunk(2, 1), chunk(3, 2)], pages=pages("ok", "ok", "ok", "empty"))
    assert far["diagnosis"] == "semantic"


def test_diagnoses_are_tried_in_the_declared_order() -> None:
    both = TABLE + " " + NOISY
    assert diagnose(initial_sent=True, chunks=[chunk(1, 0, both)], pages=pages("ok", "empty"))["diagnosis"] == "missing_text"
    assert diagnose(initial_sent=True, chunks=[chunk(1, 0, both)], pages=pages("ok"))["diagnosis"] == "structural"
    word = diagnose(initial_sent=True, chunks=[chunk(1, 0, NOISY)], pages=pages("ok"))
    assert (word["diagnosis"], word["route"]) == ("word_level", "correct_text")
    plain = diagnose(initial_sent=True, chunks=[chunk(1, 0)], pages=pages("ok"))
    assert (plain["diagnosis"], plain["route"]) == ("semantic", "retry_retrieval")
    # the top chunk decides, whatever lower-ranked chunks hold
    assert diagnose(initial_sent=True, chunks=[chunk(2, 0, NOISY), chunk(1, 0)], pages=pages("ok"))["diagnosis"] == "semantic"


def test_gate_reports_every_reason_and_only_real_ones() -> None:
    features = {"top_fused_score": 0.9, "fused_margin": 0.1, "query_token_coverage": 0.4}
    base = GateRule()
    assert gate(initial_status="answered", abstained=False, pages=pages("ok"), features=features, rule=base) == (False, [])
    assert gate(initial_status="no_evidence", abstained=True, pages=pages("ok", "missing"), features=features, rule=base) == (
        True, ["no_evidence", "abstained", "document_ocr_gap"])
    low = GateRule("query_token_coverage", 0.5)
    assert gate(initial_status="answered", abstained=False, pages=pages("ok"), features=features, rule=low) == (True, ["low_query_token_coverage"])
    at = GateRule("query_token_coverage", 0.4)  # strictly below flags
    assert gate(initial_status="answered", abstained=False, pages=pages("ok"), features=features, rule=at) == (False, [])
    none = dict.fromkeys(features)
    assert gate(initial_status="answered", abstained=False, pages=pages("ok"), features=none, rule=low) == (False, [])
    assert base.gate_id == GATE_BASE_ID and low.gate_id == f"{GATE_BASE_ID}+query_token_coverage<0.5"


def test_a_confidence_condition_needs_a_known_feature_and_a_threshold() -> None:
    with pytest.raises(DiagnosisError):
        GateRule("query_token_coverage", None)
    with pytest.raises(DiagnosisError):
        GateRule(None, 0.5)
    with pytest.raises(DiagnosisError):
        GateRule("gold_overlap", 0.5)


def test_confidence_features() -> None:
    chunks = [chunk(2, 0, "signed in Seattle", 0.6), chunk(1, 0, "the agreement", 0.9)]
    f = confidence_features("Where was the agreement signed?", chunks, "multilingual-v1")
    assert f["top_fused_score"] == 0.9 and f["fused_margin"] == pytest.approx(0.3)
    assert 0 < f["query_token_coverage"] < 1
    assert confidence_features("anything", [chunk(1, 0, score=0.7)], "multilingual-v1")["fused_margin"] == 0.7
    assert confidence_features("anything", [], "multilingual-v1") == dict.fromkeys(("top_fused_score", "fused_margin", "query_token_coverage"))


def test_a_sent_question_without_chunks_is_refused() -> None:
    with pytest.raises(DiagnosisError):
        diagnose(initial_sent=True, chunks=[], pages=pages("ok"))


def test_flagged_file_matches_diagnosis_and_config(tmp_path, monkeypatch) -> None:
    import hashlib
    import json

    import diagnose_run as dr
    from test_text_repair import fake_replay
    from test_visual_repair import PILOT, Project

    project = Project(tmp_path)
    predictions = [{"question_id": q, "status": "no_evidence" if q == "q3" else "answered", "abstained": False}
                   for q in ("q1", "q2", "q3")]
    (project.baseline / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in predictions))
    replay = fake_replay(project)
    loaded = dr.rvr.load_baseline(project.root, PILOT, project.baseline)
    for doc in loaded["runtime"]["documents"]:
        for page in doc["pages"]:
            page["ocr_status"] = "ok"
    monkeypatch.setattr(dr.rvr, "load_baseline", lambda *args: loaded)
    monkeypatch.setattr(dr, "replay_retrieval", lambda *args: replay)
    out = project.run_dir("diagnosis")
    rows = dr.diagnose_run(root=project.root, pilot_id=PILOT, baseline_run_dir=project.baseline, out_dir=out, rule=GateRule())
    flagged = (out / "flagged_questions.txt").read_bytes()
    assert flagged == b"q3\n"
    assert flagged == "".join(r["question_id"] + "\n" for r in rows if r["flagged"]).encode()
    assert [json.loads(line) for line in (out / "diagnosis.jsonl").read_text().splitlines()] == rows
    config = json.loads((out / "flagged_questions_meta.json").read_text())
    assert config["flagged_questions_sha256"] == hashlib.sha256(flagged).hexdigest()
    assert config["flagged_count"] == sum(r["flagged"] for r in rows)
