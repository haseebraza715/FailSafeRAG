"""Tests for the visual repair runner, written from its failure list.

Ways the repair could fail, each covered below:

R1. A gold field reaches the repair: prepare or run opens the evaluation manifest, clean reference text or a score.
R2. The page policy differs from the design: wrong order, a duplicate page, a page outside the document, or more
    pages than the budget.
R3. A page is rendered from a PDF whose hash differs from the runtime manifest.
R4. The request differs from the design: the user message is not the no-recovery message, the no-text message is
    wrong, the system message lacks the image note, or the images come in another order than the policy.
R5. An unknown outcome is resent outside the rule, the rule resends more than once or another kind of failure, or
    a resent question's first attempt stops counting at its upper bound.
R6. The cap is crossed: a request is sent when measured plus reserved plus its bound exceeds the cap.
R7. A changed run reuses a run directory, or the repair runs on a baseline that is invalid, of another pilot or
    of another template.
R8. The gate differs from ``abstain-or-ocr-gap-v1``.
R9. A subset includes unlisted questions, accepts invalid ids or an empty file, or changes the full-run format.
R10. A subset cannot run, export and score only its own questions.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest
import run_visual_repair as rvr

from faar.answer_prompt import TEMPLATE_V1, TEMPLATE_V2
from faar.live_contract import ProviderError, ProviderResponse, ProviderUsage
from faar.live_runner import load_provider_config
from faar.visual_repair import IMAGE_NOTE, RepairError, gate_flags, select_pages

PILOT = "pilot_t"
CONFIG = {
    "provider": "openai", "model": "gpt-6.1-terra", "endpoint": "https://api.openai.com/v1",
    "params": {"temperature": 0, "reasoning_effort": "none"}, "max_output_tokens": 128, "max_input_tokens": 12000,
    "timeout_seconds": 60, "tokenizer_bound": "utf8-bytes", "token_limit_param": "max_completion_tokens",
    "prices": {"currency": "USD", "input_per_million": 1.0, "cached_input_per_million": 0.05, "output_per_million": 5.0,
               "source": "test", "source_date": "2026-10-31", "service_tier": "default"},
    "service_tier": "default", "store": False,
}


def _pdf(pages: int) -> bytes:
    import pypdfium2

    pdf = pypdfium2.PdfDocument.new()
    for i in range(pages):
        pdf.new_page(200 + 10 * i, 300)
    out = io.BytesIO()
    pdf.save(out)
    pdf.close()
    return out.getvalue()


def _pages(statuses: list[str]) -> list[dict[str, Any]]:
    return [{"page_idx": i, "ocr_status": s, "pdf_page_number": i + 1} for i, s in enumerate(statuses)]


class Project:
    """A tiny pilot with three questions on two documents, a no-recovery run and a PDF archive."""

    def __init__(self, root: Path, *, template=TEMPLATE_V2, valid_baseline: bool = True) -> None:
        self.root = root
        self.template = template
        pdfs = {"d/a": _pdf(4), "d/b": _pdf(1)}
        self.archive = root / "pdfs.zip"
        with zipfile.ZipFile(self.archive, "w") as z:
            for doc_id, data in pdfs.items():
                z.writestr(f"{doc_id}.pdf", data)
        docs = [
            {"doc_id": "d/a", "pages": _pages(["ok", "ok", "empty", "missing"]),
             "pdf": {"member": "d/a.pdf", "sha256": hashlib.sha256(pdfs["d/a"]).hexdigest()}},
            {"doc_id": "d/b", "pages": _pages(["empty"]),
             "pdf": {"member": "d/b.pdf", "sha256": hashlib.sha256(pdfs["d/b"]).hexdigest()}},
        ]
        questions = [{"question_id": "q1", "doc_id": "d/a", "question": "Who signed?"},
                     {"question_id": "q2", "doc_id": "d/a", "question": "When?"},
                     {"question_id": "q3", "doc_id": "d/b", "question": "What is written?"}]
        pilot = root / "results" / "pilots" / PILOT
        pilot.mkdir(parents=True)
        runtime = {"pilot_id": PILOT, "documents": docs, "questions": questions}
        (pilot / "runtime_manifest.json").write_text(json.dumps(runtime))
        manifest_sha = hashlib.sha256((pilot / "runtime_manifest.json").read_bytes()).hexdigest()
        answers = {"q1": "Ann", "q2": "May", "q3": "Ann Smith"}
        (pilot / "evaluation_manifest.json").write_text(json.dumps({"pilot_id": PILOT, "questions": {
            q["question_id"]: {"answers": answers[q["question_id"]], "evidence_pages": [0]} for q in questions}}))
        gt = root / "OHR-Bench" / "data" / "retrieval_base" / "gt" / "d" / "a.json"
        gt.parent.mkdir(parents=True)
        gt.write_text("[]")
        self.baseline = root / "results" / "development" / "base"
        self.baseline.mkdir(parents=True)
        (self.baseline / "run_config.json").write_text(json.dumps({"mode": "live", "identity": {"runtime_manifest": {"sha256": manifest_sha}, "run_kind": "development_pilot", "pilot_manifest": {"canonical": True}}}))
        (self.baseline / "run_summary.json").write_text(json.dumps({"pilot_id": PILOT, "valid_baseline": valid_baseline, "run_state": "complete", "valid_baseline_blockers": [] if valid_baseline else ["invalid"]}))
        (self.baseline / "scores.jsonl").write_text("{}\n")
        (self.baseline / "predictions.jsonl").write_text("".join(json.dumps({"question_id": q["question_id"], "status": "no_evidence" if q["question_id"] == "q3" else "answered", "abstained": False}) + "\n" for q in questions))
        evidence = {"q1": [{"rank": 1, "page_idx": 1}, {"rank": 2, "page_idx": 0}, {"rank": 3, "page_idx": 1}],
                    "q2": [{"rank": 1, "page_idx": 3}]}
        rows = []
        for q in questions:
            sent = q["question_id"] in evidence
            rows.append({"question_id": q["question_id"], "doc_id": q["doc_id"], "question": q["question"],
                         "action": "send" if sent else "skip", "skip_reason": None if sent else "no_text_chunks",
                         "template_id": template.template_id, "system": template.system if sent else None,
                         "user": f"USER MESSAGE {q['question_id']}" if sent else None,
                         "evidence": evidence.get(q["question_id"], [])})
        (self.baseline / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.provider_config = root / "provider.json"
        self.provider_config.write_text(json.dumps(CONFIG))
        self.config = load_provider_config(self.provider_config)

    def run_dir(self, name: str = "repair") -> Path:
        return self.root / "results" / "development" / name

    def prepare(self, name: str = "repair", **kwargs: Any) -> list[dict[str, Any]]:
        args = {"budget": 3, "resend_rule": "none"} | kwargs
        records, _ = rvr.prepare(root=self.root, pilot_id=PILOT, baseline_run_dir=self.baseline, run_dir=self.run_dir(name),
                                 run_id=name, provider_config=self.provider_config, archive=self.archive, **args)
        return records

    def run(self, provider: Any, name: str = "repair", cap: float = 10.0) -> dict[str, Any]:
        return rvr.run(root=self.root, run_dir=self.run_dir(name), cap=cap, provider=provider, config=self.config, archive=self.archive)


def ok_response(text: str = "Ann") -> ProviderResponse:
    return ProviderResponse(text, "stop", None, "gpt-6.1-terra", "r1", ProviderUsage(1000, 0, 3, 0), {}, "default")


def lost() -> ProviderError:
    return ProviderError("ReadError: bad record mac", kind="connection_lost", outcome="unknown", retryable=False)


def timeout() -> ProviderError:
    return ProviderError("timed out", kind="timeout", outcome="unknown", retryable=False)


class ScriptedProvider:
    """Plays each question's scripted outcomes in order, then answers. Records every request it receives."""

    def __init__(self, script: dict[str, list[Any]] | None = None) -> None:
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.requests: list[Any] = []

    def send(self, request: Any) -> ProviderResponse:
        self.requests.append(request)
        steps = self.script.get(request.request_id)
        step = steps.pop(0) if steps else ok_response()
        if isinstance(step, Exception):
            raise step
        return step

    def sent(self) -> list[str]:
        return [r.request_id for r in self.requests]


def events(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "attempts.jsonl").read_text().splitlines()]


# --------------------------------------------------------------------------- R2 page policy


def test_policy_takes_retrieved_pages_then_ocr_gaps_then_the_rest_up_to_the_budget() -> None:
    pages = _pages(["ok", "ok", "ok", "empty", "ok", "missing"])
    assert select_pages(pages, [4, 1, 4], 3) == [4, 1, 3]
    assert select_pages(pages, [4, 1, 4], 5) == [4, 1, 3, 5, 0]
    assert select_pages(pages, [], 3) == [3, 5, 0]
    assert select_pages(pages, [], 10) == [3, 5, 0, 1, 2, 4]
    assert select_pages(_pages(["ok"]), [0], 3) == [0]
    with pytest.raises(RepairError):
        select_pages(pages, [9], 3)
    with pytest.raises(RepairError):
        select_pages(pages, [], 0)


# --------------------------------------------------------------------------- R8 gate


def test_gate_flags_no_evidence_abstentions_and_ocr_gaps_only() -> None:
    clean, gap = _pages(["ok", "ok"]), _pages(["ok", "empty"])
    assert gate_flags(initial_status="answered", abstained=False, pages=clean) is False
    assert gate_flags(initial_status="answered", abstained=True, pages=clean) is True
    assert gate_flags(initial_status="no_evidence", abstained=False, pages=clean) is True
    assert gate_flags(initial_status="answered", abstained=False, pages=gap) is True
    assert gate_flags(initial_status="execution_failed", abstained=False, pages=clean) is False


# --------------------------------------------------------------------------- R1 boundary


def test_prepare_run_and_export_never_open_evaluation_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path)
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(Path(file).resolve().as_posix())
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    project.prepare()
    project.run(ScriptedProvider())
    rvr.export(project.run_dir())
    assert opened, "the spy saw no file at all"
    forbidden = [p for p in opened if "evaluation_manifest" in p or "retrieval_base/gt" in p or "scores.jsonl" in p]
    assert forbidden == []


def test_a_forbidden_path_is_refused_before_it_is_read(tmp_path: Path) -> None:
    project = Project(tmp_path)
    with pytest.raises(RepairError, match="evaluation-only"):
        rvr._read_json(project.root / "results" / "pilots" / PILOT / "evaluation_manifest.json")


# --------------------------------------------------------------------------- R3 PDF hash


def test_a_pdf_whose_hash_differs_from_the_manifest_is_refused(tmp_path: Path) -> None:
    project = Project(tmp_path)
    with zipfile.ZipFile(project.archive, "w") as z:
        z.writestr("d/a.pdf", _pdf(5))
        z.writestr("d/b.pdf", _pdf(1))
    with pytest.raises(RepairError, match="hash differs"):
        project.prepare()


# --------------------------------------------------------------------------- R4 request shape


def test_requests_keep_the_initial_user_message_and_attach_the_policy_pages_in_order(tmp_path: Path) -> None:
    project = Project(tmp_path)
    records = {r["question_id"]: r for r in project.prepare()}
    assert records["q1"]["user"] == "USER MESSAGE q1"
    assert records["q1"]["selected_pages"] == [1, 0, 2]
    assert records["q2"]["selected_pages"] == [3, 2, 0]
    assert records["q3"]["selected_pages"] == [0]
    assert records["q3"]["user"] == ("Question:\nWhat is written?\n\nNo text evidence is available. The evidence is the page "
                                     "image(s) below.\n\n" + TEMPLATE_V2.closing_line)
    assert all(r["system"] == TEMPLATE_V2.system + IMAGE_NOTE for r in records.values())

    provider = ScriptedProvider()
    project.run(provider)
    sent = {r.request_id: r for r in provider.requests}
    content = sent["q1"].messages[1]["content"]
    assert content[0] == {"type": "text", "text": "USER MESSAGE q1"}
    assert [c["text"] for c in content[1::2]] == ["Image of page 2:", "Image of page 1:", "Image of page 3:"]
    assert all(c["type"] == "image_url" and c["image_url"]["detail"] == "high" for c in content[2::2])
    assert sent["q1"].messages[0] == {"role": "system", "content": TEMPLATE_V2.system + IMAGE_NOTE}


# --------------------------------------------------------------------------- R5 unknown outcomes


def test_without_a_rule_an_unknown_outcome_is_never_resent(tmp_path: Path) -> None:
    project = Project(tmp_path)
    project.prepare()
    provider = ScriptedProvider({"q1": [lost()]})
    project.run(provider)
    project.run(provider)
    assert provider.sent() == ["q1", "q2", "q3"]
    rows = {r["question_id"]: r for r in rvr.export(project.run_dir())}
    assert rows["q1"]["status"] == "execution_failed" and rows["q1"]["blocked"] is True


def test_the_rule_resends_one_lost_connection_once_and_keeps_its_bound_reserved(tmp_path: Path) -> None:
    project = Project(tmp_path)
    records = {r["question_id"]: r for r in project.prepare(resend_rule="one-resend-connection-lost-v1")}
    provider = ScriptedProvider({"q1": [lost()], "q2": [lost(), lost()], "q3": [timeout()]})
    project.run(provider)
    project.run(provider)
    project.run(provider)
    assert provider.sent().count("q1") == 2
    assert provider.sent().count("q2") == 2
    assert provider.sent().count("q3") == 1
    state = rvr.ledger(events(project.run_dir()), "one-resend-connection-lost-v1")
    assert state["saved"] == {"q1"}
    assert state["blocked"] == {"q2", "q3"}
    expected = records["q1"]["cost_upper_bound"] + 2 * records["q2"]["cost_upper_bound"] + records["q3"]["cost_upper_bound"]
    assert state["reserved"] == pytest.approx(expected)
    resends = [e for e in events(project.run_dir()) if e["event"] == "dispatch" and e["resend"]]
    assert sorted(e["question_id"] for e in resends) == ["q1", "q2"]


# --------------------------------------------------------------------------- R6 cap


def test_the_cap_stops_before_a_request_that_could_cross_it(tmp_path: Path) -> None:
    project = Project(tmp_path)
    records = project.prepare()
    provider = ScriptedProvider({"q1": [lost()]})
    cap = records[0]["cost_upper_bound"] + records[1]["cost_upper_bound"] * 0.5
    result = project.run(provider, cap=cap)
    assert provider.sent() == ["q1"]
    assert result["end_reason"] == "cap_reached"


# --------------------------------------------------------------------------- R7 identity and baseline


def test_a_changed_run_cannot_reuse_the_run_directory(tmp_path: Path) -> None:
    project = Project(tmp_path)
    project.prepare()
    with pytest.raises(RepairError, match="different identity"):
        project.prepare(budget=2)


def test_an_invalid_baseline_is_refused(tmp_path: Path) -> None:
    project = Project(tmp_path, valid_baseline=False)
    with pytest.raises(RepairError, match="valid_baseline"):
        project.prepare()


def test_a_baseline_with_another_system_message_is_refused(tmp_path: Path) -> None:
    project = Project(tmp_path, template=TEMPLATE_V1)
    rows = [json.loads(line) for line in (project.baseline / "requests.jsonl").read_text().splitlines()]
    rows[0]["system"] = TEMPLATE_V2.system
    (project.baseline / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(RepairError, match="system message"):
        project.prepare()


def test_a_baseline_of_another_pilot_manifest_is_refused(tmp_path: Path) -> None:
    project = Project(tmp_path)
    (project.baseline / "run_config.json").write_text(json.dumps({"identity": {"runtime_manifest": {"sha256": "0" * 64}}}))
    with pytest.raises(RepairError, match="does not record the runtime manifest"):
        project.prepare()


def test_subset_identity_validation_and_full_run_format(tmp_path: Path) -> None:
    project = Project(tmp_path)
    full = project.prepare()
    full_config = json.loads((project.run_dir() / "run_config.json").read_text())
    subset = tmp_path / "questions.txt"
    subset.write_text("q3\n\nq1\nq1\n")
    rows = project.prepare(name="subset", questions_file=subset)
    assert [r["question_id"] for r in rows] == ["q1", "q3"]
    config = json.loads((project.run_dir("subset") / "run_config.json").read_text())
    assert config["questions_file_sha256"] == hashlib.sha256(subset.read_bytes()).hexdigest()
    assert config["question_count"] == 2
    assert "questions_file_sha256" not in full_config and "question_count" not in full_config
    assert rows == [full[0], full[2]] and config["requests_sha256"] != full_config["requests_sha256"]
    with pytest.raises(RepairError, match="different identity"):
        project.prepare(questions_file=subset)
    subset.write_text("q1\nunknown\n")
    with pytest.raises(RepairError, match="not in"):
        project.prepare(name="bad", questions_file=subset)
    subset.write_text(" \n\n")
    with pytest.raises(RepairError, match="empty"):
        project.prepare(name="bad", questions_file=subset)


def test_subset_run_export_and_score(tmp_path: Path) -> None:
    project = Project(tmp_path)
    subset = tmp_path / "questions.txt"
    subset.write_text("q3\nq1\n")
    project.prepare(questions_file=subset)
    provider = ScriptedProvider()
    project.run(provider)
    assert provider.sent() == ["q1", "q3"]
    assert [r["question_id"] for r in rvr.export(project.run_dir())] == ["q1", "q3"]
    summary = rvr.score(project.root, project.run_dir())
    assert summary["questions"] == 2
    assert summary["em"] == pytest.approx(0.5)
    assert summary["f1"] == pytest.approx(5 / 6)
    scores = [json.loads(line) for line in (project.run_dir() / "scores.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in scores] == ["q1", "q3"]
    assert [(r["em"], r["f1"]) for r in scores] == [(1, 1.0), (0, pytest.approx(2 / 3))]
