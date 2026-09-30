"""Tests for the answer-model run driver (``faar.live_runner`` and ``scripts/experiments/run_pilot_live.py``).

Every test uses the fake provider, a small project in a temporary directory and no credentials.
Nothing here sends a request. Live mode appears only in tests that prove it refuses to start.

Ways the driver could fail, written before the code. Each test below names the item it covers.

Resume and duplicates
  D1. A question with a saved response is sent again on resume.
  D2. A resume with changed prompt, evidence, model settings, retry parameters, prices or code dispatches anyway.
  D3. A crash loses completed work, or a resume repeats it.
  D4. Two invocations own the run at once.
Crash windows (contract rule 3)
  W1. Crash before ``dispatch_started``: the question is treated as unknown, or is lost.
  W2. Crash after ``dispatch_started``: the request is resent without reconciliation, or the whole run blocks.
  W3. Response received but not saved: the request is resent. A response file written before the event is discarded.
  W4. Saved but not exported: the export is missing, or ``export`` needs the provider.
Unknown outcomes
  U1. An unknown outcome is retried automatically, or blocks other questions.
  U2. ``reconcile`` accepts an attempt that is not unknown, drops the earlier attempt or lets attempts pass the limit.
Retry
  R1. A retry loop is unbounded, or an attempt is not preserved.
  R2. A stop_run error keeps the run going, or a non-retryable error stops it.
Budget
  B1. A dispatch happens when the ceiling would be exceeded.
  B2. Raising the ceiling erases or rewrites earlier events, or needs no note.
  B3. Unserved questions vanish from the export or count as abstentions.
Records
  E1. The event log accepts a corrupt line, a torn final line breaks later appends, or a tampered response is trusted.
  E2. An export drops a question, holds a status the scorer rejects, or changes on repeat.
Leakage and offline guarantees
  L1. Generation or preparation opens an evaluation file.
  O1. ``--help``, ``dry-run``, ``status`` or ``export`` builds a provider or a client.
  O2. Live mode starts without every requirement, or reads a credential before the checks pass.
  O3. A fake run touches the network or a credential.
Directories and CLI
  P1. A run or a dry run lands in ``results/pilots/``.
  C1. Exit codes share meaning.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import run_pilot_live as cli

from faar import live_runner as lr
from faar import pilot_runner as pr
from faar.answer_providers import FakeProvider, FakeStep, SimulatedCrash
from faar.live_contract import MODE_FAKE, ProviderError
from faar.pilot_runner import RunnerRefusal

PILOT_ID = "fix_v1"
MINERU_ROOT = "OHR-Bench/data/retrieval_base/MinerU"
DOC_A, DOC_B, DOC_C, DOC_BLANK = "law/docA", "law/doc,B with space", "law/docC", "law/blank"
WARRANTY_A = "Alpha Corp warranty period is twelve months from the delivery date. The buyer pays 500 dollars."
WARRANTY_B = "Beta Corp warranty period is twenty four months from the delivery date. The buyer pays 900 dollars."
LEASE_C = "Gamma Corp lease runs five years. The rent is 2000 dollars per month and the tenant pays for repairs."
WARRANTY_QUESTION = "What is the warranty period from the delivery date?"
TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
SCRIPTS_DIR = TESTS_DIR.parent / "scripts" / "experiments"
QUESTIONS = ["q1", "q2", "q3", "q4", "q5", "q6"]
SENT = ["q1", "q2", "q3", "q4", "q6"]  # q5 sits on an empty document and is never sent


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Project:
    root: Path
    docs: dict[str, list[str | None]]
    questions: list[dict[str, str]]

    @property
    def pilot_dir(self) -> Path:
        return self.root / "results" / "pilots" / PILOT_ID

    @property
    def runtime_manifest(self) -> Path:
        return self.pilot_dir / "runtime_manifest.json"

    @property
    def evaluation_manifest(self) -> Path:
        return self.pilot_dir / "evaluation_manifest.json"

    def mineru_path(self, doc_id: str) -> Path:
        return self.root / MINERU_ROOT / f"{doc_id}.json"

    def run_dir(self, name: str) -> Path:
        return self.root / "results" / "engineering" / name

    def write(self) -> Project:
        documents = []
        for doc_id, pages in self.docs.items():
            entries = [{"page_idx": i, "text": text} for i, text in enumerate(pages) if text is not None]
            path = self.mineru_path(doc_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(entries).encode("utf-8")
            path.write_bytes(raw)
            rows = []
            for i, text in enumerate(pages):
                status = "missing" if text is None else ("ok" if text.strip() else "empty")
                digest = None if text is None else sha256_bytes(text.encode("utf-8"))
                rows.append({"page_idx": i, "pdf_page_number": i + 1, "ocr_status": status, "ocr_text_sha256": digest})
            documents.append(
                {
                    "doc_id": doc_id,
                    "noisy_text": {"path": f"{MINERU_ROOT}/{doc_id}.json", "sha256": sha256_bytes(raw), "status": "present"},
                    "pages": rows,
                }
            )
        runtime = {
            "kind": "runtime",
            "pilot_id": PILOT_ID,
            "sources": {"noisy_text": {"root": MINERU_ROOT}},
            "documents": documents,
            "questions": self.questions,
        }
        self.pilot_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_manifest.write_text(json.dumps(runtime, indent=1), encoding="utf-8")
        self.write_evaluation()
        (self.pilot_dir / "selection_record.json").write_text("{}", encoding="utf-8")
        (self.root / "OHR-Bench" / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "OHR-Bench" / "data" / "qas_v2.json").write_text("[]", encoding="utf-8")
        return self

    def write_evaluation(self, *, raw: str | None = None) -> None:
        questions = {q["question_id"]: {"answers": "twelve months", "doc_id": q["doc_id"]} for q in self.questions}
        payload = raw if raw is not None else json.dumps({"kind": "evaluation", "pilot_id": PILOT_ID, "questions": questions})
        self.evaluation_manifest.write_text(payload, encoding="utf-8")


def default_project(root: Path) -> Project:
    return Project(
        root,
        {
            DOC_A: [WARRANTY_A, "Alpha Corp termination clause: either party may end this agreement."],
            DOC_B: [WARRANTY_B],
            DOC_C: [LEASE_C],
            DOC_BLANK: ["", None],
        },
        [
            {"question_id": "q1", "doc_id": DOC_A, "question": WARRANTY_QUESTION},
            {"question_id": "q2", "doc_id": DOC_B, "question": WARRANTY_QUESTION},
            {"question_id": "q3", "doc_id": DOC_A, "question": "How much does the buyer pay?"},
            {"question_id": "q4", "doc_id": DOC_C, "question": "How many years does the lease run?"},
            {"question_id": "q5", "doc_id": DOC_BLANK, "question": "What is the warranty period?"},
            {"question_id": "q6", "doc_id": DOC_B, "question": "How much does the buyer pay?"},
        ],
    ).write()


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return default_project(tmp_path / "project")


class Rig:
    """A scripted fake provider that survives fresh invocations, plus the record of every send."""

    def __init__(self, steps: Mapping[str, list[FakeStep]] | None = None, default: FakeStep | None = None) -> None:
        self.steps = dict(steps or {})
        self.default = default or FakeStep("answer", text="twelve months")
        self.providers: list[FakeProvider] = []
        self.factory_calls = 0

    @property
    def descriptor(self) -> dict[str, Any]:
        payload = {"steps": {k: [vars(s) for s in v] for k, v in sorted(self.steps.items())}, "default": vars(self.default)}
        return {"adapter": "faar.answer_providers.FakeProvider", "fake_script_sha256": sha256_bytes(json.dumps(payload).encode())}

    def factory(self, context: lr.ProviderContext) -> FakeProvider:
        self.factory_calls += 1
        provider = lr.build_fake_provider(self.steps, self.default, lr.FAKE_CONFIG, context)
        self.providers.append(provider)
        return provider

    def sent(self, run_dir: Path) -> list[str]:
        """Question ids of every send, in order, across all invocations of this rig."""
        by_request = {r["request_id"]: r["question_id"] for r in requests_of(run_dir)}
        return [by_request[call.request_id] for provider in self.providers for call in provider.calls]

    def sent_since(self, run_dir: Path, first_provider: int) -> list[str]:
        by_request = {r["request_id"]: r["question_id"] for r in requests_of(run_dir)}
        return [by_request[c.request_id] for p in self.providers[first_provider:] for c in p.calls]


def requests_of(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()]


def events_of(run_dir: Path) -> list[dict[str, Any]]:
    return list(lr.iter_committed_events(run_dir))


def predictions_of(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]


def summary_of(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))


def options(project: Project, name: str = "run-a", **kwargs: Any) -> lr.RunOptions:
    kwargs.setdefault("safety_ceiling", 100.0)
    return lr.RunOptions(
        project_root=project.root, run_dir=project.run_dir(name), mode=MODE_FAKE, pilot_id=PILOT_ID, **kwargs
    )


def go(
    project: Project,
    rig: Rig,
    name: str = "run-a",
    *,
    crash_hook: Any = None,
    sleep: Any = None,
    services: lr.Services | None = None,
    **kwargs: Any,
) -> lr.LiveResult:
    return lr.execute_run(
        options(project, name, **kwargs),
        services=services,
        provider_factory=rig.factory,
        descriptor=rig.descriptor,
        sleep=sleep or (lambda seconds: None),
        crash_hook=crash_hook,
        environ={},
    )


def crash_at(point: str, question_id: str | None = None, attempt: int | None = None) -> Any:
    def hook(name: str, info: Mapping[str, Any]) -> None:
        if name != point:
            return
        if question_id is not None and info.get("question_id") != question_id:
            return
        if attempt is not None and info.get("attempt") != attempt:
            return
        raise SimulatedCrash(f"crash at {point}")

    return hook


def by_question(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {p["question_id"]: p for p in predictions_of(run_dir)}


def names(run_dir: Path, question_id: str | None = None) -> list[str]:
    ids = {r["question_id"]: r["request_id"] for r in requests_of(run_dir)}
    return [
        e["event"] for e in events_of(run_dir) if question_id is None or e.get("request_id") == ids[question_id]
    ]


def bounds(project: Project) -> dict[str, float]:
    """Per-attempt cost upper bound of each question that is sent, from a dry run."""
    out = project.root / "dry-for-bounds"
    lr.dry_run(project_root=project.root, out_dir=out, pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    prepared = [json.loads(line) for line in (out / "prepared_requests.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["question_id"]: r["cost_upper_bound"] for r in prepared if r["action"] == "send"}


def file_snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# Public retrieval helper in pilot_runner
# ---------------------------------------------------------------------------


def test_retrieval_helper_returns_hits_with_text_in_manifest_order(project: Project) -> None:
    """The helper gives per-question hits with chunk text, in manifest order, from the same index generate_run uses."""
    run = pr.retrieve_runtime_questions(project_root=project.root, pilot_id=PILOT_ID)
    assert [o.question.question_id for o in run.questions] == QUESTIONS
    by_id = {o.question.question_id: o for o in run.questions}
    assert by_id["q1"].hits and all(h.chunk.doc_name == DOC_A for h in by_id["q1"].hits)
    assert any("twelve months" in h.chunk.text for h in by_id["q1"].hits)
    assert by_id["q2"].hits and all(h.chunk.doc_name == DOC_B for h in by_id["q2"].hits)
    assert by_id["q5"].hits == () and by_id["q5"].no_evidence_reason == pr.NO_TEXT_CHUNKS
    assert by_id["q1"].query_retrieval_tokens > 0
    assert run.runtime_manifest_sha256 == sha256_bytes(project.runtime_manifest.read_bytes())
    assert set(run.document_noisy_text_sha256) == set(project.docs)
    generated = pr.generate_run(
        project_root=project.root, run_dir=project.run_dir("offline"), pilot_id=PILOT_ID
    )
    offline = {r["question_id"]: r for r in map(json.loads, (project.run_dir("offline") / "predictions.jsonl").read_text().splitlines())}
    assert generated.exit_code == pr.EXIT_OK
    for qid, outcome in by_id.items():
        assert [h.chunk.chunk_id for h in outcome.hits] == [e["chunk_id"] for e in offline[qid]["evidence"]]


def test_retrieval_helper_reads_only_the_runtime_manifest_and_mineru_files(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """L1: the helper opens nothing else under the project."""
    opened = spy_on_opens(monkeypatch)
    pr.retrieve_runtime_questions(project_root=project.root, pilot_id=PILOT_ID)
    allowed = {str(project.runtime_manifest.resolve())} | {str(project.mineru_path(d).resolve()) for d in project.docs}
    assert reads_under(opened, project.root) == allowed


def spy_on_opens(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    opened: list[tuple[str, str]] = []
    real_open = io.open

    def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append((str(Path(file).resolve()), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    return opened


def reads_under(opened: list[tuple[str, str]], root: Path) -> set[str]:
    base = str(root.resolve())
    return {path for path, mode in opened if path.startswith(base) and not any(flag in mode for flag in "wax+")}


# ---------------------------------------------------------------------------
# Preparation and dry run
# ---------------------------------------------------------------------------


def test_dry_run_prepares_every_question_and_writes_the_three_files(project: Project) -> None:
    """The dry run holds send/skip, exact messages, hashes, bounds and full evidence text for all questions."""
    out = project.root / "dry"
    result = lr.dry_run(project_root=project.root, out_dir=out, pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    assert result.exit_code == 0
    assert sorted(p.name for p in out.iterdir()) == ["dry_run_summary.json", "prepared_requests.jsonl", "prompt_preview.md"]
    prepared = [json.loads(line) for line in (out / "prepared_requests.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in prepared] == QUESTIONS
    sends = {r["question_id"]: r for r in prepared if r["action"] == "send"}
    assert sorted(sends) == sorted(SENT)
    skip = next(r for r in prepared if r["question_id"] == "q5")
    assert (skip["action"], skip["skip_reason"], skip["system"], skip["user"]) == ("skip", "no_text_chunks", None, None)
    assert skip["evidence"] == [] and skip["evidence_sha256"] is None and skip["cost_upper_bound"] is None
    record = sends["q1"]
    assert record["template_id"] == "faar-answer-draft-v1"
    assert record["max_output_tokens"] == lr.FAKE_CONFIG.max_output_tokens
    assert record["input_token_upper_bound"] > len((record["system"] + record["user"]).encode("utf-8"))
    assert record["cost_upper_bound"] > 0 and record["cost_simulated"] is True
    assert record["token_estimate"]["estimate"] > 0
    assert record["prompt_sha256"] and record["evidence_sha256"]
    for item in record["evidence"]:
        assert item["doc_id"] == DOC_A
        assert item["pdf_page_number"] == item["page_idx"] + 1
        assert item["chars"] == len(item["text"]) and item["text_sha256"] == sha256_bytes(item["text"].encode("utf-8"))
        assert item["text"] in record["user"]
    summary = json.loads((out / "dry_run_summary.json").read_text())
    assert summary["provider_calls"] == 0
    assert summary["counts"] == {"questions": 6, "send": 5, "skip": 1, "skip_reasons": {"no_text_chunks": 1}}
    assert summary["hashes"]["prepared_requests_sha256"] == sha256_bytes((out / "prepared_requests.jsonl").read_bytes())
    assert summary["hashes"]["prompt_preview_sha256"] == sha256_bytes((out / "prompt_preview.md").read_bytes())
    assert "twelve months" in (out / "prompt_preview.md").read_text()


def test_request_id_is_the_documented_hash_and_requests_jsonl_drops_chunk_text(project: Project) -> None:
    """The request id is sha256(identity + NUL + question id + NUL + prompt hash)[:32]; the stored form keeps text only inside ``user``."""
    rig = Rig()
    go(project, rig)
    run_dir = project.run_dir("run-a")
    config = json.loads((run_dir / "run_config.json").read_text())
    for record in requests_of(run_dir):
        expected = hashlib.sha256(
            f"{config['identity_sha256']}\0{record['question_id']}\0{record['prompt_sha256'] or ''}".encode()
        ).hexdigest()[:32]
        assert record["request_id"] == expected
        assert all("text" not in item for item in record["evidence"])
    q1 = next(r for r in requests_of(run_dir) if r["question_id"] == "q1")
    assert "twelve months" in q1["user"]


def test_dry_run_makes_no_provider_call_and_builds_no_client(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """O1: dry-run needs no provider config or credentials and constructs nothing."""
    tripwire_clients(monkeypatch)
    result = lr.dry_run(project_root=project.root, out_dir=project.root / "dry", pilot_id=PILOT_ID)
    assert result.exit_code == 0
    assert result.summary["provider_config"]["source"].startswith("built-in provisional")
    assert result.summary["provider_calls"] == 0


def tripwire_clients(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make building any provider or SDK client fail loudly. Returns a list that stays empty when nothing was built."""
    built: list[str] = []

    def boom(name: str) -> Any:
        def inner(*args: Any, **kwargs: Any) -> Any:
            built.append(name)
            raise AssertionError(f"{name} must not be called here")

        return inner

    import openai

    from faar import answer_providers

    monkeypatch.setattr(openai, "OpenAI", boom("openai.OpenAI"))
    monkeypatch.setattr(lr, "build_live_provider", boom("build_live_provider"))
    monkeypatch.setattr(lr, "build_fake_provider", boom("build_fake_provider"))
    monkeypatch.setattr(answer_providers, "OpenAIChatProvider", boom("OpenAIChatProvider"))
    monkeypatch.setattr(answer_providers, "FakeProvider", boom("FakeProvider"))
    return built


@pytest.mark.parametrize("sub", ["results/pilots/x", "results/Pilots/x", "results/pilots", "config/x", "OHR-Bench/data/out", "logs/x"])
def test_dry_run_refuses_frozen_locations(project: Project, sub: str) -> None:
    """P1: no dry-run output inside results/pilots (any letter case) or another frozen directory."""
    with pytest.raises(RunnerRefusal, match="refusing"):
        lr.dry_run(project_root=project.root, out_dir=project.root / sub, pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    assert not (project.root / sub / "prepared_requests.jsonl").exists()


def test_dry_run_outside_the_project_and_under_results_engineering_are_allowed(project: Project, tmp_path: Path) -> None:
    lr.dry_run(project_root=project.root, out_dir=tmp_path / "elsewhere", pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    lr.dry_run(project_root=project.root, out_dir=project.run_dir("dry"), pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)


def test_dry_run_repeat_is_verified_identical_and_a_changed_dry_run_is_refused(project: Project) -> None:
    out = project.root / "dry"
    first = lr.dry_run(project_root=project.root, out_dir=out, pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    before = file_snapshot(out)
    second = lr.dry_run(project_root=project.root, out_dir=out, pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    assert "verified identical" in second.message and file_snapshot(out) == before
    assert first.summary == second.summary
    project.mineru_path(DOC_C).write_text(json.dumps([{"page_idx": 0, "text": LEASE_C + " Extra."}]))
    with pytest.raises(RunnerRefusal):
        lr.dry_run(project_root=project.root, out_dir=out, pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    assert file_snapshot(out) == before


def test_preparation_and_run_read_only_the_runtime_manifest_and_mineru_files(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """L1: file access during dry-run, run, status and export stays inside the runtime inputs and the run directory."""
    opened = spy_on_opens(monkeypatch)
    rig = Rig()
    lr.dry_run(project_root=project.root, out_dir=project.root / "dry", pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    go(project, rig)
    lr.run_status(project.run_dir("run-a"))
    lr.export_run(project.run_dir("run-a"))
    allowed = {str(project.runtime_manifest.resolve())} | {str(project.mineru_path(d).resolve()) for d in project.docs}
    outside = {
        p
        for p in reads_under(opened, project.root)
        if p not in allowed
        and not p.startswith(str((project.root / "results" / "engineering").resolve()))
        and not p.startswith(str((project.root / "dry").resolve()))
    }
    assert outside == set()


def test_poisoned_evaluation_files_change_nothing_in_the_prepared_requests(tmp_path: Path) -> None:
    """L1: requests.jsonl is byte-identical when every evaluation-only file is replaced with poison."""
    clean, poisoned = default_project(tmp_path / "clean"), default_project(tmp_path / "poisoned")
    poisoned.write_evaluation(raw="{ not json")
    (poisoned.pilot_dir / "selection_record.json").write_text("\x00 poison")
    (poisoned.root / "OHR-Bench" / "data" / "qas_v2.json").write_text("poison")
    go(clean, Rig())
    go(poisoned, Rig())
    assert (clean.run_dir("run-a") / "requests.jsonl").read_bytes() == (poisoned.run_dir("run-a") / "requests.jsonl").read_bytes()


def test_evidence_stays_inside_the_questions_document(project: Project) -> None:
    """R1: every evidence item of every prepared request belongs to the question's own document."""
    go(project, Rig())
    docs = {q["question_id"]: q["doc_id"] for q in project.questions}
    for record in requests_of(project.run_dir("run-a")):
        assert {e["doc_id"] for e in record["evidence"]} <= {docs[record["question_id"]]}
        assert all(e["chunk_id"].startswith(record["doc_id"] + "-p") for e in record["evidence"])


def test_a_foreign_chunk_in_the_retrieval_result_is_refused_before_anything_is_written(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1: if retrieval ever returned another document's chunk, preparation refuses instead of sending it."""
    real = pr.retrieve_runtime_questions

    def poisoned(**kwargs: Any) -> pr.RetrievalRun:
        run = real(**kwargs)
        other = next(h for o in run.questions if o.question.doc_id == DOC_B for h in o.hits)
        target = next(i for i, o in enumerate(run.questions) if o.question.question_id == "q1")
        outcomes = list(run.questions)
        outcomes[target] = pr.QuestionRetrieval(
            outcomes[target].question, (other,), None, 3, outcomes[target].ocr_condition
        )
        return pr.RetrievalRun(**{**vars(run), "questions": tuple(outcomes)})

    monkeypatch.setattr(lr, "retrieve_runtime_questions", poisoned)
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="belongs to"):
        go(project, rig)
    assert rig.factory_calls == 0
    assert not project.run_dir("run-a").exists()


def test_a_question_whose_prompt_cannot_be_built_is_recorded_and_the_run_continues(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ValueError from the prompt builder is a per-question failure, not a crash (lead amendment)."""
    real = lr.Services.default()

    def picky(question: str, evidence: Any) -> Any:
        if evidence[0].doc_id == DOC_C:
            raise ValueError("chunk id is not header safe")
        return real.build_prompt(question, evidence)

    services = lr.Services(**{**vars(real), "build_prompt": picky})
    rig = Rig()
    result = go(project, rig, services=services)
    records = by_question(project.run_dir("run-a"))
    assert records["q4"]["status"] == "execution_failed"
    assert records["q4"]["failure"]["stage"] == "prepare" and "header safe" in records["q4"]["failure"]["message"]
    assert records["q4"]["unserved"] is False and records["q4"]["abstained"] is False
    assert rig.sent(project.run_dir("run-a")) == ["q1", "q2", "q3", "q6"]
    assert result.exit_code == lr.EXIT_EXECUTION_FAILED


def test_a_prompt_over_the_input_limit_is_not_sent_and_is_execution_failed(project: Project) -> None:
    """A prompt whose input bound passes max_input_tokens is skipped with prompt_over_limit, never truncated."""
    small = lr.ProviderConfig(**{**vars(lr.FAKE_CONFIG), "max_input_tokens": 400})
    rig = Rig()
    result = go(project, rig, config=small)
    records = by_question(project.run_dir("run-a"))
    over = [q for q, r in records.items() if r["skip_reason"] == "prompt_over_limit"]
    assert over, "the fixture prompts should pass a 400-token bound"
    for qid in over:
        assert records[qid]["status"] == "execution_failed" and records[qid]["failure"]["type"] == "prompt_over_limit"
        assert qid not in rig.sent(project.run_dir("run-a"))
    assert result.summary["counts"]["prompt_over_limit"] == len(over)


# ---------------------------------------------------------------------------
# A full fake run
# ---------------------------------------------------------------------------


def test_a_fake_run_answers_every_question_and_exports_them_in_manifest_order(project: Project) -> None:
    """E2: one record per question, only scorer statuses, complete state, exit 0, simulated cost."""
    rig = Rig()
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert result.exit_code == 0
    records = predictions_of(run_dir)
    assert [r["question_id"] for r in records] == QUESTIONS
    assert {r["status"] for r in records} <= {"answered", "no_evidence", "execution_failed"}
    by_id = by_question(run_dir)
    assert by_id["q5"]["status"] == "no_evidence" and by_id["q5"]["abstained"] is True and by_id["q5"]["answer"] == ""
    assert by_id["q5"]["no_evidence_reason"] == "no_text_chunks" and by_id["q5"]["attempt_ids"] == []
    assert all(by_id[q]["status"] == "answered" and by_id[q]["answer"] == "twelve months" for q in SENT)
    assert all(len(by_id[q]["attempt_ids"]) == 1 and by_id[q]["output_status"] == "ok" for q in SENT)
    summary = summary_of(run_dir)
    assert summary["run_state"] == "complete" and summary["valid_baseline"] is False
    assert summary["kind"] == "engineering_check" and summary["cost"]["simulated"] is True
    assert summary["counts"]["answered"] == 5 and summary["counts"]["no_evidence"] == 1
    assert summary["predictions_sha256"] == sha256_bytes((run_dir / "predictions.jsonl").read_bytes())
    assert (run_dir / "run.lock").exists()
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["mode"] == "fake" and config["kind"] == "engineering_check"
    assert config["identity_sha256"] == lr.identity_sha256(config["identity"])
    assert set(config["provenance"]) >= {"commit", "dirty", "dirty_paths", "environment", "command"}
    saved = [e for e in events_of(run_dir) if e["event"] == "response_saved"]
    assert len(saved) == 5
    for event in saved:
        assert sha256_bytes((run_dir / event["response_file"]).read_bytes()) == event["response_sha256"]
        assert event["response_file"] == f"responses/{event['attempt_id']}.json"


def test_events_are_numbered_lines_with_the_documented_fields(project: Project) -> None:
    go(project, Rig())
    events = events_of(project.run_dir("run-a"))
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["event"] == "invocation_started" and events[-1]["event"] == "invocation_ended"
    assert events[0]["safety_ceiling"] == {"amount": 100.0, "currency": "USD", "simulated": True}
    assert events[0]["ceiling_change"] is None and events[0]["mode"] == "fake"
    dispatch = next(e for e in events if e["event"] == "dispatch_started")
    assert set(dispatch) >= {"question_id", "request_id", "attempt", "attempt_id", "cost_upper_bound", "ledger_before"}
    assert dispatch["attempt_id"] == f"{dispatch['request_id']}-a1"
    assert {"measured", "reserved"} <= set(dispatch["ledger_before"])
    assert all(len(e["invocation_id"]) == 32 for e in events)


def test_the_identity_holds_no_time_and_no_ceiling_and_repeats_across_run_directories(project: Project) -> None:
    """The identity and every request id are content only: two directories with the same inputs agree."""
    go(project, Rig(), "run-a", safety_ceiling=100.0)
    go(project, Rig(), "run-b", safety_ceiling=7.5)
    a = json.loads((project.run_dir("run-a") / "run_config.json").read_text())
    b = json.loads((project.run_dir("run-b") / "run_config.json").read_text())
    assert a["identity_sha256"] == b["identity_sha256"]
    assert (project.run_dir("run-a") / "requests.jsonl").read_bytes() == (project.run_dir("run-b") / "requests.jsonl").read_bytes()
    assert "safety_ceiling" not in json.dumps(a["identity"])
    assert set(a["identity"]) >= {
        "pilot_id",
        "runtime_manifest_sha256",
        "document_noisy_text_sha256",
        "retrieval",
        "prompt",
        "provider",
        "prices",
        "retry_policy",
        "scientific_budget",
        "code",
        "contract_version",
    }
    assert a["identity"]["scientific_budget"] is None
    assert a["identity"]["prices"]["simulated"] is True
    assert a["identity"]["code"]["measurement_code_digest"]


# ---------------------------------------------------------------------------
# Resume, duplicates and identity
# ---------------------------------------------------------------------------


def test_a_successful_question_is_never_sent_again_on_resume(project: Project) -> None:
    """D1: a second invocation of a finished run sends nothing and builds no provider."""
    rig = Rig()
    go(project, rig)
    run_dir = project.run_dir("run-a")
    assert sorted(rig.sent(run_dir)) == sorted(SENT)
    before = predictions_of(run_dir)
    result = go(project, rig)
    assert rig.factory_calls == 1, "no provider is built when nothing is pending"
    assert sorted(rig.sent(run_dir)) == sorted(SENT)
    assert result.exit_code == 0 and predictions_of(run_dir) == before
    assert [e["event"] for e in events_of(run_dir)][-2:] == ["invocation_started", "invocation_ended"]


@pytest.mark.parametrize(
    "change",
    ["template", "evidence", "model", "params", "retry", "prices", "code", "provider_adapter"],
)
def test_an_incompatible_resume_is_refused_before_any_dispatch(project: Project, change: str, tmp_path: Path) -> None:
    """D2: a change to the prompt template, evidence, model config, retry parameters, prices, code or adapter refuses the resume."""
    # The first invocation dies before q3, so the run has pending work that a compatible resume would send.
    rig = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_BEFORE_DISPATCH, "q3"))
    run_dir = project.run_dir("run-a")
    sent_before = rig.sent(run_dir)
    before = file_snapshot(run_dir)
    services = None
    kwargs: dict[str, Any] = {}
    descriptor_change: dict[str, Any] = {}
    real = lr.Services.default()
    if change == "template":
        services = lr.Services(**{**vars(real), "template_sha256": "0" * 64, "build_prompt": lambda q, e: _retemplate(real, q, e)})
    elif change == "evidence":
        project.mineru_path(DOC_A).write_text(json.dumps([{"page_idx": 0, "text": WARRANTY_A}, {"page_idx": 1, "text": "changed"}]))
    elif change == "model":
        kwargs["config"] = lr.ProviderConfig(**{**vars(lr.FAKE_CONFIG), "model": "fake-answer-model-2"})
    elif change == "params":
        kwargs["config"] = lr.ProviderConfig(**{**vars(lr.FAKE_CONFIG), "max_output_tokens": 64})
    elif change == "retry":
        from faar.retry_policy import RetryPolicy

        services = lr.Services(**{**vars(real), "retry_policy": RetryPolicy(max_attempts=5)})
    elif change == "prices":
        prices = lr.PriceTable(**{**lr.FAKE_CONFIG.prices.as_dict(), "input_per_million": 3.0})
        kwargs["config"] = lr.ProviderConfig(**{**vars(lr.FAKE_CONFIG), "prices": prices})
    elif change == "code":
        script = tmp_path / "cli.py"
        script.write_text("print('a')")
        with pytest.raises(SimulatedCrash):
            go(project, Rig(), "run-c", cli_script=script, crash_hook=crash_at(lr.CRASH_BEFORE_DISPATCH, "q3"))
        script.write_text("print('b')")
        rig2 = Rig()
        with pytest.raises(RunnerRefusal, match="different identity.*code.cli_script_sha256"):
            go(project, rig2, "run-c", cli_script=script)
        assert rig2.factory_calls == 0
        return
    else:
        descriptor_change = {"adapter": "something else"}
    rig2 = Rig()
    rig2_descriptor = {**rig2.descriptor, **descriptor_change}
    with pytest.raises(RunnerRefusal) as excinfo:
        lr.execute_run(
            options(project, **kwargs),
            services=services,
            provider_factory=rig2.factory,
            descriptor=rig2_descriptor,
            sleep=lambda s: None,
            environ={},
        )
    assert rig2.factory_calls == 0 and rig2.providers == [] and rig.sent(run_dir) == sent_before
    assert strip_lock(file_snapshot(run_dir)) == strip_lock(before)
    assert "Nothing was dispatched" in str(excinfo.value) or change == "evidence"


def _retemplate(real: lr.Services, question: str, evidence: Any) -> Any:
    payload = real.build_prompt(question, evidence)
    return type(payload)(**{**vars(payload), "template_sha256": "0" * 64})


def test_a_changed_prompt_text_with_the_same_template_id_is_refused(project: Project) -> None:
    """D2: prompts that regenerate differently (same template id and hash, different text) refuse the resume."""
    go(project, Rig())
    real = lr.Services.default()

    def altered(question: str, evidence: Any) -> Any:
        payload = real.build_prompt(question, evidence)
        return type(payload)(**{**vars(payload), "user": payload.user + "\nExtra line", "prompt_sha256": "f" * 64})

    services = lr.Services(**{**vars(real), "build_prompt": altered})
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="regenerate differently"):
        go(project, rig, services=services)
    assert rig.factory_calls == 0


def test_a_fake_script_change_is_a_different_run(project: Project) -> None:
    go(project, Rig())
    rig = Rig(steps={"q1": [FakeStep("abstain")]})
    with pytest.raises(RunnerRefusal, match="provider.adapter.fake_script_sha256"):
        go(project, rig)


def test_the_run_id_must_match_and_be_valid(project: Project) -> None:
    go(project, Rig())
    with pytest.raises(RunnerRefusal, match="run_id"):
        go(project, Rig(), run_id="other-id")
    with pytest.raises(RunnerRefusal, match="run_id"):
        go(project, Rig(), "Bad Name!")


# ---------------------------------------------------------------------------
# Crash windows (contract rule 3)
# ---------------------------------------------------------------------------


def crash_and_resume(project: Project, point: str, question_id: str = "q3", attempt: int = 1, rig: Rig | None = None) -> tuple[Path, Rig, lr.LiveResult]:
    """Crash the first invocation at ``point`` of ``question_id``, then resume in a fresh invocation."""
    rig = rig or Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(point, question_id, attempt))
    run_dir = project.run_dir("run-a")
    assert not (run_dir / "predictions.jsonl").exists(), "a crash writes no export"
    resumed = go(project, rig)
    return run_dir, rig, resumed


def test_window_1_crash_before_dispatch_started_leaves_the_question_pending(project: Project) -> None:
    """W1: nothing was sent or logged for the question. The resume sends it once, and only then."""
    rig = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_BEFORE_DISPATCH, "q3"))
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2"]
    assert names(run_dir, "q3") == []
    assert [e["event"] for e in events_of(run_dir)][-1] == "response_saved"
    status = lr.run_status(run_dir)
    assert status["requests"]["pending"] == 3 and status["requests"]["answered"] == 2 and status["run_state"] == "incomplete"
    resumed = go(project, rig)
    assert rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"], "each question is sent exactly once across both invocations"
    assert resumed.exit_code == 0 and summary_of(run_dir)["run_state"] == "complete"
    assert by_question(run_dir)["q3"]["attempt_ids"] == [f"{by_question(run_dir)['q3']['request_id']}-a1"]


def test_window_2_crash_after_dispatch_started_is_unknown_and_not_resent(project: Project) -> None:
    """W2: dispatch_started is on disk and nothing resolves it. The resume records outcome_unknown, does not resend, and serves the rest."""
    rig = Rig()
    run_dir, rig, resumed = crash_and_resume(project, lr.CRASH_AFTER_DISPATCH, rig=rig)
    assert rig.sent(run_dir) == ["q1", "q2", "q4", "q6"], "q3 was never sent again, q4 and q6 continued"
    events = [e for e in events_of(run_dir) if e.get("question_id") == "q3"]
    assert [e["event"] for e in events] == ["dispatch_started", "outcome_unknown"]
    unknown = events[1]
    assert unknown["kind"] == "orphaned_dispatch" and unknown["recovered"] is True
    assert unknown["recovered_from_invocation_id"] == events[0]["invocation_id"] != unknown["invocation_id"]
    assert resumed.exit_code == lr.EXIT_NEEDS_ATTENTION
    record = by_question(run_dir)["q3"]
    assert record["status"] == "execution_failed" and record["awaiting_reconciliation"] is True and record["abstained"] is False
    assert summary_of(run_dir)["run_state"] == "needs_reconciliation"
    assert summary_of(run_dir)["cost"]["reserved"] == pytest.approx(next(r["cost_upper_bound"] for r in requests_of(run_dir) if r["question_id"] == "q3"))
    # A third invocation still does not resend it.
    go(project, rig)
    assert rig.sent(run_dir) == ["q1", "q2", "q4", "q6"]
    # Reconcile, then a new attempt goes out and the run completes.
    attempt_id = record["attempt_ids"][0]
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=attempt_id, resolution="allow_new_attempt", note="dashboard shows no request")
    final = go(project, rig)
    assert rig.sent(run_dir)[-1] == "q3" and final.exit_code == 0
    done = by_question(run_dir)["q3"]
    assert done["status"] == "answered" and len(done["attempt_ids"]) == 2
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown", "reconciled", "dispatch_started", "response_saved"]


def test_window_3_response_received_but_not_saved_is_unknown_and_not_resent(project: Project) -> None:
    """W3: the provider answered and the process died before the response file existed. The answer is lost and never resent."""
    rig = Rig()
    run_dir, rig, resumed = crash_and_resume(project, lr.CRASH_AFTER_SEND, rig=rig)
    assert rig.sent(run_dir).count("q3") == 1, "q3 reached the provider once and is not resent"
    assert not list((run_dir / "responses").glob(f"{by_question(run_dir)['q3']['request_id']}*"))
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown"]
    assert resumed.exit_code == lr.EXIT_NEEDS_ATTENTION
    assert by_question(run_dir)["q3"]["awaiting_reconciliation"] is True
    assert by_question(run_dir)["q4"]["status"] == "answered"


def test_window_3_fake_step_crash_after_send_records_the_call_then_dies(project: Project) -> None:
    """W3: the fake's crash_after_send step raises SimulatedCrash after the call is recorded, and the driver does not swallow it."""
    rig = Rig(steps={"q3": [FakeStep("crash_after_send")]})
    # The script is keyed by question in the rig, so build it through the same path the CLI uses.
    with pytest.raises(SimulatedCrash):
        go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2", "q3"]
    assert names(run_dir, "q3") == ["dispatch_started"]
    go(project, rig)
    assert rig.sent(run_dir).count("q3") == 1
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown"]


def test_window_3_response_file_written_before_the_event_is_kept_on_resume(project: Project) -> None:
    """W3, file on disk: the resume appends response_saved from the file, resends nothing and loses no answer."""
    rig = Rig()
    run_dir, rig, resumed = crash_and_resume(project, lr.CRASH_AFTER_RESPONSE_FILE, rig=rig)
    assert rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"]
    events = [e for e in events_of(run_dir) if e.get("question_id") == "q3"]
    assert [e["event"] for e in events] == ["dispatch_started", "response_saved"]
    assert events[1]["recovered"] is True and events[1]["recovered_from_invocation_id"] == events[0]["invocation_id"]
    assert resumed.exit_code == 0
    record = by_question(run_dir)["q3"]
    assert record["status"] == "answered" and record["answer"] == "twelve months"
    assert sha256_bytes((run_dir / events[1]["response_file"]).read_bytes()) == events[1]["response_sha256"]
    assert summary_of(run_dir)["cost"]["measured"] > 0 and summary_of(run_dir)["cost"]["reserved"] == 0


def test_window_4_saved_but_not_exported_is_rebuilt_by_export_and_by_resume(project: Project) -> None:
    """W4: crash right after response_saved, and crash after the last event but before the export. Nothing is lost."""
    rig = Rig()
    run_dir, rig, resumed = crash_and_resume(project, lr.CRASH_AFTER_SAVED, rig=rig)
    assert rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"] and resumed.exit_code == 0
    assert by_question(run_dir)["q3"]["status"] == "answered"

    other = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, other, "run-b", crash_hook=crash_at(lr.CRASH_BEFORE_EXPORT))
    run_b = project.run_dir("run-b")
    assert not (run_b / "predictions.jsonl").exists() and not (run_b / "run_summary.json").exists()
    assert names(run_b)[-1] == "invocation_ended"
    exported = lr.export_run(run_b)
    assert exported.exit_code == 0 and exported.summary["run_state"] == "complete"
    assert [r["status"] for r in predictions_of(run_b)] == ["answered", "answered", "answered", "answered", "no_evidence", "answered"]
    assert other.factory_calls == 1, "export builds no provider"
    assert lr.run_status(run_b)["predictions_current"] is True


def test_a_real_process_killed_in_window_2_leaves_a_run_the_next_process_resumes(project: Project, tmp_path: Path) -> None:
    """W2 with a real kill: os._exit(9) after dispatch_started, then a fresh process (this one) resumes."""
    result = run_driver(
        project,
        """
        def hook(name, info):
            if name == "after_dispatch_started" and info["question_id"] == "q3":
                os._exit(9)
        lr.execute_run(opts, provider_factory=rig.factory, descriptor=rig.descriptor, crash_hook=hook, sleep=lambda s: None, environ={})
        """,
    )
    assert result.returncode == 9, result.stderr
    run_dir = project.run_dir("run-a")
    assert names(run_dir, "q3") == ["dispatch_started"]
    rig = Rig()
    resumed = go(project, rig)
    assert rig.sent(run_dir) == ["q4", "q6"], "q1 and q2 were saved before the kill and are not resent; q3 is unknown"
    assert resumed.exit_code == lr.EXIT_NEEDS_ATTENTION
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown"]


DRIVER_TEMPLATE = """
import os, sys
sys.path[:0] = [{src!r}, {tests!r}, {scripts!r}]
from pathlib import Path
import test_live_runner as t
from faar import live_runner as lr
project = t.Project(Path({root!r}), {{}}, [])
rig = t.Rig()
opts = t.options(project, "run-a")
{body}
"""


def run_driver(project: Project, body: str, *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    script = DRIVER_TEMPLATE.format(
        src=str(SRC_DIR), tests=str(TESTS_DIR), scripts=str(SCRIPTS_DIR), root=str(project.root), body=textwrap.dedent(body)
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("OPENAI")}
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, input=stdin, timeout=120)


def test_ctrl_c_ends_the_invocation_as_interrupted_and_marks_the_open_attempt_unknown(project: Project) -> None:
    """W2 by signal: KeyboardInterrupt during dispatch records outcome_unknown at once, ends the invocation and exports."""

    def interrupt(name: str, info: Mapping[str, Any]) -> None:
        if name == lr.CRASH_AFTER_DISPATCH and info["question_id"] == "q3":
            raise KeyboardInterrupt

    rig = Rig()
    result = go(project, rig, crash_hook=interrupt)
    run_dir = project.run_dir("run-a")
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown"]
    assert events_of(run_dir)[-1]["reason"] == "interrupted" and result.exit_code == lr.EXIT_NEEDS_ATTENTION
    assert rig.sent(run_dir) == ["q1", "q2"] and by_question(run_dir)["q3"]["awaiting_reconciliation"] is True
    assert by_question(run_dir)["q4"]["unserved"] is True


def test_a_provider_that_raises_something_other_than_provider_error_leaves_the_outcome_unknown(project: Project) -> None:
    """A bug in a provider after send may hide a processed request, so it is unknown, not retried and not a crash."""

    class Broken:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, request: Any) -> Any:
            self.calls += 1
            raise RuntimeError("adapter bug")

    broken = Broken()
    result = lr.execute_run(
        options(project), provider_factory=lambda ctx: broken, descriptor={"adapter": "broken"}, sleep=lambda s: None, environ={}
    )
    run_dir = project.run_dir("run-a")
    assert broken.calls == 3, "each question is tried once, none is retried, and the circuit breaker stops after three"
    unknown = [e for e in events_of(run_dir) if e["event"] == "outcome_unknown"]
    assert len(unknown) == 3 and unknown[0]["kind"] == "provider_exception" and "adapter bug" in unknown[0]["message"]
    assert result.exit_code == lr.EXIT_NEEDS_ATTENTION


def test_the_response_file_keeps_the_raw_payload_and_the_parsed_fields(project: Project) -> None:
    """The raw provider payload is stored unchanged next to the parsed answer, before the response_saved event names it."""
    rig = Rig(steps={"q1": [FakeStep("answer", text="Answer:  twelve months  "), ]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    event = next(e for e in events_of(run_dir) if e["event"] == "response_saved" and e["question_id"] == "q1")
    payload = json.loads((run_dir / event["response_file"]).read_text())
    assert payload["provider_response"]["text"] == "Answer:  twelve months  "
    assert payload["answer"] == "twelve months" == event["answer"] and payload["output_status"] == "ok"
    assert payload["provider_response"]["raw"]["simulated"] is True and payload["provider_response"]["raw"]["choices"]
    assert payload["usage"] == event["usage"] and payload["measured_cost"] == event["measured_cost"]
    assert payload["attempt_id"] == event["attempt_id"]


@pytest.mark.parametrize(
    ("step", "output_status", "answer", "abstained"),
    [
        (FakeStep("abstain"), "abstained", "", True),
        (FakeStep("empty"), "empty", "", False),
        (FakeStep("truncated"), "truncated", "FAKE PARTIAL ANSWER", False),
        (FakeStep("refusal"), "refusal", "", False),
    ],
)
def test_an_abstention_an_empty_reply_a_cut_reply_and_a_refusal_are_answers(project: Project, step: FakeStep, output_status: str, answer: str, abstained: bool) -> None:
    """Study brief 15.7: these are scored as returned with an output status. They are not API failures and not no_evidence."""
    go(project, Rig(steps={"q1": [step]}))
    record = by_question(project.run_dir("run-a"))["q1"]
    assert (record["status"], record["output_status"], record["answer"], record["abstained"]) == ("answered", output_status, answer, abstained)
    assert record["failure"] is None and record["no_evidence_reason"] is None


# ---------------------------------------------------------------------------
# Unknown outcomes and reconcile
# ---------------------------------------------------------------------------


def test_reconcile_moves_a_torn_final_line_aside_and_records_it(project: Project) -> None:
    rig = Rig(steps={"q2": [FakeStep("timeout_unknown")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    torn = b'{"event":"dispatch_star'
    with (run_dir / "attempts.jsonl").open("ab") as handle:
        handle.write(torn)
    attempt_id = by_question(run_dir)["q2"]["attempt_ids"][0]
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=attempt_id, resolution="mark_failed", note="checked")
    event = events_of(run_dir)[-1]
    assert event["event"] == "reconciled" and event["recovered_uncommitted_tail"]["bytes"] == len(torn)
    (side,) = run_dir.glob("attempts.uncommitted.*")
    assert side.read_bytes() == torn and lr.run_status(run_dir)["uncommitted_tail_bytes"] == 0


def test_reconcile_can_resolve_an_orphaned_dispatch_before_any_new_run(project: Project) -> None:
    """After a crash in window 2, reconcile itself records outcome_unknown and then the resolution."""
    rig = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_AFTER_DISPATCH, "q3"))
    run_dir = project.run_dir("run-a")
    request = next(r for r in requests_of(run_dir) if r["question_id"] == "q3")
    result = lr.reconcile_attempt(run_dir=run_dir, attempt_id=f"{request['request_id']}-a1", resolution="mark_failed", note="crashed mid-dispatch; provider shows nothing")
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown", "reconciled"]
    assert by_question(run_dir)["q3"]["failure"]["type"] == "outcome_unknown_marked_failed"
    assert result.summary["run_state"] == "incomplete", "q4 and q6 are still pending"


def test_an_unknown_outcome_is_not_resent_and_other_questions_continue(project: Project) -> None:
    """U1: a timeout after send is recorded as outcome_unknown, never retried, and the other questions are served."""
    rig = Rig(steps={"q2": [FakeStep("timeout_unknown")]})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"]
    assert names(run_dir, "q2") == ["dispatch_started", "outcome_unknown"]
    assert result.exit_code == lr.EXIT_NEEDS_ATTENTION and result.summary["run_state"] == "needs_reconciliation"
    ended = events_of(run_dir)[-1]
    assert ended["event"] == "invocation_ended" and ended["reason"] == "needs_reconciliation"
    go(project, rig)
    assert rig.sent(run_dir).count("q2") == 1


def test_an_ambiguous_error_after_send_is_also_unknown(project: Project) -> None:
    rig = Rig(steps={"q1": [FakeStep("ambiguous")]})
    go(project, rig)
    assert names(project.run_dir("run-a"), "q1") == ["dispatch_started", "outcome_unknown"]


def test_reconcile_mark_failed_ends_the_question_as_execution_failed(project: Project) -> None:
    rig = Rig(steps={"q2": [FakeStep("timeout_unknown")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    attempt_id = by_question(run_dir)["q2"]["attempt_ids"][0]
    result = lr.reconcile_attempt(run_dir=run_dir, attempt_id=attempt_id, resolution="mark_failed", note="provider log shows no request at that time")
    assert result.summary["run_state"] == "complete" and result.exit_code == 0
    record = by_question(run_dir)["q2"]
    assert record["status"] == "execution_failed" and record["failure"]["type"] == "outcome_unknown_marked_failed"
    assert "no request at that time" in record["failure"]["message"] and record["awaiting_reconciliation"] is False
    assert summary_of(run_dir)["exit_code"] == lr.EXIT_EXECUTION_FAILED
    assert summary_of(run_dir)["cost"]["reserved"] > 0, "the unknown attempt stays counted at its upper bound"
    reconciled = [e for e in events_of(run_dir) if e["event"] == "reconciled"]
    assert reconciled[0]["note"].startswith("provider log") and reconciled[0]["resolution"] == "mark_failed"


def test_reconcile_refuses_bad_requests_and_changes_nothing(project: Project) -> None:
    """U2: unknown attempt id, an attempt that is not unknown, an empty note, a double reconcile, or a missing run."""
    rig = Rig(steps={"q2": [FakeStep("timeout_unknown")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    saved_attempt = by_question(run_dir)["q1"]["attempt_ids"][0]
    unknown_attempt = by_question(run_dir)["q2"]["attempt_ids"][0]
    before = strip_lock(file_snapshot(run_dir))
    for kwargs, pattern in [
        ({"attempt_id": "nope-a1", "note": "x"}, "no attempt"),
        ({"attempt_id": saved_attempt, "note": "x"}, "not awaiting reconciliation"),
        ({"attempt_id": unknown_attempt, "note": "  "}, "note"),
    ]:
        with pytest.raises(RunnerRefusal, match=pattern):
            lr.reconcile_attempt(run_dir=run_dir, resolution="mark_failed", **kwargs)
    with pytest.raises(RunnerRefusal, match="resolution"):
        lr.reconcile_attempt(run_dir=run_dir, attempt_id=unknown_attempt, resolution="delete_it", note="x")
    assert strip_lock(file_snapshot(run_dir)) == before
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=unknown_attempt, resolution="mark_failed", note="checked")
    with pytest.raises(RunnerRefusal, match="not awaiting reconciliation"):
        lr.reconcile_attempt(run_dir=run_dir, attempt_id=unknown_attempt, resolution="mark_failed", note="again")
    with pytest.raises(RunnerRefusal, match="not a run directory"):
        lr.reconcile_attempt(run_dir=run_dir / "missing", attempt_id="a", resolution="mark_failed", note="x")


def strip_lock(files: dict[str, bytes]) -> dict[str, bytes]:
    """The lock file holds the last holder's pid; every other byte must stay."""
    return {k: v for k, v in files.items() if k not in ("run.lock", "predictions.jsonl", "run_summary.json")}


def test_reconcile_allow_new_attempt_is_bounded_by_the_retry_limit(project: Project) -> None:
    """U2: a question at the policy's attempt limit cannot get another attempt through reconcile."""
    rig = Rig(steps={"q1": [FakeStep("timeout_unknown"), FakeStep("timeout_unknown"), FakeStep("timeout_unknown"), FakeStep("answer")]})
    run_dir = project.run_dir("run-a")
    go(project, rig)
    for round_number in (1, 2):
        attempt_id = by_question(run_dir)["q1"]["attempt_ids"][-1]
        lr.reconcile_attempt(run_dir=run_dir, attempt_id=attempt_id, resolution="allow_new_attempt", note=f"round {round_number}")
        go(project, rig)
    assert len(by_question(run_dir)["q1"]["attempt_ids"]) == 3
    third = by_question(run_dir)["q1"]["attempt_ids"][-1]
    with pytest.raises(RunnerRefusal, match="limit"):
        lr.reconcile_attempt(run_dir=run_dir, attempt_id=third, resolution="allow_new_attempt", note="one more")
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=third, resolution="mark_failed", note="give up")
    assert by_question(run_dir)["q1"]["status"] == "execution_failed"


def test_a_step_index_survives_a_fresh_invocation(project: Project) -> None:
    """The fake script advances with attempts across invocations: attempt 2 uses step 2 even in a new provider."""
    rig = Rig(steps={"q1": [FakeStep("timeout_unknown"), FakeStep("answer", text="second attempt")]})
    run_dir = project.run_dir("run-a")
    go(project, rig)
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=by_question(run_dir)["q1"]["attempt_ids"][0], resolution="allow_new_attempt", note="n")
    go(project, rig)
    assert by_question(run_dir)["q1"]["answer"] == "second attempt"


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def test_a_retryable_error_is_retried_with_policy_delays_and_every_attempt_is_kept(project: Project) -> None:
    """R1: two 408 answers, then success. Three attempts, the injected sleep gets the policy delays."""
    rig = Rig(steps={"q1": [FakeStep("retryable_error"), FakeStep("retryable_error"), FakeStep("answer", text="third time")]})
    sleeps: list[float] = []
    result = go(project, rig, sleep=sleeps.append)
    run_dir = project.run_dir("run-a")
    assert result.exit_code == 0
    assert names(run_dir, "q1") == [
        "dispatch_started", "attempt_failed", "dispatch_started", "attempt_failed", "dispatch_started", "response_saved",
    ]
    failed = [e for e in events_of(run_dir) if e["event"] == "attempt_failed"]
    assert [(e["attempt"], e["decision"], e["outcome"], e["http_status"], e["retryable"]) for e in failed] == [
        (1, "retry", "rejected", 408, True),
        (2, "retry", "rejected", 408, True),
    ]
    real = lr.Services.default().retry_policy
    assert sleeps == [real.delay_for(1), real.delay_for(2)] and [e["delay_seconds"] for e in failed] == sleeps
    record = by_question(run_dir)["q1"]
    assert record["answer"] == "third time" and len(record["attempt_ids"]) == 3
    assert summary_of(run_dir)["attempts"] == {"total": 7, "saved": 5, "failed": 2, "unknown": 0, "open": 0}


def test_retries_stop_at_the_attempt_limit_and_the_question_fails(project: Project) -> None:
    """R1: a request that always fails retryably makes exactly max_attempts attempts, then execution_failed."""
    rig = Rig(default=FakeStep("answer"), steps={"q1": [FakeStep("retryable_error")] * 9})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    attempts = lr.Services.default().retry_policy.max_attempts
    assert rig.sent(run_dir).count("q1") == attempts == 3
    decisions = [e["decision"] for e in events_of(run_dir) if e["event"] == "attempt_failed"]
    assert decisions == ["retry", "retry", "fail_question"]
    record = by_question(run_dir)["q1"]
    assert record["status"] == "execution_failed" and record["failure"]["type"] == "transient_status" and record["failure"]["attempts"] == 3
    assert result.exit_code == lr.EXIT_EXECUTION_FAILED
    assert by_question(run_dir)["q2"]["status"] == "answered", "other questions are unaffected"


def test_a_non_retryable_error_fails_only_its_question(project: Project) -> None:
    rig = Rig(steps={"q3": [FakeStep("non_retryable_error")]})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == SENT and names(run_dir, "q3") == ["dispatch_started", "attempt_failed"]
    assert [e["decision"] for e in events_of(run_dir) if e["event"] == "attempt_failed"] == ["fail_question"]
    assert by_question(run_dir)["q3"]["failure"]["http_status"] == 400 and result.exit_code == lr.EXIT_EXECUTION_FAILED


def test_a_connect_failure_costs_nothing_and_is_retried(project: Project) -> None:
    """A not_sent failure reserves nothing (contract rule 6) and retries."""
    rig = Rig(steps={"q1": [FakeStep("connect_error")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    failed = next(e for e in events_of(run_dir) if e["event"] == "attempt_failed")
    assert failed["outcome"] == "not_sent" and failed["decision"] == "retry"
    assert summary_of(run_dir)["cost"]["reserved"] == 0


def test_an_auth_error_stops_the_run_and_a_resume_serves_the_rest(project: Project) -> None:
    """R2: stop_run after the first auth error. Unsent questions are unserved, not abstentions. A resume continues."""
    rig = Rig(steps={"q2": [FakeStep("auth_error")]})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2"]
    assert events_of(run_dir)[-1]["reason"] == "stop_run" and result.exit_code == lr.EXIT_NEEDS_ATTENTION
    assert result.summary["run_state"] == "stopped"
    records = by_question(run_dir)
    for qid in ("q2", "q3", "q4", "q6"):
        assert records[qid]["status"] == "execution_failed" and records[qid]["unserved"] is True and records[qid]["abstained"] is False
        assert records[qid]["failure"]["type"] == "not_attempted"
    assert records["q1"]["status"] == "answered"
    resumed = go(project, rig)
    assert resumed.exit_code == 0 and rig.sent(run_dir)[2:] == ["q2", "q3", "q4", "q6"]
    assert len(by_question(run_dir)["q2"]["attempt_ids"]) == 2


# ---------------------------------------------------------------------------
# Budget and the safety ceiling
# ---------------------------------------------------------------------------


def test_the_ceiling_stops_dispatch_before_it_would_be_exceeded(project: Project) -> None:
    """B1, B3: with room for one request the run ends budget_limited, every dispatch fit, and unserved questions are not abstentions."""
    costs = bounds(project)
    ceiling = costs["q1"] + 0.000001
    rig = Rig()
    result = go(project, rig, safety_ceiling=ceiling)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1"]
    assert result.exit_code == lr.EXIT_BUDGET_LIMITED and result.summary["run_state"] == "budget_limited"
    assert result.summary["valid_baseline"] is False
    assert events_of(run_dir)[-1]["reason"] == "budget_exhausted"
    for event in events_of(run_dir):
        if event["event"] == "dispatch_started":
            assert event["ledger_before"]["measured"] + event["ledger_before"]["reserved"] + event["cost_upper_bound"] <= ceiling + 1e-9
    records = by_question(run_dir)
    for qid in ("q2", "q3", "q4", "q6"):
        r = records[qid]
        assert (r["status"], r["unserved"], r["abstained"], r["answer"]) == ("execution_failed", True, False, None)
        assert r["failure"] == {**r["failure"], "stage": "dispatch", "type": "budget_exhausted"}
    assert records["q5"]["status"] == "no_evidence" and records["q1"]["status"] == "answered"
    assert result.summary["counts"]["unserved"] == 4 and result.summary["counts"]["model_abstained"] == 0


def test_the_ceiling_boundary_is_inclusive(project: Project) -> None:
    """B1: a ceiling equal to the last dispatch's committed total plus its bound completes. One micro-unit less stops before it."""
    rig = Rig()
    go(project, rig, "full")
    dispatches = [e for e in events_of(project.run_dir("full")) if e["event"] == "dispatch_started"]
    last = dispatches[-1]
    exact = round(last["ledger_before"]["measured"] + last["ledger_before"]["reserved"] + last["cost_upper_bound"], 6)
    ok = go(project, Rig(), "at-ceiling", safety_ceiling=exact)
    assert ok.exit_code == 0
    short = go(project, Rig(), "below-ceiling", safety_ceiling=round(exact - 0.000001, 6))
    assert short.exit_code == lr.EXIT_BUDGET_LIMITED
    assert by_question(project.run_dir("below-ceiling"))["q6"]["failure"]["type"] == "budget_exhausted"


def test_raising_the_ceiling_is_recorded_and_earlier_events_stay_byte_identical(project: Project) -> None:
    """B2: --raise-safety-ceiling with a note adds ceiling_change to the new invocation_started and rewrites nothing."""
    costs = bounds(project)
    rig = Rig()
    ceiling = costs["q1"] + 0.000001
    go(project, rig, safety_ceiling=ceiling)
    run_dir = project.run_dir("run-a")
    before = (run_dir / "attempts.jsonl").read_bytes()
    result = go(project, rig, safety_ceiling=ceiling, raise_safety_ceiling=50.0, authorization_note="lead approved 50 for the dev pilot")
    after = (run_dir / "attempts.jsonl").read_bytes()
    assert after.startswith(before), "earlier events are untouched"
    assert result.exit_code == 0 and rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"]
    starts = [e for e in events_of(run_dir) if e["event"] == "invocation_started"]
    assert starts[0]["ceiling_change"] is None and starts[0]["safety_ceiling"]["amount"] == ceiling
    assert starts[1]["safety_ceiling"]["amount"] == 50.0
    assert starts[1]["ceiling_change"] == {"from": ceiling, "to": 50.0, "authorization_note": "lead approved 50 for the dev pilot"}
    assert [i["ceiling_change"] for i in summary_of(run_dir)["invocations"]][0] is None
    assert summary_of(run_dir)["safety_ceiling"]["amount"] == 50.0


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"safety_ceiling": 100.0, "raise_safety_ceiling": 200.0}, "authorization-note"),
        ({"safety_ceiling": 100.0, "raise_safety_ceiling": 200.0, "authorization_note": "  "}, "authorization-note"),
        ({"safety_ceiling": 90.0, "raise_safety_ceiling": 200.0, "authorization_note": "n"}, "does not match the current ceiling"),
        ({"safety_ceiling": 100.0, "raise_safety_ceiling": 100.0, "authorization_note": "n"}, "must be above"),
        ({"safety_ceiling": 150.0}, "above the current ceiling"),
        ({"safety_ceiling": 100.0, "authorization_note": "n"}, "belongs with"),
        ({"safety_ceiling": float("nan")}, "finite"),
        ({"safety_ceiling": -1.0}, "above zero"),
    ],
)
def test_a_bad_ceiling_change_is_refused_before_dispatch(project: Project, kwargs: dict[str, Any], pattern: str) -> None:
    """B2: the ceiling cannot rise without a note, without naming the current ceiling, or silently."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    before = file_snapshot(run_dir)
    rig = Rig()
    with pytest.raises(RunnerRefusal, match=pattern):
        go(project, rig, **kwargs)
    assert rig.factory_calls == 0 and strip_lock(file_snapshot(run_dir)) == strip_lock(before)


def test_lowering_the_ceiling_needs_no_note_and_is_recorded(project: Project) -> None:
    go(project, Rig())
    go(project, Rig(), safety_ceiling=50.0)
    last = [e for e in events_of(project.run_dir("run-a")) if e["event"] == "invocation_started"][-1]
    assert last["ceiling_change"] == {"from": 100.0, "to": 50.0, "authorization_note": None}


def test_a_raise_on_the_first_invocation_is_refused_and_leaves_no_directory(project: Project) -> None:
    with pytest.raises(RunnerRefusal, match="earlier invocation"):
        go(project, Rig(), safety_ceiling=1.0, raise_safety_ceiling=2.0, authorization_note="n")
    assert not project.run_dir("run-a").exists()


def test_an_unknown_attempt_counts_against_the_ceiling_at_its_upper_bound(project: Project) -> None:
    """B1: after an outcome_unknown, the ceiling check sees the reserved upper bound, not zero."""
    costs = bounds(project)
    rig = Rig(steps={"q1": [FakeStep("timeout_unknown")]})
    # Room for q1 and q2 at their bounds but not for q1's reservation plus q2, q3 measured costs.
    ceiling = costs["q1"] + costs["q2"] + 0.000001
    result = go(project, rig, safety_ceiling=ceiling)
    assert rig.sent(project.run_dir("run-a")) == ["q1", "q2"], "q3 is refused because q1's reservation is still counted"
    assert result.summary["cost"]["reserved"] == pytest.approx(costs["q1"])


def test_the_ledger_and_events_agree_with_the_summary(project: Project) -> None:
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    saved = [e for e in events_of(run_dir) if e["event"] == "response_saved"]
    assert summary_of(run_dir)["cost"]["measured"] == pytest.approx(sum(e["measured_cost"] for e in saved))
    assert all(e["usage"]["input_tokens"] is not None for e in saved)


def test_a_response_without_usage_reserves_its_upper_bound(project: Project) -> None:
    """Rule 6: a saved response whose usage is missing has no measured cost and stays reserved at the bound."""
    rig = Rig(steps={"q1": [FakeStep("missing_usage")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    event = next(e for e in events_of(run_dir) if e["event"] == "response_saved")
    assert event["measured_cost"] is None and event["usage"]["input_tokens"] is None
    assert summary_of(run_dir)["cost"]["reserved"] == pytest.approx(next(r["cost_upper_bound"] for r in requests_of(run_dir) if r["question_id"] == "q1"))
    assert by_question(run_dir)["q1"]["status"] == "answered"


# ---------------------------------------------------------------------------
# The run lock
# ---------------------------------------------------------------------------


def test_a_held_lock_refuses_a_second_invocation_before_any_dispatch(project: Project) -> None:
    """D4: while one holder has the lock, run, export and reconcile refuse and write nothing."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    before = file_snapshot(run_dir)
    holder = lr.RunLock(run_dir, "holder")
    holder.acquire()
    try:
        rig = Rig()
        with pytest.raises(RunnerRefusal, match="another invocation holds"):
            go(project, rig)
        with pytest.raises(RunnerRefusal, match="another invocation holds"):
            lr.export_run(run_dir)
        with pytest.raises(RunnerRefusal, match="another invocation holds"):
            lr.reconcile_attempt(run_dir=run_dir, attempt_id="x", resolution="mark_failed", note="n")
        assert rig.factory_calls == 0
        assert lr.run_status(run_dir)["lock_held"] is True
    finally:
        holder.release()
    assert strip_lock(file_snapshot(run_dir)) == strip_lock(before)
    assert lr.run_status(run_dir)["lock_held"] is False


def test_two_real_processes_cannot_both_own_the_run(project: Project) -> None:
    """D4: one process holds the lock and a second process, running the real CLI, is refused with exit 1 and dispatches nothing."""
    run_dir = project.run_dir("run-a")
    run_dir.mkdir(parents=True)
    script = DRIVER_TEMPLATE.format(
        src=str(SRC_DIR),
        tests=str(TESTS_DIR),
        scripts=str(SCRIPTS_DIR),
        root=str(project.root),
        body=textwrap.dedent(
            """
            lock = lr.RunLock(opts.run_dir, "holder-process")
            lock.acquire()
            print("LOCKED", flush=True)
            sys.stdin.readline()
            lock.release()
            """
        ),
    )
    holder = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        fake_script = project.root / "fake.json"
        fake_script.write_text('{"script": {}}')
        second = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "run_pilot_live.py"), "run",
             "--project-root", str(project.root), "--pilot-id", PILOT_ID, "--run-dir", str(run_dir),
             "--fake-script", str(fake_script), "--safety-ceiling", "10"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC_DIR)}, timeout=120,
        )
        assert second.returncode == lr.EXIT_REFUSED, second.stdout + second.stderr
        assert "another invocation holds" in second.stderr and "holder-process" in second.stderr
        assert sorted(p.name for p in run_dir.iterdir()) == ["run.lock"]
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=30)
    # With the lock free, the same command works.
    rig = Rig()
    assert go(project, rig).exit_code == 0


# ---------------------------------------------------------------------------
# The event log and the response files
# ---------------------------------------------------------------------------


def test_a_torn_final_line_is_ignored_reported_and_moved_aside_on_the_next_run(project: Project) -> None:
    """E1: a last line without a newline is uncommitted. status reports it, run keeps the bytes in a side file and appends cleanly."""
    rig = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_AFTER_SAVED, "q2"))
    run_dir = project.run_dir("run-a")
    torn = b'{"event":"dispatch_started","invocation_id":"abc","seq":9,"at":"2026'
    with (run_dir / "attempts.jsonl").open("ab") as handle:
        handle.write(torn)
    committed = len(events_of(run_dir))
    status = lr.run_status(run_dir)
    assert status["uncommitted_tail_bytes"] == len(torn) and status["requests"]["answered"] == 2
    assert (run_dir / "attempts.jsonl").read_bytes().endswith(torn), "status changes nothing"
    resumed = go(project, rig)
    assert resumed.exit_code == 0
    side = sorted(run_dir.glob("attempts.uncommitted.*"))
    assert len(side) == 1 and side[0].read_bytes() == torn
    events = events_of(run_dir)
    assert len(events) > committed and [e["seq"] for e in events] == list(range(1, len(events) + 1))
    start = next(e for e in events if e["event"] == "invocation_started" and e["seq"] == committed + 1)
    assert start["recovered_uncommitted_tail"] == {"file": side[0].name, "bytes": len(torn), "sha256": sha256_bytes(torn)}


def test_a_complete_final_line_without_a_newline_is_still_uncommitted(project: Project) -> None:
    """E1: only a newline commits a line. A parseable last line without one is set aside, not trusted."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    last = (run_dir / "attempts.jsonl").read_bytes().rstrip(b"\n")
    (run_dir / "attempts.jsonl").write_bytes(last)
    committed = len(events_of(run_dir))
    assert committed == len(last.splitlines()) - 1
    assert lr.run_status(run_dir)["uncommitted_tail_bytes"] == len(last.splitlines()[-1])
    go(project, Rig())
    assert len(list(run_dir.glob("attempts.uncommitted.*"))) == 1
    assert [e["event"] for e in events_of(run_dir)][-2:] == ["invocation_started", "invocation_ended"]


def test_a_ceiling_flag_that_disagrees_with_the_price_table_is_refused_by_the_ledger(project: Project) -> None:
    """The ledger rejects a log whose ceiling is not in the price table's simulated currency. The driver refuses with exit 1."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    lines = (run_dir / "attempts.jsonl").read_bytes().splitlines(keepends=True)
    first = json.loads(lines[0])
    first["safety_ceiling"]["simulated"] = False
    lines[0] = (json.dumps(first) + "\n").encode()
    (run_dir / "attempts.jsonl").write_bytes(b"".join(lines))
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="ledger refused"):
        go(project, rig)
    with pytest.raises(RunnerRefusal, match="ledger refused"):
        lr.run_status(run_dir)
    assert rig.factory_calls == 0


@pytest.mark.parametrize(
    "damage",
    ["garbage_line", "wrong_seq", "unknown_event", "missing_key", "not_object", "unknown_request", "double_resolution"],
)
def test_a_corrupt_event_log_is_refused_by_every_command_and_left_untouched(project: Project, damage: str) -> None:
    """E1: any committed line that does not parse, breaks the sequence, or contradicts the requests refuses run, status and export."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    lines = (run_dir / "attempts.jsonl").read_bytes().splitlines(keepends=True)
    if damage == "garbage_line":
        lines[3] = b"this is not json\n"
    elif damage == "wrong_seq":
        record = json.loads(lines[3])
        record["seq"] = 99
        lines[3] = (json.dumps(record) + "\n").encode()
    elif damage == "unknown_event":
        record = json.loads(lines[3])
        record["event"] = "made_up"
        lines[3] = (json.dumps(record) + "\n").encode()
    elif damage == "missing_key":
        record = json.loads(lines[1])
        del record["attempt_id"]
        lines[1] = (json.dumps(record) + "\n").encode()
    elif damage == "not_object":
        lines[3] = b"[1, 2]\n"
    elif damage == "unknown_request":
        record = json.loads(lines[1])
        record["request_id"] = "f" * 32
        lines[1] = (json.dumps(record) + "\n").encode()
    else:
        lines.insert(3, lines[2].replace(b'"seq":3', b'"seq":4'))
        lines = [lines[0], *lines[1:]]
    (run_dir / "attempts.jsonl").write_bytes(b"".join(lines))
    before = file_snapshot(run_dir)
    rig = Rig()
    for action in (lambda: go(project, rig), lambda: lr.run_status(run_dir), lambda: lr.export_run(run_dir)):
        with pytest.raises(RunnerRefusal):
            action()
    assert rig.factory_calls == 0
    assert strip_lock(file_snapshot(run_dir)) == strip_lock(before)


def test_a_tampered_or_missing_response_file_is_refused(project: Project) -> None:
    """E1: a response_saved event whose file changed or vanished stops every command. A saved answer is never trusted blindly."""
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    target = next(iter(sorted((run_dir / "responses").iterdir())))
    original = target.read_bytes()
    target.write_bytes(original.replace(b"twelve months", b"twenty months"))
    with pytest.raises(RunnerRefusal, match="response file changed"):
        lr.export_run(run_dir)
    target.unlink()
    with pytest.raises(RunnerRefusal, match="missing or unreadable"):
        lr.run_status(run_dir)
    target.write_bytes(original)
    assert lr.export_run(run_dir).exit_code == 0


def test_requests_jsonl_is_pinned_by_the_config_hash(project: Project) -> None:
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    path = run_dir / "requests.jsonl"
    path.write_text(path.read_text().replace("Alpha", "Omega", 1))
    with pytest.raises(RunnerRefusal, match="prepared requests changed"):
        lr.run_status(run_dir)


def test_a_full_fsync_happens_for_every_event(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """The log is fsynced after every line: at least one os.fsync per event appended."""
    calls: list[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    go(project, Rig())
    assert len(calls) >= len(events_of(project.run_dir("run-a")))


# ---------------------------------------------------------------------------
# Export, status and score
# ---------------------------------------------------------------------------


def test_export_is_rebuilt_from_the_records_and_repeating_it_changes_nothing(project: Project) -> None:
    """E2: delete the exports and rebuild them byte for byte from the three record files."""
    rig = Rig()
    go(project, rig)
    run_dir = project.run_dir("run-a")
    predictions, summary = (run_dir / "predictions.jsonl").read_bytes(), (run_dir / "run_summary.json").read_bytes()
    (run_dir / "predictions.jsonl").unlink()
    (run_dir / "run_summary.json").unlink()
    lr.export_run(run_dir)
    assert (run_dir / "predictions.jsonl").read_bytes() == predictions and (run_dir / "run_summary.json").read_bytes() == summary
    lr.export_run(run_dir)
    assert (run_dir / "predictions.jsonl").read_bytes() == predictions
    assert rig.factory_calls == 1


def test_every_run_state_keeps_every_question_in_the_export(project: Project) -> None:
    """E2: complete, budget_limited, needs_reconciliation, stopped and incomplete all export exactly six records."""
    costs = bounds(project)
    states: dict[str, str] = {}

    def check(name: str, **kwargs: Any) -> None:
        rig = kwargs.pop("rig", None) or Rig()
        hook = kwargs.pop("hook", None)
        try:
            go(project, rig, name, crash_hook=hook, **kwargs)
        except SimulatedCrash:
            pass
        run_dir = project.run_dir(name)
        exported = lr.export_run(run_dir)
        states[name] = exported.summary["run_state"]
        records = predictions_of(run_dir)
        assert [r["question_id"] for r in records] == QUESTIONS
        assert {r["status"] for r in records} <= {"answered", "no_evidence", "execution_failed"}
        assert all(r["abstained"] is False for r in records if r["status"] == "execution_failed")

    check("s-complete")
    check("s-budget", safety_ceiling=costs["q1"] + 0.000001)
    check("s-unknown", rig=Rig(steps={"q2": [FakeStep("timeout_unknown")]}))
    check("s-stopped", rig=Rig(steps={"q2": [FakeStep("auth_error")]}))
    check("s-incomplete", hook=crash_at(lr.CRASH_BEFORE_DISPATCH, "q3"))
    assert states == {
        "s-complete": "complete",
        "s-budget": "budget_limited",
        "s-unknown": "needs_reconciliation",
        "s-stopped": "stopped",
        "s-incomplete": "incomplete",
    }


def test_status_describes_a_run_and_changes_nothing(project: Project) -> None:
    rig = Rig(steps={"q2": [FakeStep("timeout_unknown")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    before = file_snapshot(run_dir)
    status = lr.run_status(run_dir)
    assert status["run_state"] == "needs_reconciliation" and status["requests"]["unknown"] == 1
    assert status["awaiting_reconciliation"] == by_question(run_dir)["q2"]["attempt_ids"]
    text = lr.format_status(status)
    assert "awaiting reconciliation" in text and "simulated" in text
    assert file_snapshot(run_dir) == before


def test_score_joins_predictions_and_counts_a_failed_question_as_zero_not_an_abstention(project: Project) -> None:
    """B3: a complete run with one execution_failed question scores; the failure is a zero and never an abstention."""
    go(project, Rig(steps={"q2": [FakeStep("non_retryable_error")]}))
    run_dir = project.run_dir("run-a")
    result = lr.score_run_live(project_root=project.root, run_dir=run_dir)
    counts = result.summary["counts"]
    assert counts["questions"] == 6 and counts["execution_failed"] == 1 and counts["answered"] == 4 and counts["no_evidence"] == 1
    assert counts["abstained"] == 1, "only the no_evidence question abstains; the failed one does not"
    rows = {r["question_id"]: r for r in map(json.loads, (run_dir / "scores.jsonl").read_text().splitlines())}
    assert rows["q1"]["em"] == 1 and rows["q2"]["scored_by"] == "failure_as_zero"
    assert result.summary["run_state"] == "complete" and result.summary["valid_baseline"] is False
    assert result.exit_code == lr.EXIT_EXECUTION_FAILED
    again = lr.score_run_live(project_root=project.root, run_dir=run_dir)
    assert "verified identical" in again.message


def test_a_scored_run_cannot_take_more_dispatches(project: Project) -> None:
    """Scoring is final: a later run in the same directory is refused, so scores never describe a moving target."""
    rig = Rig()
    go(project, rig)
    lr.score_run_live(project_root=project.root, run_dir=project.run_dir("run-a"))
    with pytest.raises(RunnerRefusal, match="has been scored"):
        go(project, Rig())


def test_only_score_opens_the_evaluation_manifest(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """L1: the file spy sees the evaluation manifest during score and at no other time."""
    opened = spy_on_opens(monkeypatch)
    go(project, Rig())
    lr.run_status(project.run_dir("run-a"))
    lr.export_run(project.run_dir("run-a"))
    lr.dry_run(project_root=project.root, out_dir=project.root / "dry", pilot_id=PILOT_ID, config=lr.FAKE_CONFIG)
    assert str(project.evaluation_manifest.resolve()) not in reads_under(opened, project.root)
    lr.score_run_live(project_root=project.root, run_dir=project.run_dir("run-a"))
    assert str(project.evaluation_manifest.resolve()) in reads_under(opened, project.root)


def test_score_refuses_a_manifest_for_another_pilot_and_keeps_no_partial_files(project: Project) -> None:
    go(project, Rig())
    project.evaluation_manifest.write_text(json.dumps({"pilot_id": "other", "questions": {}}))
    with pytest.raises(RunnerRefusal, match="pilot"):
        lr.score_run_live(project_root=project.root, run_dir=project.run_dir("run-a"))
    assert not (project.run_dir("run-a") / "scores.jsonl").exists()


# ---------------------------------------------------------------------------
# Offline guarantees and live refusal
# ---------------------------------------------------------------------------


def test_help_dry_run_status_and_export_build_no_provider_and_no_client(project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """O1: through the CLI entry point, these commands never call a provider factory or an SDK client."""
    go(project, Rig())
    built = tripwire_clients(monkeypatch)
    run_dir = str(project.run_dir("run-a"))
    for argv in (["--help"], ["dry-run", "--help"], ["run", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        assert excinfo.value.code == 0
    assert cli.main(["dry-run", "--project-root", str(project.root), "--pilot-id", PILOT_ID, "--out", str(project.root / "dry")]) == 0
    assert cli.main(["status", "--run-dir", run_dir]) == 0
    assert cli.main(["status", "--run-dir", run_dir, "--json"]) == 0
    assert cli.main(["export", "--run-dir", run_dir]) == 0
    assert built == []
    out = capsys.readouterr().out
    assert "exit codes:" in out and "budget_limited" in out and "results/pilots" in out


def test_importing_the_modules_imports_no_sdk_and_builds_no_client() -> None:
    code = "import sys; import faar.live_runner, faar.pilot_runner; sys.path.insert(0, %r); import run_pilot_live; assert 'openai' not in sys.modules and 'httpx' not in sys.modules, sorted(m for m in sys.modules if m in ('openai','httpx'))" % str(TESTS_DIR.parent / "scripts" / "experiments")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC_DIR)})
    assert proc.returncode == 0, proc.stderr


class RecordingEnv(dict):
    """An environ that records every key read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.reads: list[str] = []

    def get(self, key: str, default: Any = None) -> Any:
        self.reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key: str) -> Any:
        self.reads.append(key)
        return super().__getitem__(key)


def test_a_fake_run_needs_no_credential_and_opens_no_socket(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """O3: with no key, no network access and a recording environment, a fake run completes and touches neither."""
    touched: list[str] = []

    def record(name: str) -> Any:
        def inner(*args: Any, **kwargs: Any) -> Any:
            touched.append(name)
            raise OSError(f"{name} is not allowed in a fake run")

        return inner

    for attr in ("connect", "connect_ex", "bind", "sendto"):
        monkeypatch.setattr(socket.socket, attr, record(attr))
    monkeypatch.setattr(socket, "getaddrinfo", record("getaddrinfo"))
    monkeypatch.setattr(socket, "create_connection", record("create_connection"))
    environ = RecordingEnv({"PATH": "/usr/bin"})
    rig = Rig()
    result = lr.execute_run(
        options(project), provider_factory=rig.factory, descriptor=rig.descriptor, sleep=lambda s: None, environ=environ
    )
    assert result.exit_code == 0 and touched == []
    assert "OPENAI_API_KEY" not in environ.reads
    assert "OPENAI_API_KEY" not in os.environ


def test_no_secret_shaped_value_reaches_the_run_directory(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-never-be-read")
    go(project, Rig())
    for path in project.run_dir("run-a").rglob("*"):
        if path.is_file():
            assert b"sk-test-should-never-be-read" not in path.read_bytes()


LIVE_ARGS = ["run", "--mode", "live", "--pilot-id", PILOT_ID]


@pytest.mark.parametrize(
    ("missing", "extra_args", "env", "message"),
    [
        ("everything", [], {}, "--provider-config"),
        ("provider config", ["--safety-ceiling", "1"], {lr.LIVE_ENV_NAME: lr.LIVE_ENV_VALUE}, "--provider-config"),
        ("safety ceiling", ["--provider-config", "CFG"], {lr.LIVE_ENV_NAME: lr.LIVE_ENV_VALUE}, "--safety-ceiling"),
        ("environment variable", ["--provider-config", "CFG", "--safety-ceiling", "1"], {}, lr.LIVE_ENV_NAME),
        ("wrong env value", ["--provider-config", "CFG", "--safety-ceiling", "1"], {lr.LIVE_ENV_NAME: "yes"}, lr.LIVE_ENV_NAME),
    ],
)
def test_live_mode_refuses_without_every_requirement_and_builds_nothing(
    project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str, extra_args: list[str], env: dict[str, str], message: str
) -> None:
    """O2: each missing live requirement refuses with exit 1 before anything is prepared; no client, no key read, no directory."""
    config = project.root / "provider.json"
    config.write_text(json.dumps(valid_live_config()))
    built = tripwire_clients(monkeypatch)
    monkeypatch.setattr(os, "environ", RecordingEnv({**env, "OPENAI_API_KEY": "sk-test-not-a-key"}))
    run_dir = project.root / "results" / "development" / "live-a"
    argv = [*LIVE_ARGS, "--project-root", str(project.root), "--run-dir", str(run_dir), *[str(config) if a == "CFG" else a for a in extra_args]]
    assert cli.main(argv) == lr.EXIT_REFUSED
    assert message in capsys.readouterr().err
    assert built == [] and "OPENAI_API_KEY" not in os.environ.reads
    assert not run_dir.exists() and not (project.root / "results" / "engineering").exists()


def valid_live_config() -> dict[str, Any]:
    return {
        "provider": "openai",
        "model": "gpt-4o-2024-11-20",
        "endpoint": "https://api.openai.com/v1",
        "params": {"temperature": 0},
        "max_output_tokens": 128,
        "timeout_seconds": 60,
        "tokenizer_bound": "utf8-bytes",
        "prices": {"currency": "USD", "input_per_million": 2.5, "cached_input_per_million": 1.25, "output_per_million": 10.0, "source": "test fixture", "source_date": "2026-09-29", "service_tier": "default"},
    }


@pytest.mark.parametrize(
    "damage",
    [
        {"tokenizer_bound": "cl100k"},
        {"tokenizer_bound": None},
        {"max_output_tokens": None},
        {"max_output_tokens": 0},
        {"timeout_seconds": "60"},
        {"prices": None},
        {"prices": {"currency": "USD", "input_per_million": 2.5, "output_per_million": 10.0, "source": "", "source_date": "d"}},
        {"prices": {"currency": "USD", "input_per_million": float("nan"), "output_per_million": 10.0, "source": "s", "source_date": "d"}},
        {"prices": {"currency": "USD", "input_per_million": 0, "output_per_million": 10.0, "source": "s", "source_date": "d"}},
        {"token_limit_param": "max_output"},
        {"params": {"temperature": 0, "max_tokens": 5}},
        {"params": {"stream": True}},
    ],
)
def test_an_invalid_live_provider_config_refuses_with_all_requirements_present(project: Project, monkeypatch: pytest.MonkeyPatch, damage: dict[str, Any]) -> None:
    """O2: even with the env variable and ceiling, a config that misses a rule-11 field refuses before a client exists."""
    payload = {**valid_live_config(), **damage}
    config = project.root / "provider.json"
    config.write_text(json.dumps(payload))
    built = tripwire_clients(monkeypatch)
    monkeypatch.setattr(os, "environ", {lr.LIVE_ENV_NAME: lr.LIVE_ENV_VALUE})
    run_dir = project.root / "results" / "development" / "live-a"
    argv = [*LIVE_ARGS, "--project-root", str(project.root), "--run-dir", str(run_dir), "--provider-config", str(config), "--safety-ceiling", "1"]
    assert cli.main(argv) == lr.EXIT_REFUSED
    assert built == [] and not run_dir.exists()


def test_a_live_run_directory_must_be_under_results_development(project: Project) -> None:
    """P1: live runs use results/development/<run_id>/ inside the project, fake runs results/engineering/<run_id>/."""
    with pytest.raises(RunnerRefusal, match="results/development"):
        lr.refuse_run_directory(project.run_dir("x"), project.root, "live")
    with pytest.raises(RunnerRefusal, match="results/engineering"):
        lr.refuse_run_directory(project.root / "results" / "development" / "x", project.root, "fake")
    lr.refuse_run_directory(project.root / "results" / "development" / "x", project.root, "live")
    lr.refuse_run_directory(project.run_dir("x"), project.root, "fake")
    lr.refuse_run_directory(project.root.parent / "elsewhere" / "x", project.root, "fake")


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sub", ["results/pilots/run-a", "results/Pilots/run-a", "results/run-a", "run-a", "config/run-a", "OHR-Bench/run-a"])
def test_a_run_outside_results_engineering_or_under_pilots_is_refused_and_leaves_nothing(project: Project, sub: str) -> None:
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="refusing"):
        lr.execute_run(
            lr.RunOptions(project_root=project.root, run_dir=project.root / sub, pilot_id=PILOT_ID, safety_ceiling=1.0),
            provider_factory=rig.factory,
            descriptor=rig.descriptor,
            environ={},
        )
    assert not (project.root / sub).exists() and rig.factory_calls == 0


def test_a_run_may_live_outside_the_project(project: Project, tmp_path: Path) -> None:
    rig = Rig()
    result = lr.execute_run(
        lr.RunOptions(project_root=project.root, run_dir=tmp_path / "outside" / "run-o", pilot_id=PILOT_ID, safety_ceiling=1.0),
        provider_factory=rig.factory,
        descriptor=rig.descriptor,
        environ={},
    )
    assert result.exit_code == 0


def test_an_unrecognised_or_partial_directory_is_refused_and_kept(project: Project) -> None:
    run_dir = project.run_dir("run-a")
    run_dir.mkdir(parents=True)
    (run_dir / "notes.txt").write_text("mine")
    with pytest.raises(RunnerRefusal, match="unrecognised entries"):
        go(project, Rig())
    assert (run_dir / "notes.txt").read_text() == "mine"
    (run_dir / "notes.txt").unlink()
    (run_dir / "requests.jsonl").write_text("{}")
    (run_dir / "attempts.jsonl").write_text("")
    with pytest.raises(RunnerRefusal, match="partial run"):
        go(project, Rig())


def test_a_crash_between_requests_and_config_leaves_a_directory_the_next_run_can_finish(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialisation writes requests.jsonl, then run_config.json. A crash between them holds no events, so it may be redone."""
    run_dir = project.run_dir("run-a")
    real = lr.atomic_write_text
    seen: list[str] = []

    def dying(path: Path, text: str) -> None:
        if path.name == "run_config.json":
            raise SimulatedCrash("crash before run_config.json")
        seen.append(path.name)
        real(path, text)

    monkeypatch.setattr(lr, "atomic_write_text", dying)
    with pytest.raises(SimulatedCrash):
        go(project, Rig())
    monkeypatch.setattr(lr, "atomic_write_text", real)
    assert seen == ["requests.jsonl"] and not (run_dir / "run_config.json").exists()
    assert go(project, Rig()).exit_code == 0


def test_a_stale_temporary_file_from_a_crashed_atomic_write_is_tolerated(project: Project) -> None:
    go(project, Rig())
    run_dir = project.run_dir("run-a")
    (run_dir / ".predictions.jsonl.abc123.tmp").write_text("partial")
    assert lr.export_run(run_dir).exit_code == 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def write_fake_script(project: Project, script: dict[str, Any] | None = None, name: str = "fake.json") -> Path:
    path = project.root / name
    path.write_text(json.dumps({"script": script or {}, "default": {"kind": "answer", "text": "twelve months"}}))
    return path


def cli_run(project: Project, name: str, fake: Path, *extra: str) -> int:
    return cli.main(
        ["run", "--project-root", str(project.root), "--pilot-id", PILOT_ID, "--run-dir", str(project.run_dir(name)),
         "--fake-script", str(fake), "--safety-ceiling", "10", *extra]
    )


def test_the_cli_exit_codes_mean_what_the_help_says(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    """C1: 0 complete, 2 execution failures, 4 budget limited, 5 needs reconciliation or stopped, 1 refusal."""
    assert cli_run(project, "cli-ok", write_fake_script(project)) == 0
    assert cli_run(project, "cli-fail", write_fake_script(project, {"q1": [{"kind": "non_retryable_error"}]}, "f2.json")) == 2
    unknown = write_fake_script(project, {"q2": [{"kind": "timeout_unknown"}]}, "f5.json")
    assert cli_run(project, "cli-unknown", unknown) == 5
    stopped = write_fake_script(project, {"q2": [{"kind": "auth_error"}]}, "f6.json")
    assert cli_run(project, "cli-stopped", stopped) == 5
    assert cli.main(["run", "--run-dir", str(project.run_dir("cli-refused"))]) == 1, "missing --fake-script and ceiling"
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["nonsense"])
    assert excinfo.value.code == 1
    capsys.readouterr()


def test_the_cli_budget_limited_exit_code(project: Project) -> None:
    costs = bounds(project)
    fake = write_fake_script(project)
    args = ["run", "--project-root", str(project.root), "--pilot-id", PILOT_ID, "--run-dir", str(project.run_dir("cli-budget")),
            "--fake-script", str(fake), "--safety-ceiling", str(costs["q1"] + 0.000001)]
    assert cli.main(args) == 4


def test_a_cli_process_killed_by_a_fake_crash_step_resumes_in_a_fresh_process(project: Project) -> None:
    """W3 end to end through the CLI: crash_after_send kills the process with SIGKILL, and a new process resumes."""
    fake = write_fake_script(project, {"q3": [{"kind": "crash_after_send"}]})
    run_dir = project.run_dir("cli-kill")
    command = [
        sys.executable, str(SCRIPTS_DIR / "run_pilot_live.py"), "run", "--project-root", str(project.root), "--pilot-id", PILOT_ID,
        "--run-dir", str(run_dir), "--fake-script", str(fake), "--safety-ceiling", "10",
    ]
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    first = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert first.returncode == -9 and "simulated crash" in first.stderr
    assert names(run_dir, "q3") == ["dispatch_started"] and not (run_dir / "predictions.jsonl").exists()
    second = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
    assert second.returncode == lr.EXIT_NEEDS_ATTENTION, second.stdout + second.stderr
    assert names(run_dir, "q3") == ["dispatch_started", "outcome_unknown"]
    assert [p for p in by_question(run_dir).values() if p["status"] == "answered"].__len__() == 4


def test_a_price_or_bound_that_cannot_be_computed_refuses_the_run_before_anything_is_written(project: Project) -> None:
    """Contract rule 6: a non-finite price refuses. It is not an internal error and it writes nothing."""
    nan_prices = lr.PriceTable(**{**lr.FAKE_CONFIG.prices.as_dict(), "input_per_million": float("nan")})
    config = lr.ProviderConfig(**{**vars(lr.FAKE_CONFIG), "prices": nan_prices})
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="cannot be computed"):
        go(project, rig, config=config)
    assert rig.factory_calls == 0 and not project.run_dir("run-a").exists()


def test_the_cli_refuses_status_on_a_directory_that_is_not_a_run(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status", "--run-dir", str(project.root)]) == 1
    assert "not an initialised run" in capsys.readouterr().err


def test_the_cli_reports_an_internal_error_with_exit_3(project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(lr, "run_status", lambda run_dir: (_ for _ in ()).throw(KeyError("boom")))
    assert cli.main(["status", "--run-dir", str(project.run_dir("x"))]) == 3
    assert "Traceback" in capsys.readouterr().err


def test_the_cli_resume_reconcile_export_and_score_round_trip(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    """A CLI run with an unknown outcome, reconcile, resume, export and score, all through main()."""
    fake = write_fake_script(project, {"q2": [{"kind": "timeout_unknown"}, {"kind": "answer", "text": "twenty four months"}]})
    assert cli_run(project, "cli-round", fake) == 5
    run_dir = project.run_dir("cli-round")
    attempt_id = by_question(run_dir)["q2"]["attempt_ids"][0]
    assert cli.main(["reconcile", "--run-dir", str(run_dir), attempt_id, "--resolution", "allow_new_attempt", "--note", "no record at the provider"]) == 0
    assert cli_run(project, "cli-round", fake) == 0
    assert by_question(run_dir)["q2"]["answer"] == "twenty four months"
    assert cli.main(["export", "--run-dir", str(run_dir)]) == 0
    assert cli.main(["score", "--project-root", str(project.root), "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "score_summary.json").exists()
    out = capsys.readouterr().out
    assert "reconciled as allow_new_attempt" in out and "scores written" in out


def test_the_cli_refuses_a_bad_fake_script_before_any_directory_exists(project: Project) -> None:
    bad = project.root / "bad.json"
    bad.write_text(json.dumps({"script": {"no-such-question": [{"kind": "answer"}]}}))
    assert cli_run(project, "cli-bad", bad) == 1
    assert not project.run_dir("cli-bad").exists()
    bad.write_text(json.dumps({"script": {"q1": [{"kind": "nonsense"}]}}))
    assert cli_run(project, "cli-bad", bad) == 1
    bad.write_text("{not json")
    assert cli_run(project, "cli-bad", bad) == 1
    assert not project.run_dir("cli-bad").exists()


def test_the_cli_refuses_a_run_dir_under_results_pilots(project: Project) -> None:
    fake = write_fake_script(project)
    assert cli.main(["run", "--project-root", str(project.root), "--pilot-id", PILOT_ID, "--run-dir",
                     str(project.root / "results" / "pilots" / "x"), "--fake-script", str(fake), "--safety-ceiling", "1"]) == 1
    assert cli.main(["dry-run", "--project-root", str(project.root), "--pilot-id", PILOT_ID, "--out",
                     str(project.root / "results" / "pilots" / "y")]) == 1
    assert not (project.pilot_dir / "x").exists() and not (project.root / "results" / "pilots" / "y").exists()


# ---------------------------------------------------------------------------
# Review fixes: interrupts inside the event log (H1) and durability on macOS (L4)
# ---------------------------------------------------------------------------


def _interrupt_fsync_of(monkeypatch: pytest.MonkeyPatch, event_name: str) -> dict[str, int]:
    """Raise KeyboardInterrupt from the fsync of the first append of ``event_name``, after its bytes are written."""
    state = {"armed": 0, "fired": 0}
    real_fsync = os.fsync
    real_append = lr.EventLog.append

    def append(self: Any, event: str, **fields: Any) -> Any:
        if event == event_name and not state["fired"]:
            state["armed"] = 1
        return real_append(self, event, **fields)

    def fsync(fd: int) -> None:
        real_fsync(fd)
        if state["armed"]:
            state["armed"], state["fired"] = 0, 1
            raise KeyboardInterrupt

    monkeypatch.setattr(lr.EventLog, "append", append)
    monkeypatch.setattr(os, "fsync", fsync)
    return state


def _assert_log_is_sound(run_dir: Path) -> None:
    seqs = [json.loads(line)["seq"] for line in (run_dir / "attempts.jsonl").read_text().splitlines()]
    assert seqs == list(range(1, len(seqs) + 1)), "seq numbers are unique and contiguous"


def test_ctrl_c_in_the_fsync_of_dispatch_started_leaves_a_loadable_log_and_resends_nothing(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """H1: the line is in the file when the interrupt lands. The in-memory sequence has moved on, the handler appends cleanly."""
    state = _interrupt_fsync_of(monkeypatch, "dispatch_started")
    rig = Rig()
    result = go(project, rig)
    monkeypatch.undo()
    run_dir = project.run_dir("run-a")
    assert state["fired"] == 1
    _assert_log_is_sound(run_dir)
    assert result.exit_code == lr.EXIT_NEEDS_ATTENTION and events_of(run_dir)[-1]["reason"] == "interrupted"
    assert rig.sent(run_dir) == [], "the interrupt came before the send"
    assert names(run_dir, "q1") == ["dispatch_started", "outcome_unknown"]
    status = lr.run_status(run_dir)
    assert status["requests"]["unknown"] == 1
    assert lr.export_run(run_dir).exit_code == lr.EXIT_NEEDS_ATTENTION
    resumed = Rig()
    assert go(project, resumed).exit_code == lr.EXIT_NEEDS_ATTENTION
    assert "q1" not in resumed.sent(run_dir), "the open attempt waits for reconcile and is never resent"
    _assert_log_is_sound(run_dir)


def test_ctrl_c_in_the_fsync_of_response_saved_keeps_the_paid_response(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """H1: the response file and the response_saved line are on disk. The answer survives and nothing is resent."""
    state = _interrupt_fsync_of(monkeypatch, "response_saved")
    rig = Rig()
    result = go(project, rig)
    monkeypatch.undo()
    run_dir = project.run_dir("run-a")
    assert state["fired"] == 1 and rig.sent(run_dir) == ["q1"]
    _assert_log_is_sound(run_dir)
    assert result.exit_code == lr.EXIT_NEEDS_ATTENTION and events_of(run_dir)[-1]["reason"] == "interrupted"
    assert by_question(run_dir)["q1"]["status"] == "answered", "the saved response is exported"
    assert lr.run_status(run_dir)["requests"]["answered"] == 1
    resumed = Rig()
    assert go(project, resumed).exit_code == 0
    assert "q1" not in resumed.sent(run_dir) and resumed.sent(run_dir) == ["q2", "q3", "q4", "q6"]
    _assert_log_is_sound(run_dir)


def test_an_exception_from_the_append_callback_does_not_skip_the_sync_or_repeat_a_seq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H1: on_append raising after the write still syncs the line, and the next append gets the next seq."""
    seen: list[int] = []
    synced: list[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])

    def flaky(event: dict[str, Any]) -> None:
        seen.append(event["seq"])
        if event["seq"] == 1:
            raise RuntimeError("callback bug")

    log = lr.EventLog(tmp_path / "attempts.jsonl", "inv", 1, flaky)
    synced.clear()
    try:
        with pytest.raises(RuntimeError, match="callback bug"):
            log.append("invocation_ended", reason="completed", ledger={})
        assert len(synced) == 1, "the failing callback did not skip the sync"
        log.append("invocation_ended", reason="completed", ledger={})
    finally:
        log.close()
    assert seen == [1, 2]
    assert [json.loads(line)["seq"] for line in (tmp_path / "attempts.jsonl").read_text().splitlines()] == [1, 2]


def test_a_write_cut_short_by_ctrl_c_leaves_no_torn_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """H1: half a line written and then an interrupt is rolled back, so the next append starts on a clean line."""
    path = tmp_path / "attempts.jsonl"
    log = lr.EventLog(path, "inv", 1, lambda event: None)
    log.append("invocation_ended", reason="completed", ledger={})
    real_write = os.write
    calls = {"n": 0}

    def short_write(fd: int, data: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return real_write(fd, bytes(data)[: len(data) // 2])
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(KeyboardInterrupt):
        log.append("invocation_ended", reason="completed", ledger={})
    monkeypatch.undo()
    log.append("invocation_ended", reason="completed", ledger={})
    log.close()
    assert [json.loads(line)["seq"] for line in path.read_text().splitlines()] == [1, 2]
    assert lr.read_event_file(path).uncommitted_tail == b""


def test_a_full_fsync_is_requested_on_macos_after_fsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L4: on darwin each append calls F_FULLFSYNC after os.fsync, and a filesystem that refuses it does not break the log."""
    import fcntl

    order: list[str] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    refuse = {"on": False}

    def fake_fcntl(fd: int, command: int, *args: Any) -> int:
        assert command == 51
        order.append("fullfsync")
        if refuse["on"]:
            raise OSError("not supported on this filesystem")
        return 0

    monkeypatch.setattr(fcntl, "fcntl", fake_fcntl)
    log = lr.EventLog(tmp_path / "attempts.jsonl", "inv", 1, lambda event: None)
    order.clear()
    log.append("invocation_ended", reason="completed", ledger={})
    assert order == ["fsync", "fullfsync"]
    refuse["on"] = True
    order.clear()
    log.append("invocation_ended", reason="completed", ledger={})
    log.close()
    assert order == ["fsync", "fullfsync"], "the refused call falls back to the plain fsync that already ran"
    assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 2


def test_no_full_sync_is_attempted_off_macos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L4: on other platforms only os.fsync runs."""
    import fcntl

    calls: list[int] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, command, *args: calls.append(command) or 0)
    log = lr.EventLog(tmp_path / "attempts.jsonl", "inv", 1, lambda event: None)
    log.append("invocation_ended", reason="completed", ledger={})
    log.close()
    assert calls == []


def test_atomic_writes_use_the_same_full_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """L4: the response, request and export files go through the sync helper that adds F_FULLFSYNC on macOS."""
    import fcntl

    calls: list[int] = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, command, *args: calls.append(command) or 0)
    lr.atomic_write_text(tmp_path / "out.json", "{}\n")
    assert (tmp_path / "out.json").read_text() == "{}\n"
    assert len(calls) >= 2, "the file and its directory are both fully synced"


# ---------------------------------------------------------------------------
# Review fixes: the endpoint (H3) and the parameter allowlist (L2)
# ---------------------------------------------------------------------------

LIVE_READY_ENV = {lr.LIVE_ENV_NAME: lr.LIVE_ENV_VALUE, "OPENAI_API_KEY": "test-not-a-key"}
CLIENT_ENV_VARS = ("OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_ORGANIZATION")


def live_config(**changes: Any) -> lr.ProviderConfig:
    return lr.parse_provider_config({**valid_live_config(), **changes})


def _completion_json(model: str = "gpt-4o-2024-11-20") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "twelve months"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


def test_the_live_client_is_built_on_the_configured_base_url_with_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """H3: base_url comes from the config endpoint. A mock transport sees the request; no socket opens."""
    import httpx

    for name in CLIENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError("no network in tests")))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=_completion_json())

    config = live_config(endpoint="https://gateway.example.test/v1")
    provider = lr.build_live_provider(config, {"OPENAI_API_KEY": "test-not-a-key"}, http_client=httpx.Client(transport=httpx.MockTransport(handler), trust_env=False))
    assert str(provider._client.base_url) == "https://gateway.example.test/v1/"
    assert provider.identity()["endpoint"] == "https://gateway.example.test/v1/"
    response = provider.send(
        lr.ProviderRequest("r", "r-a1", ({"role": "user", "content": "hi"},), dict(config.params), 128, 60.0)
    )
    assert response.text == "twelve months" and seen == ["https://gateway.example.test/v1/chat/completions"]


def test_the_live_client_built_for_the_cli_does_not_follow_redirects_and_keeps_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2: the SDK's own client follows redirects, so build_live_provider passes its own. No request is sent."""
    for name in CLIENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError("no network in tests")))
    config = live_config(timeout_seconds=42)
    provider = lr.build_live_provider(config, {"OPENAI_API_KEY": "test-not-a-key"})
    assert provider._client._client.follow_redirects is False
    assert provider._client.timeout == 42 and provider._client.max_retries == 0
    assert str(provider._client.base_url) == "https://api.openai.com/v1/"


@pytest.mark.parametrize("injected", ["follows", "unknown"])
def test_an_injected_http_client_that_follows_redirects_is_refused(monkeypatch: pytest.MonkeyPatch, injected: str) -> None:
    """M2: a test seam must not reopen the redirect hole. A client whose setting cannot be read is refused too."""
    import httpx

    built = tripwire_clients_except_builder(monkeypatch)
    client: Any = httpx.Client(follow_redirects=True) if injected == "follows" else object()
    with pytest.raises(RunnerRefusal, match="redirect"):
        lr.build_live_provider(live_config(), {"OPENAI_API_KEY": "test-not-a-key"}, http_client=client)
    assert built == []


def test_the_live_provider_descriptor_records_the_base_url_and_the_endpoint_is_identity() -> None:
    """H3: the run identity names the URL the client will use, so a different endpoint is a different run."""
    a = lr.provider_descriptor(MODE_LIVE_VALUE, live_config())
    b = lr.provider_descriptor(MODE_LIVE_VALUE, live_config(endpoint="https://gateway.example.test/v1/"))
    assert a["base_url"] == "https://api.openai.com/v1/" and b["base_url"] == "https://gateway.example.test/v1/"
    assert lr.provider_descriptor(MODE_FAKE) == {"adapter": "faar.answer_providers.FakeProvider"}
    options_a = lr.RunOptions(project_root=Path("."), run_dir=Path("x"), mode=MODE_LIVE_VALUE, config=live_config(), safety_ceiling=1.0)
    _, wired = lr.wire_provider(options_a, fake_script=None, environ={})
    assert wired["base_url"] == "https://api.openai.com/v1/"


MODE_LIVE_VALUE = "live"


@pytest.mark.parametrize("name", CLIENT_ENV_VARS)
def test_live_mode_refuses_when_the_environment_can_redirect_the_client(project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str) -> None:
    """H3: OPENAI_BASE_URL, OPENAI_ORG_ID, OPENAI_PROJECT_ID and OPENAI_ORGANIZATION change where or as whom the key is used."""
    config = project.root / "provider.json"
    config.write_text(json.dumps(valid_live_config()))
    built = tripwire_clients(monkeypatch)
    monkeypatch.setattr(os, "environ", RecordingEnv({**LIVE_READY_ENV, name: "https://elsewhere.example.test/v1"}))
    run_dir = project.root / "results" / "development" / "live-a"
    argv = [*LIVE_ARGS, "--project-root", str(project.root), "--run-dir", str(run_dir), "--provider-config", str(config), "--safety-ceiling", "1"]
    assert cli.main(argv) == lr.EXIT_REFUSED
    assert name in capsys.readouterr().err
    assert built == [] and not run_dir.exists()
    assert "OPENAI_API_KEY" not in os.environ.reads


@pytest.mark.parametrize("name", CLIENT_ENV_VARS)
def test_build_live_provider_refuses_them_too_without_building_a_client(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """H3: the builder repeats the check on the mapping it was given and on the real environment the SDK reads."""
    built = tripwire_clients_except_builder(monkeypatch)
    with pytest.raises(RunnerRefusal, match=name):
        lr.build_live_provider(live_config(), {"OPENAI_API_KEY": "test-not-a-key", name: "x"})
    monkeypatch.setenv(name, "x")
    with pytest.raises(RunnerRefusal, match=name):
        lr.build_live_provider(live_config(), {"OPENAI_API_KEY": "test-not-a-key"})
    assert built == []


def tripwire_clients_except_builder(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    built: list[str] = []
    import openai

    from faar import answer_providers

    def boom(*args: Any, **kwargs: Any) -> Any:
        built.append("client")
        raise AssertionError("no client may be built")

    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.setattr(answer_providers, "OpenAIChatProvider", boom)
    for name in CLIENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return built


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1/chat/completions",
        "api.openai.com/v1",
        "ftp://api.openai.com/v1",
        "http://api.openai.com/v1",
        "https://user:pass@api.openai.com/v1",
        "https://api.openai.com/v1?x=1",
        "https://",
        "",
    ],
)
def test_the_endpoint_must_be_an_https_api_base_url(endpoint: str) -> None:
    """H3: the config field is the API base URL, not the request path. Plain http is refused off the loopback."""
    with pytest.raises(RunnerRefusal, match="endpoint"):
        live_config(endpoint=endpoint)


def test_a_loopback_http_base_url_and_a_trailing_slash_are_accepted() -> None:
    assert live_config(endpoint="http://127.0.0.1:8080/v1").endpoint == "http://127.0.0.1:8080/v1"
    assert lr.provider_descriptor(MODE_LIVE_VALUE, live_config(endpoint="https://api.openai.com/v1/"))["base_url"] == "https://api.openai.com/v1/"


def test_the_provisional_dry_run_endpoint_is_a_base_url() -> None:
    assert lr.PROVISIONAL_DRY_RUN_CONFIG.endpoint == "https://api.openai.com/v1"


@pytest.mark.parametrize(
    "params",
    [{"service_tier": "flex"}, {"tools": []}, {"logprobs": True}, {"response_format": {"type": "json_object"}}, {"reasoning_effort": "high"}, {"n": 2}, {"user": "x"}, {"temperature": 0, "stream": True}],
)
def test_the_provider_config_refuses_request_parameters_outside_the_allowlist(params: dict[str, Any]) -> None:
    """L2: an unknown parameter can change billing outside the cost bound, so only the shared allowlist passes."""
    with pytest.raises(RunnerRefusal, match="params"):
        live_config(params=params)


def test_every_allowlisted_parameter_is_accepted() -> None:
    from faar.live_contract import ALLOWED_OPENAI_PARAMS

    values = {"temperature": 0, "top_p": 1, "seed": 7, "stop": ["END"], "presence_penalty": 0, "frequency_penalty": 0}
    assert set(values) == set(ALLOWED_OPENAI_PARAMS)
    assert dict(live_config(params=values).params) == values


def test_the_help_states_the_endpoint_rule_the_refused_variables_and_the_allowed_parameters(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    for word in (*CLIENT_ENV_VARS, "API base URL", "temperature", "frequency_penalty"):
        assert word in text


# ---------------------------------------------------------------------------
# Review fixes: scoring is final (M1) and a mistyped path is left alone (M2)
# ---------------------------------------------------------------------------


def scored_run(project: Project, name: str = "run-a", steps: dict[str, list[FakeStep]] | None = None) -> Path:
    go(project, Rig(steps=steps or {}), name)
    run_dir = project.run_dir(name)
    lr.score_run_live(project_root=project.root, run_dir=run_dir)
    return run_dir


def test_score_refuses_a_run_that_is_not_complete_and_writes_nothing(project: Project) -> None:
    """M1: budget_limited, needs_reconciliation, stopped and incomplete runs cannot be scored."""
    costs = bounds(project)
    cases = {
        "s-budget": dict(safety_ceiling=costs["q1"] + 0.000001),
        "s-unknown": dict(rig=Rig(steps={"q2": [FakeStep("timeout_unknown")]})),
        "s-stopped": dict(rig=Rig(steps={"q2": [FakeStep("auth_error")]})),
        "s-incomplete": dict(crash_hook=crash_at(lr.CRASH_BEFORE_DISPATCH, "q3")),
    }
    for name, kwargs in cases.items():
        rig = kwargs.pop("rig", None) or Rig()
        try:
            go(project, rig, name, **kwargs)
        except SimulatedCrash:
            pass
        run_dir = project.run_dir(name)
        before = file_snapshot(run_dir)
        with pytest.raises(RunnerRefusal, match="complete"):
            lr.score_run_live(project_root=project.root, run_dir=run_dir)
        assert strip_lock(file_snapshot(run_dir)) == strip_lock(before), name
        assert not (run_dir / "scores.jsonl").exists() and not (run_dir / "score_summary.json").exists()


def test_reconcile_and_export_refuse_after_scoring_and_change_nothing(project: Project) -> None:
    """M1: scores describe one set of predictions, so no writer may change the records after `score`."""
    rig = Rig(steps={"q1": [FakeStep("timeout_unknown")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    attempt_id = by_question(run_dir)["q1"]["attempt_ids"][0]
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=attempt_id, resolution=lr.RESOLUTION_FAIL, note="checked, not billed")
    lr.score_run_live(project_root=project.root, run_dir=run_dir)
    before = file_snapshot(run_dir)
    with pytest.raises(RunnerRefusal, match="scored"):
        lr.reconcile_attempt(run_dir=run_dir, attempt_id=attempt_id, resolution=lr.RESOLUTION_ALLOW, note="changed my mind")
    assert strip_lock(file_snapshot(run_dir)) == strip_lock(before)


def test_export_after_scoring_is_a_no_op_when_it_would_write_identical_files_and_refuses_otherwise(project: Project) -> None:
    run_dir = scored_run(project)
    before = file_snapshot(run_dir)
    result = lr.export_run(run_dir)
    assert "unchanged" in result.message and strip_lock(file_snapshot(run_dir)) == strip_lock(before)
    predictions = run_dir / "predictions.jsonl"
    predictions.write_text(predictions.read_text() + "\n")
    tampered = file_snapshot(run_dir)
    with pytest.raises(RunnerRefusal, match="scored"):
        lr.export_run(run_dir)
    assert strip_lock(file_snapshot(run_dir)) == strip_lock(tampered)


def test_score_after_scoring_rewrites_no_export_file(project: Project) -> None:
    run_dir = scored_run(project)
    (run_dir / "predictions.jsonl").unlink()
    with pytest.raises(RunnerRefusal, match="scored"):
        lr.score_run_live(project_root=project.root, run_dir=run_dir)
    assert not (run_dir / "predictions.jsonl").exists(), "score did not rebuild a deleted export"


def test_status_reports_a_scored_run_as_final(project: Project) -> None:
    run_dir = scored_run(project)
    status = lr.run_status(run_dir)
    assert status["scored"] is True and "scored" in lr.format_status(status)


def test_the_score_help_states_the_complete_run_rule(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["score", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "complete" in text and "final" in text


def _frozen_like(project: Project, sub: str, *, with_config: bool) -> Path:
    directory = project.root.joinpath(*sub.split("/"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "keep.txt").write_text("frozen")
    if with_config:
        (directory / "run_config.json").write_text("{}")
    return directory


@pytest.mark.parametrize("sub", ["results/pilots/x", "results/Pilots/x", "results/PILOTS/x/y"])
@pytest.mark.parametrize("with_config", [False, True])
def test_export_reconcile_score_and_reopen_leave_a_frozen_like_directory_untouched(project: Project, sub: str, with_config: bool) -> None:
    """M2: the directory rules and run_config.json are checked before run.lock is created or opened."""
    directory = _frozen_like(project, sub, with_config=with_config)
    before = file_snapshot(directory)
    entries = sorted(p.name for p in directory.iterdir())
    for action in (
        lambda: lr.export_run(directory),
        lambda: lr.reconcile_attempt(run_dir=directory, attempt_id="a", resolution=lr.RESOLUTION_FAIL, note="n"),
        lambda: lr.score_run_live(project_root=project.root, run_dir=directory),
        lambda: lr.reopen_question(run_dir=directory, question_id="q1", note="n"),
    ):
        with pytest.raises(RunnerRefusal):
            action()
    assert file_snapshot(directory) == before and sorted(p.name for p in directory.iterdir()) == entries
    assert not (directory / "run.lock").exists()


def test_a_mistyped_directory_without_run_config_gets_no_lock_file(project: Project, tmp_path: Path) -> None:
    """M2: an existing directory that is not a run, and a path that does not exist, are left as they are."""
    typo = project.root / "results" / "engineering" / "run-typo"
    typo.mkdir(parents=True)
    (typo / "notes.txt").write_text("mine")
    missing = project.root / "results" / "engineering" / "not-there"
    for target in (typo, missing, project.root):
        before = file_snapshot(target) if target.exists() else None
        for action in (
            lambda: lr.export_run(target),
            lambda: lr.reconcile_attempt(run_dir=target, attempt_id="a", resolution=lr.RESOLUTION_FAIL, note="n"),
            lambda: lr.score_run_live(project_root=project.root, run_dir=target),
            lambda: lr.reopen_question(run_dir=target, question_id="q1", note="n"),
            lambda: lr.run_status(target),
        ):
            with pytest.raises(RunnerRefusal):
                action()
        assert not (target / "run.lock").exists()
        assert (file_snapshot(target) if target.exists() else None) == before
    assert not missing.exists()


def test_the_cli_leaves_a_mistyped_path_untouched(project: Project) -> None:
    typo = project.root / "results" / "engineering" / "run-typo"
    typo.mkdir(parents=True)
    for argv in (["export", "--run-dir", str(typo)], ["score", "--project-root", str(project.root), "--run-dir", str(typo)],
                 ["reconcile", "--run-dir", str(typo), "a", "--resolution", "mark_failed", "--note", "n"],
                 ["reopen", "--run-dir", str(typo), "q1", "--note", "n"]):
        assert cli.main(argv) == lr.EXIT_REFUSED
    assert list(typo.iterdir()) == []


# ---------------------------------------------------------------------------
# Review fixes: the attempt limit across invocations (L3) and reopening a failed question (M4)
# ---------------------------------------------------------------------------


def test_repeated_stop_run_errors_exhaust_the_attempt_limit_and_fail_the_question(project: Project) -> None:
    """L3: three invocations that each end in an auth error use up q1's three attempts. A fourth invocation must not try q1 again."""
    steps = {"q1": [FakeStep("auth_error")] * 3 + [FakeStep("answer", text="fourth attempt")]}
    run_dir = project.run_dir("run-stop")
    rigs = []
    for expected_attempts in (1, 2, 3):
        rigs.append(Rig(steps))
        result = go(project, rigs[-1], "run-stop")
        assert result.summary["run_state"] == "stopped"
        assert len(by_question(run_dir)["q1"]["attempt_ids"]) == expected_attempts
    record = by_question(run_dir)["q1"]
    assert record["status"] == "execution_failed" and record["failure"]["type"] == "auth" and record["unserved"] is False
    assert record["failure"]["attempts_exhausted"] is True
    last_failed = [e for e in events_of(run_dir) if e["event"] == "attempt_failed"][-1]
    assert last_failed["decision"] == "stop_run" and last_failed["attempts_exhausted"] is True
    fourth = Rig(steps)
    result = go(project, fourth, "run-stop")
    assert "q1" not in fourth.sent(run_dir), "q1 gets no fourth attempt"
    assert len(by_question(run_dir)["q1"]["attempt_ids"]) == 3 and by_question(run_dir)["q1"]["status"] == "execution_failed"
    assert result.summary["run_state"] == "complete" and result.exit_code == lr.EXIT_EXECUTION_FAILED
    assert [q for q, p in by_question(run_dir).items() if p["status"] == "answered"] == ["q2", "q3", "q4", "q6"]


SIX_FAILURES_THEN_AN_ANSWER = {"q1": [FakeStep("connect_error")] * 6 + [FakeStep("answer", text="twelve months")]}


def _fail_q1(project: Project, kind: str = "connect_error", count: int = 3, name: str = "run-a") -> tuple[Rig, Path]:
    rig = Rig(steps={"q1": [FakeStep(kind)] * count})
    go(project, rig, name)
    return rig, project.run_dir(name)


def test_a_question_failed_after_three_connect_errors_is_reopened_and_answered_on_the_next_run(project: Project) -> None:
    rig, run_dir = _fail_q1(project)
    assert by_question(run_dir)["q1"]["status"] == "execution_failed" and len(by_question(run_dir)["q1"]["attempt_ids"]) == 3
    result = lr.reopen_question(run_dir=run_dir, question_id="q1", note="network fixed; checked with curl")
    assert result.exit_code == lr.EXIT_OK and result.summary["run_state"] == "incomplete"
    reopened = [e for e in events_of(run_dir) if e["event"] == "question_reopened"]
    assert len(reopened) == 1 and reopened[0]["question_id"] == "q1" and reopened[0]["note"].startswith("network fixed")
    assert reopened[0]["attempts_before"] == 3
    assert by_question(run_dir)["q1"]["status"] == "execution_failed" and by_question(run_dir)["q1"]["unserved"] is True
    resumed = go(project, rig)
    assert resumed.exit_code == 0 and rig.sent(run_dir).count("q1") == 4
    record = by_question(run_dir)["q1"]
    assert record["status"] == "answered" and len(record["attempt_ids"]) == 4 and record["reopen_count"] == 1


def test_a_reopened_question_keeps_its_earlier_attempts_and_their_costs(project: Project) -> None:
    rig, run_dir = _fail_q1(project, "retryable_error")
    before = summary_of(run_dir)["cost"]
    assert before["reserved"] > 0
    lr.reopen_question(run_dir=run_dir, question_id="q1", note="rate limit window passed")
    go(project, rig)
    after = summary_of(run_dir)["cost"]
    assert after["reserved"] == pytest.approx(before["reserved"]), "the three rejected attempts stay reserved"
    assert after["measured"] > before["measured"]
    assert summary_of(run_dir)["attempts"]["failed"] == 3 and summary_of(run_dir)["attempts"]["saved"] == 5


def test_the_attempt_count_resets_only_after_a_reopen(project: Project) -> None:
    """L3 and M4: without a reopen a failed question gets nothing. After one it gets max_attempts more, counted from the event."""
    rig = Rig(steps=SIX_FAILURES_THEN_AN_ANSWER)
    run_dir = project.run_dir("run-a")
    go(project, rig)
    assert rig.sent(run_dir).count("q1") == 3 and by_question(run_dir)["q1"]["status"] == "execution_failed"
    go(project, rig)
    assert rig.sent(run_dir).count("q1") == 3, "a failed question is not retried on resume"
    lr.reopen_question(run_dir=run_dir, question_id="q1", note="first reopen")
    result = go(project, rig)
    assert rig.sent(run_dir).count("q1") == 6, "exactly max_attempts further attempts, not 2 and not 4"
    assert len(by_question(run_dir)["q1"]["attempt_ids"]) == 6 and result.exit_code == lr.EXIT_EXECUTION_FAILED
    go(project, rig)
    assert rig.sent(run_dir).count("q1") == 6
    lr.reopen_question(run_dir=run_dir, question_id="q1", note="second reopen")
    assert go(project, rig).exit_code == 0
    record = by_question(run_dir)["q1"]
    assert rig.sent(run_dir).count("q1") == 7 and len(record["attempt_ids"]) == 7
    assert record["status"] == "answered" and record["reopen_count"] == 2
    reopened = [e for e in events_of(run_dir) if e["event"] == "question_reopened"]
    assert [e["attempts_before"] for e in reopened] == [3, 6]


def test_a_question_marked_failed_by_reconcile_can_be_reopened(project: Project) -> None:
    rig = Rig(steps={"q2": [FakeStep("timeout_unknown")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    lr.reconcile_attempt(run_dir=run_dir, attempt_id=by_question(run_dir)["q2"]["attempt_ids"][0], resolution="mark_failed", note="checked")
    lr.reopen_question(run_dir=run_dir, question_id="q2", note="provider confirmed nothing was billed")
    assert go(project, rig).exit_code == 0
    assert by_question(run_dir)["q2"]["status"] == "answered" and len(by_question(run_dir)["q2"]["attempt_ids"]) == 2


def test_reopen_refuses_every_question_that_is_not_execution_failed(project: Project) -> None:
    """M4: answered, unserved (budget), unknown, never-sent and unknown ids are all refused, with the right next step."""
    costs = bounds(project)
    go(project, Rig(steps={"q2": [FakeStep("timeout_unknown")]}), "unknown-run")
    unknown_dir = project.run_dir("unknown-run")
    go(project, Rig(), "budget-run", safety_ceiling=costs["q1"] + 0.000001)
    budget_dir = project.run_dir("budget-run")
    cases = [
        (unknown_dir, "q1", "answered"),
        (unknown_dir, "q2", "reconcile"),
        (unknown_dir, "q5", "never sent"),
        (unknown_dir, "no-such-question", "no question"),
        (budget_dir, "q2", "Run again"),
    ]
    for run_dir, question_id, pattern in cases:
        before = strip_lock(file_snapshot(run_dir))
        with pytest.raises(RunnerRefusal, match=pattern):
            lr.reopen_question(run_dir=run_dir, question_id=question_id, note="please")
        assert strip_lock(file_snapshot(run_dir)) == before, (question_id, pattern)
        assert not [e for e in events_of(run_dir) if e["event"] == "question_reopened"]


def test_reopen_needs_a_note_and_an_unheld_lock_and_an_unscored_run(project: Project) -> None:
    rig, run_dir = _fail_q1(project)
    before = strip_lock(file_snapshot(run_dir))
    for note in ("", "   "):
        with pytest.raises(RunnerRefusal, match="note"):
            lr.reopen_question(run_dir=run_dir, question_id="q1", note=note)
    holder = lr.RunLock(run_dir, "holder")
    holder.acquire()
    try:
        with pytest.raises(RunnerRefusal, match="another invocation holds"):
            lr.reopen_question(run_dir=run_dir, question_id="q1", note="n")
    finally:
        holder.release()
    assert strip_lock(file_snapshot(run_dir)) == before
    lr.score_run_live(project_root=project.root, run_dir=run_dir)
    with pytest.raises(RunnerRefusal, match="scored"):
        lr.reopen_question(run_dir=run_dir, question_id="q1", note="too late")
    assert not [e for e in events_of(run_dir) if e["event"] == "question_reopened"]


def test_reopen_resolves_an_orphaned_dispatch_before_it_looks_at_the_question(project: Project) -> None:
    """A dead invocation left q3 open. Reopening another question first records q3 as unknown, as reconcile does."""
    rig = Rig(steps={"q1": [FakeStep("connect_error")] * 3})
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_AFTER_DISPATCH, "q3"))
    run_dir = project.run_dir("run-a")
    lr.reopen_question(run_dir=run_dir, question_id="q1", note="n")
    assert names(run_dir, "q3")[-1] == "outcome_unknown"


def test_the_cli_reopen_command(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    fake = write_fake_script(project, {"q1": [{"kind": "connect_error"}] * 3})
    assert cli_run(project, "cli-reopen", fake) == 2
    run_dir = project.run_dir("cli-reopen")
    assert cli.main(["reopen", "--run-dir", str(run_dir), "q1", "--note", "network fixed"]) == 0
    assert "reopened" in capsys.readouterr().out
    assert cli.main(["reopen", "--run-dir", str(run_dir), "q1", "--note", "again"]) == 1, "now pending, not failed"
    assert cli_run(project, "cli-reopen", fake) == 0
    with pytest.raises(SystemExit):
        cli.main(["reopen", "--run-dir", str(run_dir), "q1"])
    assert by_question(run_dir)["q1"]["status"] == "answered"


def test_the_ledger_ignores_reopen_events(project: Project) -> None:
    rig, run_dir = _fail_q1(project)
    lr.reopen_question(run_dir=run_dir, question_id="q1", note="n")
    go(project, rig)
    assert lr.run_status(run_dir)["cost"]["committed_upper"] > 0


# ---------------------------------------------------------------------------
# Review fixes: the circuit breaker (M5)
# ---------------------------------------------------------------------------


def _ended(run_dir: Path) -> dict[str, Any]:
    return [e for e in events_of(run_dir) if e["event"] == "invocation_ended"][-1]


def test_three_consecutive_unknown_outcomes_stop_the_invocation(project: Project) -> None:
    """M5: a systematic fault must not run through the whole pilot. Three unknown outcomes in a row stop the run."""
    steps = {q: [FakeStep("timeout_unknown")] for q in ("q1", "q2", "q3")}
    rig = Rig(steps)
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2", "q3"], "no fourth request"
    ended = _ended(run_dir)
    assert ended["reason"] == "circuit_breaker" and ended["circuit_breaker"] == {"consecutive_failures": 3, "threshold": 3}
    assert result.summary["run_state"] == "needs_reconciliation" and result.exit_code == lr.EXIT_NEEDS_ATTENTION
    assert by_question(run_dir)["q4"]["unserved"] is True and by_question(run_dir)["q6"]["unserved"] is True


def test_three_consecutive_rejected_failures_stop_the_invocation_and_the_run_is_stopped(project: Project) -> None:
    """M5: `rejected` with decision fail_question counts. Only pending questions remain, so the run state is `stopped`."""
    steps = {q: [FakeStep("non_retryable_error")] for q in ("q1", "q2", "q3")}
    rig = Rig(steps)
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2", "q3"]
    assert _ended(run_dir)["reason"] == "circuit_breaker"
    assert result.summary["run_state"] == "stopped" and result.exit_code == lr.EXIT_NEEDS_ATTENTION
    assert [by_question(run_dir)[q]["status"] for q in ("q1", "q2", "q3")] == ["execution_failed"] * 3
    resumed = go(project, rig)
    assert rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"], "a new invocation starts its count at zero"
    assert resumed.summary["run_state"] == "complete"


def test_three_consecutive_exceptions_that_are_not_provider_errors_stop_the_invocation(project: Project) -> None:
    class Broken:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, request: Any) -> Any:
            self.calls += 1
            raise RuntimeError("adapter bug")

    broken = Broken()
    result = lr.execute_run(options(project), provider_factory=lambda ctx: broken, descriptor={"adapter": "broken"}, sleep=lambda s: None, environ={})
    run_dir = project.run_dir("run-a")
    assert broken.calls == 3 and _ended(run_dir)["reason"] == "circuit_breaker"
    assert result.exit_code == lr.EXIT_NEEDS_ATTENTION


def test_a_successful_response_resets_the_count(project: Project) -> None:
    steps = {"q1": [FakeStep("timeout_unknown")], "q2": [FakeStep("timeout_unknown")], "q4": [FakeStep("timeout_unknown")], "q6": [FakeStep("timeout_unknown")]}
    rig = Rig(steps)
    go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == SENT and _ended(run_dir)["reason"] == "needs_reconciliation"
    assert _ended(run_dir)["circuit_breaker"] == {"consecutive_failures": 2, "threshold": 3}


def test_only_a_saved_response_resets_the_count(project: Project) -> None:
    """M5, N5: a saved answer resets the count; a question that exhausted its unsent retries still counts."""
    steps = {
        "q1": [FakeStep("timeout_unknown")],
        "q2": [FakeStep("timeout_unknown")],
        "q3": [FakeStep("answer", text="twelve months")],
        "q4": [FakeStep("timeout_unknown")],
        "q6": [FakeStep("timeout_unknown")],
    }
    rig = Rig(steps)
    go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2", "q3", "q4", "q6"]
    assert _ended(run_dir)["reason"] == "needs_reconciliation"
    tripped = Rig({"q1": [FakeStep("timeout_unknown")], "q2": [FakeStep("timeout_unknown")], "q3": [FakeStep("connect_error")] * 3})
    go(project, tripped, name="run-b")
    run_b = project.run_dir("run-b")
    assert tripped.sent(run_b) == ["q1", "q2", "q3", "q3", "q3"]
    assert _ended(run_b)["reason"] == "circuit_breaker"


def test_the_threshold_is_part_of_the_run_identity(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig()
    go(project, rig)
    run_dir = project.run_dir("run-a")
    stored = json.loads((run_dir / "run_config.json").read_text())["identity"]["retry_policy"]
    assert stored["circuit_breaker"]["consecutive_failures"] == 3 and stored["max_attempts"] == 3
    monkeypatch.setattr(lr, "CIRCUIT_BREAKER_THRESHOLD", 2)
    with pytest.raises(RunnerRefusal, match="circuit_breaker"):
        go(project, Rig())


def test_every_invocation_ended_records_the_counter_and_the_help_names_the_rule(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    go(project, Rig())
    assert _ended(project.run_dir("run-a"))["circuit_breaker"] == {"consecutive_failures": 0, "threshold": 3}
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    assert "circuit_breaker" in " ".join(capsys.readouterr().out.split())


def test_status_says_what_to_do_after_the_breaker_tripped(project: Project) -> None:
    go(project, Rig({q: [FakeStep("non_retryable_error")] for q in ("q1", "q2", "q3")}))
    status = lr.run_status(project.run_dir("run-a"))
    assert status["last_invocation_ended"] == "circuit_breaker"
    assert any("circuit breaker" in step for step in status["next"])


# ---------------------------------------------------------------------------
# Review fixes: the validity flag (M6) and status while a run is active (L6)
# ---------------------------------------------------------------------------


def live_run(project: Project, rig: Rig, name: str = "live-a", *, services: lr.Services | None = None, monkeypatch: pytest.MonkeyPatch | None = None, run_kind: str = "development_pilot") -> lr.LiveResult:
    """A live-mode run driven by a fake provider through the factory seam. No client is built and nothing is sent."""
    if monkeypatch is not None:
        for env_name in CLIENT_ENV_VARS:
            monkeypatch.delenv(env_name, raising=False)
    config = lr.parse_provider_config({**valid_live_config(), "model": "gpt-4o-2024-11-20"})
    opts = lr.RunOptions(
        project_root=project.root,
        run_dir=project.root / "results" / "development" / name,
        mode=lr.MODE_LIVE,
        pilot_id=PILOT_ID,
        safety_ceiling=100.0,
        config=config,
        run_kind=run_kind,
    )
    factory = lambda ctx: lr.build_fake_provider(rig.steps, rig.default, config, ctx)  # noqa: E731
    return lr.execute_run(
        opts, services=services, provider_factory=factory, descriptor=rig.descriptor, sleep=lambda s: None,
        environ={lr.LIVE_ENV_NAME: lr.LIVE_ENV_VALUE},
    )


def test_a_clean_live_run_is_a_valid_baseline_and_the_summary_says_what_that_means(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    result = live_run(project, Rig(), monkeypatch=monkeypatch)
    summary = result.summary
    assert summary["run_state"] == "complete" and summary["valid_baseline"] is True and summary["valid_baseline_blockers"] == []
    assert summary["returned_model_mismatch"] == 0 and summary["input_bound_exceeded"] == 0 and summary["cost"]["anomalies"] == []
    text = " ".join(summary["valid_baseline_means"].split())
    assert "mechanical completeness" in text and "does not mean" in text
    for word in ("prompt", "model", "retrieval", "budget"):
        assert word in text


def test_a_returned_model_that_differs_from_the_requested_model_blocks_the_flag(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """M6: a silent alias move or a routed model is a different measurement. The match rule is exact equality."""
    rig = Rig(steps={"q2": [FakeStep("answer", text="x", returned_model="gpt-4o-2025-01-01")]})
    summary = live_run(project, rig, monkeypatch=monkeypatch).summary
    assert summary["returned_model_mismatch"] == 1 and summary["valid_baseline"] is False
    assert any("returned model" in blocker for blocker in summary["valid_baseline_blockers"])
    # The driver stops after the mismatch (safety stop): q1 and q2 are saved, the rest are unserved.
    assert summary["returned_models"] == {"gpt-4o-2024-11-20": 1, "gpt-4o-2025-01-01": 1}
    assert summary["run_state"] == "safety_stopped"


def test_a_missing_returned_model_counts_as_a_mismatch(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    for name in CLIENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    config = lr.parse_provider_config(valid_live_config())

    class NoModel:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def send(self, request: Any) -> Any:
            return dataclasses.replace(self.inner.send(request), returned_model=None)

    rig = Rig()
    opts = lr.RunOptions(
        project_root=project.root, run_dir=project.root / "results" / "development" / "live-a", mode=lr.MODE_LIVE,
        pilot_id=PILOT_ID, safety_ceiling=100.0, config=config, run_kind="development_pilot",
    )
    result = lr.execute_run(
        opts, provider_factory=lambda ctx: NoModel(lr.build_fake_provider(rig.steps, rig.default, config, ctx)),
        descriptor=rig.descriptor, sleep=lambda s: None, environ={lr.LIVE_ENV_NAME: lr.LIVE_ENV_VALUE},
    )
    # The first response has no returned model, so the driver stops (safety stop) and sends nothing more.
    assert result.summary["returned_model_mismatch"] == 1 and result.summary["valid_baseline"] is False
    assert result.summary["returned_models"] == {"(none)": 1} and result.summary["run_state"] == "safety_stopped"
    assert lr.returned_model_matches("gpt-4o-2024-11-20", "gpt-4o-2024-11-20") is True
    assert lr.returned_model_matches("gpt-4o", "gpt-4o-2024-11-20") is False, "no alias mapping is defined"


def test_an_input_bound_exceedance_blocks_the_flag(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    services = dataclasses.replace(lr.Services.default(), input_token_upper_bound=lambda messages: 1)
    summary = live_run(project, Rig(), services=services, monkeypatch=monkeypatch).summary
    # The first response exceeds the bound, so the driver stops (safety stop) and the other four are never sent.
    assert summary["input_bound_exceeded"] == 1 and summary["valid_baseline"] is False
    assert any("input" in blocker for blocker in summary["valid_baseline_blockers"])
    assert summary["run_state"] == "safety_stopped" and summary["attempts"]["total"] == 1


def test_a_ledger_anomaly_blocks_the_flag(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    real = lr.Services.default()

    def anomalous(events: Any, prices: Any) -> Any:
        return dataclasses.replace(real.ledger_from_events(events, prices), anomalies=("x-a1: measured cost exceeds its upper bound",))

    summary = live_run(project, Rig(), services=dataclasses.replace(real, ledger_from_events=anomalous), monkeypatch=monkeypatch).summary
    assert summary["valid_baseline"] is False and summary["cost"]["anomalies"]
    assert any("anomal" in blocker for blocker in summary["valid_baseline_blockers"])
    # This ledger reports an anomaly from the start, so the driver dispatches nothing (safety stop).
    assert summary["run_state"] == "safety_stopped" and summary["attempts"]["total"] == 0


def test_an_execution_failure_a_fake_run_and_an_incomplete_run_block_the_flag(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = live_run(project, Rig(steps={"q1": [FakeStep("non_retryable_error")]}), "live-failed", monkeypatch=monkeypatch).summary
    assert failed["valid_baseline"] is False and any("execution_failed" in b for b in failed["valid_baseline_blockers"])
    fake = go(project, Rig()).summary
    assert fake["valid_baseline"] is False and any("live" in b for b in fake["valid_baseline_blockers"])
    stopped = live_run(project, Rig(steps={"q2": [FakeStep("auth_error")]}), "live-stopped", monkeypatch=monkeypatch).summary
    assert stopped["valid_baseline"] is False and any("state" in b for b in stopped["valid_baseline_blockers"])


def test_the_score_summary_carries_the_same_flag_and_explanation(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    live_run(project, Rig(), monkeypatch=monkeypatch)
    run_dir = project.root / "results" / "development" / "live-a"
    scored = lr.score_run_live(project_root=project.root, run_dir=run_dir).summary
    assert scored["valid_baseline"] is True and "does not mean" in scored["valid_baseline_means"]


def _hold_while_open(project: Project) -> tuple[Path, lr.RunLock]:
    """A run with one open attempt (its process died after dispatch_started), and a live process holding the lock."""
    with pytest.raises(SimulatedCrash):
        go(project, Rig(), crash_hook=crash_at(lr.CRASH_AFTER_DISPATCH, "q3"))
    run_dir = project.run_dir("run-a")
    holder = lr.RunLock(run_dir, "the-live-invocation")
    holder.acquire()
    return run_dir, holder


def test_status_calls_an_open_attempt_in_flight_while_a_live_process_holds_the_lock(project: Project) -> None:
    """L6: an attempt with no result yet is not unknown while its own process is still running."""
    run_dir, holder = _hold_while_open(project)
    try:
        status = lr.run_status(run_dir)
        text = lr.format_status(status)
    finally:
        holder.release()
    assert status["run_state"] == "active" and status["lock_held"] is True
    assert status["requests"]["in_flight"] == 1 and status["requests"]["unknown"] == 0
    assert status["awaiting_reconciliation"] == [] and status["in_flight"] == by_question_attempts(run_dir, "q3")
    assert status["counts"]["awaiting_reconciliation"] == 0 and status["counts"]["in_flight"] == 1
    assert "active" in text and "in flight" in text and "the-live-invocation" in text
    assert "reconcile" not in text and "needs_reconciliation" not in text
    assert all("reconcile" not in step for step in status["next"])


def by_question_attempts(run_dir: Path, question_id: str) -> list[str]:
    request_id = next(r["request_id"] for r in requests_of(run_dir) if r["question_id"] == question_id)
    return [e["attempt_id"] for e in events_of(run_dir) if e["event"] == "dispatch_started" and e["request_id"] == request_id]


def test_status_after_the_process_is_gone_reports_the_open_attempt_as_needing_reconciliation(project: Project) -> None:
    run_dir, holder = _hold_while_open(project)
    holder.release()
    status = lr.run_status(run_dir)
    assert status["run_state"] == "needs_reconciliation" and status["requests"]["unknown"] == 1
    assert status["requests"]["in_flight"] == 0 and status["in_flight"] == []
    assert status["awaiting_reconciliation"] == by_question_attempts(run_dir, "q3")


def test_a_recorded_unknown_outcome_stays_unknown_while_the_lock_is_held(project: Project) -> None:
    """L6: only attempts with no result are in flight. An outcome_unknown event is a result."""
    go(project, Rig(steps={"q2": [FakeStep("timeout_unknown")]}))
    run_dir = project.run_dir("run-a")
    holder = lr.RunLock(run_dir, "someone")
    holder.acquire()
    try:
        status = lr.run_status(run_dir)
    finally:
        holder.release()
    assert status["requests"]["unknown"] == 1 and status["requests"]["in_flight"] == 0
    assert status["awaiting_reconciliation"] == by_question_attempts(run_dir, "q2")


def test_a_network_outage_that_fails_questions_before_sending_trips_the_circuit_breaker(project: Project) -> None:
    """Connect errors never leave the machine, but three questions failing in a row still share a cause."""
    outage = [FakeStep("connect_error")] * 3
    rig = Rig(steps={"q1": outage, "q2": outage, "q3": outage})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    ended = [e for e in events_of(run_dir) if e["event"] == "invocation_ended"][-1]
    assert ended["reason"] == "circuit_breaker"
    assert set(rig.sent(run_dir)) == {"q1", "q2", "q3"}, "q4 and q6 are not attempted after the breaker trips"
    assert result.exit_code == lr.EXIT_NEEDS_ATTENTION


def test_an_unknown_outcome_keeps_the_provider_payload_as_billing_evidence(project: Project) -> None:
    """A malformed 200 reply may carry usage that was billed; the payload stays in the event log."""

    class Malformed:
        def send(self, request: Any) -> Any:
            raise ProviderError(
                "200 reply without a choice message",
                kind="malformed_response",
                outcome="unknown",
                retryable=False,
                raw={"reason": "no choices", "payload": {"id": "chatcmpl-x", "usage": {"prompt_tokens": 11, "completion_tokens": 3}}},
            )

    lr.execute_run(options(project), provider_factory=lambda ctx: Malformed(), descriptor={"adapter": "malformed"}, sleep=lambda s: None, environ={})
    unknown = [e for e in events_of(project.run_dir("run-a")) if e["event"] == "outcome_unknown"]
    assert unknown and unknown[0]["provider_raw"]["payload"]["usage"] == {"prompt_tokens": 11, "completion_tokens": 3}


def test_status_on_a_scored_run_does_not_advise_reopen(project: Project) -> None:
    """A scored run is final, so status must not suggest a command that refuses."""
    rig = Rig(steps={"q1": [FakeStep("non_retryable_error")]})
    go(project, rig)
    run_dir = project.run_dir("run-a")
    lr.score_run_live(project_root=project.root, run_dir=run_dir)
    status = lr.run_status(run_dir)
    assert status["scored"] is True
    assert not any("reopen" in step for step in status["next"])


# ---------------------------------------------------------------------------
# Safety stop: dispatch ends when a safety or model assumption fails
#
# Ways the driver could fail here, written before the fix.
#   S1. A response above its input bound is saved and the next request is still sent.
#   S2. A returned model that differs from the configured one is saved and the next request is still sent.
#   S3. A ledger anomaly (measured cost off its usage, above its bound, or without usable usage) does not stop dispatch.
#   S4. The offending response, its usage, cost and model are dropped or rewritten when the run stops.
#   S5. A restart builds a provider or sends a request while the violation stands.
#   S6. A raised ceiling, reconcile or reopen clears the violation.
#   S7. A question that the stop left unserved is missing from the export, or exported as an abstention.
#   S8. A safety-stopped run is scored, or exits with a code that other outcomes use.
#   S9. The violation shows only when the last response is the offending one, or only in the final summary.
#   S10. A response whose response file exists but whose event was lost hides the violation from a restart.
# ---------------------------------------------------------------------------

SAFETY_EXIT = 6
SAFETY_STATE = "safety_stopped"
OTHER_EXIT_CODES = (0, 1, 2, 3, 4, 5)
UNSERVED_TYPE = "safety_stop"


def bound_one_services() -> lr.Services:
    """The reviewer's injection: every request claims an input upper bound of 1 token."""
    import dataclasses

    return dataclasses.replace(lr.Services.default(), input_token_upper_bound=lambda messages: 1)


def attempt_id_of(run_dir: Path, question_id: str, attempt: int = 1) -> str:
    request_id = next(r["request_id"] for r in requests_of(run_dir) if r["question_id"] == question_id)
    return f"{request_id}-a{attempt}"


def stop_violations(run_dir: Path) -> list[dict[str, Any]]:
    return _ended(run_dir)["safety_violations"]


def conditions(violations: list[dict[str, Any]]) -> set[tuple[str | None, str]]:
    return {(v["attempt_id"], v["condition"]) for v in violations}


def bound_one_run(project: Project, name: str = "run-a") -> tuple[Rig, lr.LiveResult, Path]:
    rig = Rig()
    result = go(project, rig, name, services=bound_one_services())
    return rig, result, project.run_dir(name)


def no_provider_factory(context: lr.ProviderContext) -> Any:
    raise AssertionError("the provider factory must not be called while a safety violation stands")


def go_without_provider(project: Project, name: str = "run-a", *, services: lr.Services | None = None, **kwargs: Any) -> lr.LiveResult:
    return lr.execute_run(
        options(project, name, **kwargs),
        services=services,
        provider_factory=no_provider_factory,
        descriptor=Rig().descriptor,
        sleep=lambda seconds: None,
        environ={},
    )


def test_the_safety_stop_has_its_own_exit_code_state_and_end_reason() -> None:
    assert lr.EXIT_SAFETY_STOPPED == SAFETY_EXIT and SAFETY_EXIT not in OTHER_EXIT_CODES
    assert lr.RUN_SAFETY_STOPPED == SAFETY_STATE and SAFETY_STATE in lr.RUN_STATES
    assert lr.END_SAFETY_STOP == "safety_stop" and lr.END_SAFETY_STOP in lr.END_REASONS


def test_an_input_bound_exceedance_stops_dispatch_after_the_first_offending_response(project: Project) -> None:
    """S1, S4, S7, S9: one request goes out, its response is kept in full, the rest are exported as unserved safety stops."""
    rig, result, run_dir = bound_one_run(project)
    assert rig.sent(run_dir) == ["q1"], "no request is dispatched after the violation"
    assert result.exit_code == SAFETY_EXIT
    ended = _ended(run_dir)
    assert ended["reason"] == "safety_stop"
    q1_attempt = attempt_id_of(run_dir, "q1")
    assert (q1_attempt, "input_bound_exceeded") in conditions(stop_violations(run_dir))
    assert not any(e["event"] == "dispatch_started" and e["question_id"] != "q1" for e in events_of(run_dir))
    saved = next(e for e in events_of(run_dir) if e["event"] == "response_saved")
    assert saved["input_bound_exceeded"] is True and saved["usage"]["input_tokens"] > 1
    response = json.loads((run_dir / saved["response_file"]).read_text())
    assert response["returned_model"] == lr.FAKE_CONFIG.model and response["usage"] == saved["usage"]
    assert response["measured_cost"] == saved["measured_cost"] and response["provider_response"]["text"] == "twelve months"
    summary = summary_of(run_dir)
    assert summary["run_state"] == SAFETY_STATE and summary["exit_code"] == SAFETY_EXIT
    assert summary["valid_baseline"] is False and any("safety" in b for b in summary["valid_baseline_blockers"])
    assert (q1_attempt, "input_bound_exceeded") in conditions(summary["safety_violations"])
    predictions = by_question(run_dir)
    assert list(predictions) == QUESTIONS, "every pilot question stays in the export"
    assert predictions["q1"]["status"] == "answered" and predictions["q1"]["usage"] == saved["usage"]
    assert predictions["q5"]["status"] == "no_evidence"
    for qid in ("q2", "q3", "q4", "q6"):
        record = predictions[qid]
        assert record["status"] == "execution_failed" and record["unserved"] is True and record["abstained"] is False
        assert record["answer"] is None and record["failure"]["type"] == UNSERVED_TYPE
    assert summary["counts"]["unserved"] == 4 and summary["counts"]["execution_failed"] == 4


def test_a_restart_after_the_stop_builds_no_provider_and_makes_no_call(project: Project) -> None:
    """S5: the violation is rebuilt from the durable records before any provider exists. Nothing is written."""
    _, _, run_dir = bound_one_run(project)
    before = file_snapshot(run_dir)
    fresh = Rig()
    result = lr.execute_run(
        options(project),
        services=bound_one_services(),
        provider_factory=fresh.factory,
        descriptor=fresh.descriptor,
        sleep=lambda seconds: None,
        environ={},
    )
    assert result.exit_code == SAFETY_EXIT and "safety" in result.message
    assert fresh.factory_calls == 0 and fresh.providers == []
    assert result.summary["run_state"] == SAFETY_STATE
    assert {k: v for k, v in file_snapshot(run_dir).items() if k != "run.lock"} == {k: v for k, v in before.items() if k != "run.lock"}


def test_a_restart_refuses_even_when_no_provider_factory_is_offered(project: Project) -> None:
    _, _, run_dir = bound_one_run(project)
    result = go_without_provider(project, services=bound_one_services())
    assert result.exit_code == SAFETY_EXIT and "safety" in result.message


def test_status_and_export_rebuild_the_violation_from_the_records(project: Project) -> None:
    _, _, run_dir = bound_one_run(project)
    (run_dir / "run_summary.json").unlink()
    (run_dir / "predictions.jsonl").unlink()
    status = lr.run_status(run_dir, bound_one_services())
    assert status["run_state"] == SAFETY_STATE and status["exit_code"] == SAFETY_EXIT
    assert (attempt_id_of(run_dir, "q1"), "input_bound_exceeded") in conditions(status["safety_violations"])
    text = " ".join(lr.format_status(status).split())
    assert "safety" in text and "successor run" in text
    assert "cannot undo a charge" in text and "provider-reported usage" in text and "valid bounds" in text
    assert not any("run again" in step for step in status["next"]), "run again would only be refused"
    exported = lr.export_run(run_dir, bound_one_services())
    assert exported.exit_code == SAFETY_EXIT and exported.summary["run_state"] == SAFETY_STATE
    assert list(by_question(run_dir)) == QUESTIONS


def test_a_safety_stopped_run_is_not_scored_but_still_exports_every_question(project: Project) -> None:
    """S8: score needs a complete run, and a safety-stopped run is not one."""
    _, _, run_dir = bound_one_run(project)
    with pytest.raises(RunnerRefusal, match="safety"):
        lr.score_run_live(project_root=project.root, run_dir=run_dir, services=bound_one_services())
    assert not (run_dir / "scores.jsonl").exists() and not (run_dir / "score_summary.json").exists()
    assert len(predictions_of(run_dir)) == len(QUESTIONS)


def test_a_returned_model_mismatch_stops_dispatch(project: Project) -> None:
    """S2: q2 comes back from another model; q3, q4 and q6 are never sent."""
    rig = Rig(steps={"q2": [FakeStep("answer", text="x", returned_model="fake-answer-model-2")]})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2"]
    assert result.exit_code == SAFETY_EXIT and _ended(run_dir)["reason"] == "safety_stop"
    assert conditions(stop_violations(run_dir)) == {(attempt_id_of(run_dir, "q2"), "returned_model_mismatch")}
    predictions = by_question(run_dir)
    assert predictions["q2"]["status"] == "answered" and predictions["q2"]["returned_model"] == "fake-answer-model-2"
    assert [predictions[q]["failure"]["type"] for q in ("q3", "q4", "q6")] == [UNSERVED_TYPE] * 3
    assert summary_of(run_dir)["run_state"] == SAFETY_STATE and summary_of(run_dir)["valid_baseline"] is False
    restart = Rig(steps={"q2": [FakeStep("answer", text="x", returned_model="fake-answer-model-2")]})
    assert go(project, restart).exit_code == SAFETY_EXIT
    assert restart.factory_calls == 0


def test_a_response_without_a_returned_model_stops_dispatch(project: Project) -> None:
    import dataclasses

    class NoModel:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.calls = 0

        def send(self, request: Any) -> Any:
            self.calls += 1
            return dataclasses.replace(self.inner.send(request), returned_model=None)

    rig = Rig()
    wrapped: list[NoModel] = []

    def factory(context: lr.ProviderContext) -> NoModel:
        wrapped.append(NoModel(lr.build_fake_provider(rig.steps, rig.default, lr.FAKE_CONFIG, context)))
        return wrapped[-1]

    result = lr.execute_run(options(project), provider_factory=factory, descriptor=rig.descriptor, sleep=lambda s: None, environ={})
    run_dir = project.run_dir("run-a")
    assert wrapped[0].calls == 1 and result.exit_code == SAFETY_EXIT
    assert conditions(stop_violations(run_dir)) == {(attempt_id_of(run_dir, "q1"), "returned_model_mismatch")}


def _ledger_case_scaled(real: lr.Services) -> lr.Services:
    import dataclasses

    def half(usage: Any, prices: Any) -> Any:
        cost = real.measured_cost(usage, prices)
        return None if cost is None else cost / 2

    return dataclasses.replace(real, measured_cost=half)


def _ledger_case_above_bound(real: lr.Services) -> lr.Services:
    import dataclasses

    return dataclasses.replace(real, measured_cost=lambda usage, prices: 999.0)


def test_a_measured_cost_that_differs_from_its_usage_stops_dispatch(project: Project) -> None:
    rig = Rig()
    result = go(project, rig, services=_ledger_case_scaled(lr.Services.default()))
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1"] and result.exit_code == SAFETY_EXIT
    assert (attempt_id_of(run_dir, "q1"), "ledger_anomaly") in conditions(stop_violations(run_dir))
    assert any("differs" in v["detail"] for v in stop_violations(run_dir))


def test_a_measured_cost_above_its_own_upper_bound_stops_dispatch(project: Project) -> None:
    rig = Rig()
    result = go(project, rig, services=_ledger_case_above_bound(lr.Services.default()))
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1"] and result.exit_code == SAFETY_EXIT
    assert (attempt_id_of(run_dir, "q1"), "ledger_anomaly") in conditions(stop_violations(run_dir))
    assert any("upper bound" in v["detail"] for v in stop_violations(run_dir))


def test_a_recorded_cost_without_reproducible_usage_stops_dispatch(project: Project) -> None:
    import dataclasses

    from faar.live_contract import ProviderUsage

    class Stripped:
        def __init__(self, inner: Any) -> None:
            self.inner = inner
            self.calls = 0

        def send(self, request: Any) -> Any:
            self.calls += 1
            return dataclasses.replace(self.inner.send(request), usage=ProviderUsage())

    rig = Rig()
    made: list[Stripped] = []

    def factory(context: lr.ProviderContext) -> Stripped:
        made.append(Stripped(lr.build_fake_provider(rig.steps, rig.default, lr.FAKE_CONFIG, context)))
        return made[-1]

    services = dataclasses.replace(lr.Services.default(), measured_cost=lambda usage, prices: 0.001)
    result = lr.execute_run(options(project), services=services, provider_factory=factory, descriptor=rig.descriptor, sleep=lambda s: None, environ={})
    run_dir = project.run_dir("run-a")
    assert made[0].calls == 1 and result.exit_code == SAFETY_EXIT
    assert any("cannot reproduce" in v["detail"] for v in stop_violations(run_dir))


def test_any_ledger_anomaly_blocks_even_a_kind_the_ledger_does_not_know_today(project: Project) -> None:
    """Every anomaly the ledger reports is blocking. This one appears only after the first response is saved."""
    import dataclasses

    real = lr.Services.default()

    def later_anomaly(events: Any, prices: Any) -> Any:
        ledger = real.ledger_from_events(events, prices)
        if any(e["event"] == "response_saved" for e in events):
            return dataclasses.replace(ledger, anomalies=("not-an-attempt: a kind added later",))
        return ledger

    rig = Rig()
    result = go(project, rig, services=dataclasses.replace(real, ledger_from_events=later_anomaly))
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1"] and result.exit_code == SAFETY_EXIT
    assert conditions(stop_violations(run_dir)) == {(None, "ledger_anomaly")}
    assert "a kind added later" in stop_violations(run_dir)[0]["detail"]


def test_a_violation_on_the_last_dispatch_still_ends_the_run_as_a_safety_stop(project: Project) -> None:
    """S9: nothing is left to dispatch, but the records still hold a violation, so the state and exit code say so."""
    rig = Rig(steps={"q6": [FakeStep("answer", text="x", returned_model="fake-answer-model-2")]})
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == SENT
    assert result.exit_code == SAFETY_EXIT and _ended(run_dir)["reason"] == "safety_stop"
    summary = summary_of(run_dir)
    assert summary["run_state"] == SAFETY_STATE and summary["valid_baseline"] is False
    assert summary["counts"]["unserved"] == 0 and by_question(run_dir)["q6"]["status"] == "answered"


def test_ordinary_valid_responses_still_complete_normally(project: Project) -> None:
    rig = Rig()
    result = go(project, rig)
    run_dir = project.run_dir("run-a")
    assert result.exit_code == 0 and rig.sent(run_dir) == SENT
    assert _ended(run_dir)["reason"] == "completed" and "safety_violations" not in _ended(run_dir)
    summary = summary_of(run_dir)
    assert summary["run_state"] == "complete" and "safety_violations" not in summary, "a clean run's summary is unchanged"
    assert lr.score_run_live(project_root=project.root, run_dir=run_dir).exit_code == 0


def test_a_response_without_usage_is_not_a_violation_because_its_cost_stays_reserved(project: Project) -> None:
    """Missing usage cannot show an exceedance. The ledger keeps the attempt at its upper bound, so the ceiling holds."""
    rig = Rig(steps={"q1": [FakeStep("missing_usage")]})
    result = go(project, rig)
    assert result.exit_code != SAFETY_EXIT and rig.sent(project.run_dir("run-a")) == SENT


def test_raising_the_ceiling_reconcile_and_reopen_do_not_clear_the_violation(project: Project) -> None:
    """S6: q1 fails, q2 is unknown, q3 returns a different model. Nothing later can lift the stop."""
    steps = {
        "q1": [FakeStep("non_retryable_error")],
        "q2": [FakeStep("timeout_unknown")],
        "q3": [FakeStep("answer", text="x", returned_model="fake-answer-model-2")],
    }
    rig = Rig(steps)
    assert go(project, rig).exit_code == SAFETY_EXIT
    run_dir = project.run_dir("run-a")
    assert rig.sent(run_dir) == ["q1", "q2", "q3"]
    q3_violation = (attempt_id_of(run_dir, "q3"), "returned_model_mismatch")
    assert q3_violation in conditions(stop_violations(run_dir))
    log_before = (run_dir / "attempts.jsonl").read_bytes()

    later = Rig(steps)
    raised = go(project, later, safety_ceiling=100.0, raise_safety_ceiling=500.0, authorization_note="Lead approved a higher ceiling")
    assert raised.exit_code == SAFETY_EXIT and later.factory_calls == 0
    assert (run_dir / "attempts.jsonl").read_bytes() == log_before, "a refused raise records nothing"
    assert summary_of(run_dir)["safety_ceiling"]["amount"] == 100.0

    reconciled = lr.reconcile_attempt(
        run_dir=run_dir, attempt_id=attempt_id_of(run_dir, "q2"), resolution="mark_failed", note="the provider shows no request"
    )
    assert reconciled.summary["run_state"] == SAFETY_STATE
    reopened = lr.reopen_question(run_dir=run_dir, question_id="q1", note="fixed the request")
    assert reopened.summary["run_state"] == SAFETY_STATE
    assert q3_violation in conditions(reopened.summary["safety_violations"])

    again = Rig(steps)
    assert go(project, again).exit_code == SAFETY_EXIT
    assert again.factory_calls == 0 and again.providers == []
    assert lr.export_run(run_dir).exit_code == SAFETY_EXIT
    assert lr.run_status(run_dir)["run_state"] == SAFETY_STATE


def test_only_a_successor_run_with_corrected_configuration_can_go_on(project: Project) -> None:
    """The identity check stays: the same directory with corrected code is refused, and a new directory works."""
    _, _, run_dir = bound_one_run(project)
    with pytest.raises(RunnerRefusal, match="regenerate differently|different identity"):
        go(project, Rig())
    successor_rig = Rig()
    successor = go(project, successor_rig, "run-b")
    assert successor.exit_code == 0 and successor.summary["run_state"] == "complete"
    assert summary_of(run_dir)["run_state"] == SAFETY_STATE, "the stopped run stays a record"


def test_a_crash_between_the_response_file_and_its_event_does_not_hide_the_violation(project: Project) -> None:
    """S10: the response file is a durable record. A restart reads it and refuses before any provider exists.

    The restart also appends the recovered response_saved event (M3), so the records name the answer.
    """
    rig = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, services=bound_one_services(), crash_hook=crash_at(lr.CRASH_AFTER_RESPONSE_FILE, "q1"))
    run_dir = project.run_dir("run-a")
    assert not any(e["event"] == "response_saved" for e in events_of(run_dir)) and (run_dir / "responses").is_dir()
    before = (run_dir / "attempts.jsonl").read_bytes()
    restart = Rig()
    result = lr.execute_run(
        options(project), services=bound_one_services(), provider_factory=restart.factory, descriptor=restart.descriptor,
        sleep=lambda s: None, environ={},
    )
    assert result.exit_code == SAFETY_EXIT and restart.factory_calls == 0
    after = (run_dir / "attempts.jsonl").read_bytes()
    assert after.startswith(before), "the restart only appends: the earlier lines stay byte-identical"
    appended = [json.loads(line) for line in after[len(before) :].splitlines()]
    assert [e["event"] for e in appended] == ["response_saved"] and appended[0]["recovered"] is True
    status = lr.run_status(run_dir, bound_one_services())
    assert status["run_state"] == SAFETY_STATE


def test_the_cli_exits_with_the_safety_code_and_the_help_documents_it(
    project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from faar import request_budget

    monkeypatch.setattr(request_budget, "input_token_upper_bound", lambda messages: 1)
    fake = write_fake_script(project)
    assert cli_run(project, "cli-safety", fake) == SAFETY_EXIT
    capsys.readouterr()
    assert cli_run(project, "cli-safety", fake) == SAFETY_EXIT, "a restart is refused with the same code"
    assert "safety" in capsys.readouterr().out
    assert cli.main(["status", "--run-dir", str(project.run_dir("cli-safety"))]) == 0
    assert "safety_stopped" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert f"{SAFETY_EXIT} safety_stopped" in text
    for phrase in ("safety_stop", "successor run", "do not clear the violation", "cannot undo a charge", "provider-reported usage"):
        assert phrase in text


# ---------------------------------------------------------------------------
# Review fixes: records edited to clear a safety stop (H1)
#
# Ways the stop could be cleared by an edit, written before the fix.
#   T1. run_config.json names another model in identity.provider.model while identity_sha256 stays. The
#       returned-model check then compares against the edited name and the violation disappears.
#   T2. A response_saved event in attempts.jsonl is edited (returned_model, usage, input_bound_exceeded,
#       measured_cost, answer, output_status) while the hash-pinned response file keeps the original.
#   T3. The edit is accepted by run but not by status or export, or the reverse.
#   T4. A refusal happens after the provider factory was called or after a request was sent.
#
# The event log has no hash chain, so an edit that keeps every cross-check consistent is out of scope.
# ---------------------------------------------------------------------------


def _offending_run(project: Project) -> tuple[Path, Rig]:
    rig = Rig({"q1": [FakeStep("answer", text="x", returned_model="fake-answer-model-2")]})
    assert go(project, rig, services=bound_one_services()).exit_code == SAFETY_EXIT
    return project.run_dir("run-a"), rig


def _rewrite_saved_events(run_dir: Path, edit: Any) -> None:
    path = run_dir / "attempts.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["event"] == "response_saved":
            edit(record)
        out.append(json.dumps(record, separators=(",", ":")))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _assert_every_command_refuses(project: Project, run_dir: Path, rig: Rig, match: str) -> None:
    fresh = Rig(rig.steps)  # the same script and services, so the refusal cannot come from a changed run identity
    with pytest.raises(RunnerRefusal, match=match):
        go(project, fresh, services=bound_one_services())
    assert fresh.factory_calls == 0 and fresh.providers == []
    with pytest.raises(RunnerRefusal, match=match):
        lr.run_status(run_dir, bound_one_services())
    with pytest.raises(RunnerRefusal, match=match):
        lr.export_run(run_dir, bound_one_services())


def test_an_edited_identity_in_run_config_is_refused_by_every_command(project: Project) -> None:
    """T1, T4: the stored model is changed to the returned one; identity_sha256 no longer matches the identity."""
    run_dir, rig = _offending_run(project)
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config["identity"]["provider"]["model"] = "fake-answer-model-2"
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    before = strip_lock(file_snapshot(run_dir))
    _assert_every_command_refuses(project, run_dir, rig, "run_config.json.*identity")
    assert strip_lock(file_snapshot(run_dir)) == before


@pytest.mark.parametrize(
    "edit",
    [
        lambda e: e.update(returned_model=lr.FAKE_CONFIG.model),
        lambda e: e["usage"].update(input_tokens=1),
        lambda e: e.update(input_bound_exceeded=False),
        lambda e: e.update(measured_cost=None),
        lambda e: e.update(answer="something else"),
        lambda e: e.update(output_status="empty"),
    ],
    ids=["returned_model", "usage", "input_bound_exceeded", "measured_cost", "answer", "output_status"],
)
def test_an_event_that_disagrees_with_its_hash_pinned_response_file_is_refused(project: Project, edit: Any) -> None:
    """T2, T3, T4: the response file is the stronger evidence. Any field that differs refuses run, status and export."""
    run_dir, rig = _offending_run(project)
    _rewrite_saved_events(run_dir, edit)
    before = strip_lock(file_snapshot(run_dir))
    _assert_every_command_refuses(project, run_dir, rig, "response file")
    assert strip_lock(file_snapshot(run_dir)) == before
    assert len(rig.sent(run_dir)) == 1, "no dispatch after the edit"


def test_an_unedited_stopped_run_still_reports_the_stop_after_the_new_checks(project: Project) -> None:
    run_dir, rig = _offending_run(project)
    fresh = Rig(rig.steps)
    assert go(project, fresh, services=bound_one_services()).exit_code == SAFETY_EXIT and fresh.factory_calls == 0
    assert lr.run_status(run_dir, bound_one_services())["run_state"] == SAFETY_STATE
    assert lr.export_run(run_dir, bound_one_services()).exit_code == SAFETY_EXIT


@pytest.mark.parametrize("damage", ["question_id", "request_id", "attempt", "missing_field"])
def test_a_recoverable_response_file_that_names_another_question_or_lacks_fields_is_refused(project: Project, damage: str) -> None:
    """T2: the file of an attempt whose event was lost must describe that attempt, and must hold every field recovery copies."""
    rig = Rig({"q1": [FakeStep("answer", text="x", returned_model="fake-answer-model-2")]})
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_AFTER_RESPONSE_FILE, "q1"))
    run_dir = project.run_dir("run-a")
    path = next((run_dir / "responses").iterdir())
    payload = json.loads(path.read_text(encoding="utf-8"))
    if damage == "missing_field":
        del payload["usage"]
    else:
        payload[damage] = "q3" if damage == "question_id" else ("f" * 32 if damage == "request_id" else 7)
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = strip_lock(file_snapshot(run_dir))
    fresh = Rig(rig.steps)
    with pytest.raises(RunnerRefusal, match="response file"):
        go(project, fresh)
    assert fresh.factory_calls == 0
    with pytest.raises(RunnerRefusal, match="response file"):
        lr.run_status(run_dir)
    with pytest.raises(RunnerRefusal, match="response file"):
        lr.export_run(run_dir)
    assert strip_lock(file_snapshot(run_dir)) == before


def test_the_messages_and_help_state_the_exact_model_match_and_the_limit_of_the_event_log(
    project: Project, capsys: pytest.CaptureFixture[str]
) -> None:
    """H1.3, L6: the stop is an exact string match, and it does not defend against consistent file edits."""
    run_dir, rig = _offending_run(project)
    for text in (
        " ".join(lr.format_status(lr.run_status(run_dir, bound_one_services())).split()),
        " ".join(go(project, Rig(rig.steps), services=bound_one_services()).message.split()),
    ):
        assert "exact match" in text and "dated snapshot" in text and "alias" in text
    bound_only = Rig()
    go(project, bound_only, "run-b", services=bound_one_services())
    status_text = " ".join(lr.format_status(lr.run_status(project.run_dir("run-b"), bound_one_services())).split())
    assert "exact match" not in status_text, "the model note appears only for a returned_model_mismatch"
    with pytest.raises(SystemExit):
        cli.main(["run", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    for phrase in ("exact match", "dated snapshot", "no hash chain", "honest operation", "out of scope"):
        assert phrase in help_text, phrase
    source = Path(lr.__file__).read_text(encoding="utf-8")
    assert "no hash chain" in source


# ---------------------------------------------------------------------------
# Review fixes: a crash between the response file and its event, then a safety stop (M3)
#
# Ways the restart could fail, written before the fix.
#   R1. The restart refuses on the response file but leaves the attempt open, so the durable records still
#       show an attempt awaiting reconciliation and no answer.
#   R2. The exports are missing or stale, so predictions.jsonl and run_summary.json do not name the stop.
#   R3. status advises reconcile for an attempt whose response file is readable and holds the answer.
#   R4. The recovery sends a request, builds a provider or reads the response file wrongly.
# ---------------------------------------------------------------------------


def _crashed_offending_run(project: Project) -> tuple[Path, Rig]:
    rig = Rig({"q1": [FakeStep("answer", text="twelve months", returned_model="fake-answer-model-2")]})
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_AFTER_RESPONSE_FILE, "q1"))
    run_dir = project.run_dir("run-a")
    assert not any(e["event"] == "response_saved" for e in events_of(run_dir))
    assert not (run_dir / "predictions.jsonl").exists()
    return run_dir, rig


def test_status_and_export_before_a_restart_do_not_advise_reconcile_for_a_recoverable_response(project: Project) -> None:
    """R3: the response file is readable, so the attempt is not awaiting reconciliation."""
    run_dir, _ = _crashed_offending_run(project)
    status = lr.run_status(run_dir)
    assert status["awaiting_reconciliation"] == [] and status["counts"]["awaiting_reconciliation"] == 0
    assert status["run_state"] == SAFETY_STATE
    assert not any("reconcile each attempt" in step for step in status["next"])
    assert "awaiting reconciliation" not in lr.format_status(status)
    exported = lr.export_run(run_dir)
    assert exported.exit_code == SAFETY_EXIT
    record = by_question(run_dir)["q1"]
    assert record["status"] == "answered" and record["awaiting_reconciliation"] is False
    assert record["returned_model"] == "fake-answer-model-2" and record["answer"] == "twelve months"


def test_a_restart_after_a_crash_before_the_event_records_the_response_and_exports_the_stop(project: Project) -> None:
    """R1, R2, R4: the restart appends the recovered response_saved event, writes the exports and dispatches nothing."""
    run_dir, rig = _crashed_offending_run(project)
    restart = Rig(rig.steps)
    result = go(project, restart)
    assert result.exit_code == SAFETY_EXIT and restart.factory_calls == 0 and restart.providers == []
    assert restart.sent(run_dir) == []
    saved = [e for e in events_of(run_dir) if e["event"] == "response_saved"]
    assert len(saved) == 1 and saved[0]["recovered"] is True and saved[0]["returned_model"] == "fake-answer-model-2"
    assert [e for e in events_of(run_dir) if e["event"] == "outcome_unknown"] == []
    assert (run_dir / "predictions.jsonl").exists() and (run_dir / "run_summary.json").exists()
    summary = summary_of(run_dir)
    assert summary["run_state"] == SAFETY_STATE and summary["exit_code"] == SAFETY_EXIT
    assert (attempt_id_of(run_dir, "q1"), "returned_model_mismatch") in conditions(summary["safety_violations"])
    record = by_question(run_dir)["q1"]
    assert record["status"] == "answered" and record["answer"] == "twelve months" and record["returned_model"] == "fake-answer-model-2"
    assert list(by_question(run_dir)) == QUESTIONS
    assert by_question(run_dir)["q2"]["failure"]["type"] == UNSERVED_TYPE
    status = lr.run_status(run_dir)
    assert status["awaiting_reconciliation"] == [] and not any("reconcile each attempt" in step for step in status["next"])
    assert status["predictions_current"] is True
    # A second restart finds nothing open and changes nothing.
    before = strip_lock(file_snapshot(run_dir))
    again = Rig(rig.steps)
    assert go(project, again).exit_code == SAFETY_EXIT and again.factory_calls == 0
    assert strip_lock(file_snapshot(run_dir)) == before


def test_a_restart_after_a_crash_before_the_event_moves_a_torn_tail_aside_and_keeps_the_stop(project: Project) -> None:
    run_dir, rig = _crashed_offending_run(project)
    with (run_dir / "attempts.jsonl").open("ab") as handle:
        handle.write(b'{"event":"dispatch_started","seq":')
    restart = Rig(rig.steps)
    assert go(project, restart).exit_code == SAFETY_EXIT and restart.factory_calls == 0
    assert list(run_dir.glob("attempts.uncommitted.*")), "the torn bytes are kept"
    recovered = next(e for e in events_of(run_dir) if e["event"] == "response_saved")
    assert recovered["recovered"] is True and recovered["recovered_uncommitted_tail"]["bytes"] > 0
    assert by_question(run_dir)["q1"]["status"] == "answered"


def test_score_after_a_crash_before_the_last_event_advises_run_not_reconcile(project: Project) -> None:
    """The last answer sits only in its response file: score must say to run once, not to reconcile."""
    rig = Rig()
    with pytest.raises(SimulatedCrash):
        go(project, rig, crash_hook=crash_at(lr.CRASH_AFTER_RESPONSE_FILE, "q6"))
    run_dir = project.run_dir("run-a")
    with pytest.raises(RunnerRefusal) as refused:
        lr.score_run_live(project_root=project.root, run_dir=run_dir)
    message = str(refused.value)
    assert "run" in message and "record" in message and "reconcile the unknown" not in message
    fresh = Rig(rig.steps)
    go(project, fresh)
    assert fresh.sent(run_dir) == [], "recording the recovered response sends nothing"
    assert lr.score_run_live(project_root=project.root, run_dir=run_dir).exit_code == lr.EXIT_OK
