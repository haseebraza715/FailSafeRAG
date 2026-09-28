"""Tests for the offline pilot runner (``faar.pilot_runner`` and ``scripts/experiments/run_pilot_offline.py``).

Ways the runner could fail, written before the code. Each test below names the item it covers.

Leakage
  L1. Generation opens the evaluation manifest, selection record, inspection
      files, ``qas*.json``, gt text, annotations or the audit manifest.
  L2. A gold field reaches a runtime question object, or a manifest path points
      generation at an evaluation file.
  L3. An evaluation-only value changes the predictions.
Retrieval scope
  R1. A question retrieves a chunk from another document that shares its vocabulary.
  R2. A page with empty or missing OCR yields a chunk, or is not counted.
  R3. A document with no retrievable token crashes retrieval instead of giving no evidence.
Terminal records
  T1. A question is dropped, duplicated or left non-terminal.
  T2. An exception for one question stops the run or loses the question.
  T3. An injected failure is missing from the predictions, the summary, the config or the fingerprint.
  T4. The exit status hides an execution failure.
Input integrity
  I1. A MinerU file or a page differs from the manifest hash and the run continues.
  I2. The per-page hash differs from the one the pilot builder stored.
  I3. A frozen input file changes.
Run directory
  D1. A rerun overwrites, or a compatible rerun rewrites files.
  D2. A run with a different fingerprint, a partial run or a corrupt run is reused.
  D3. A nondeterministic rerun passes as identical.
  D4. A run is written under ``results/pilots/``.
Output
  O1. A prediction record breaks the contract keys, or a file holds NaN.
  O2. The provenance fields (commit, dirty paths, hashes, settings) are missing or wrong.
Scoring
  S1. Scoring overwrites, hides an execution failure, or joins mismatched ids.
  S2. Generation calls the scorer.
CLI
  C1. Usage errors, refusals and failures share one exit code.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import subprocess
import sys
import types
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import run_pilot_offline as cli

from faar import pilot_runner as pr
from faar.pilot_runner import RunnerRefusal

PILOT_ID = "fix_v1"
MINERU_ROOT = "OHR-Bench/data/retrieval_base/MinerU"
DOC_A = "law/docA"
DOC_B = "law/doc,B with space"

WARRANTY_A = "Alpha Corp warranty period is twelve months from the delivery date. The buyer pays 500 dollars."
WARRANTY_B = "Beta Corp warranty period is twenty four months from the delivery date. The buyer pays 900 dollars."

QUESTION_TEXT = "What is the warranty period from the delivery date?"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Project:
    """A small project tree: runtime manifest, MinerU files and decoy evaluation files."""

    root: Path
    docs: dict[str, list[str | None]]
    questions: list[dict[str, str]]
    extra_question_fields: dict[str, Any] = field(default_factory=dict)

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

    def frozen_files(self) -> list[Path]:
        return [self.runtime_manifest, self.evaluation_manifest, *[self.mineru_path(d) for d in self.docs]]

    def write(self) -> Project:
        documents = []
        for doc_id, pages in self.docs.items():
            entries = [{"page_idx": i, "text": text} for i, text in enumerate(pages) if text is not None]
            path = self.mineru_path(doc_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = json.dumps(entries).encode("utf-8")
            path.write_bytes(raw)
            page_rows = []
            for i, text in enumerate(pages):
                if text is None:
                    status, digest = "missing", None
                else:
                    status = "ok" if text.strip() else "empty"
                    digest = sha256_bytes(text.encode("utf-8"))
                page_rows.append({"page_idx": i, "pdf_page_number": i + 1, "ocr_status": status, "ocr_text_sha256": digest})
            documents.append(
                {
                    "doc_id": doc_id,
                    "doc_type": "law",
                    "noisy_text": {
                        "path": f"{MINERU_ROOT}/{doc_id}.json",
                        "sha256": sha256_bytes(raw),
                        "source": "MinerU",
                        "status": "present",
                        "problems": [],
                    },
                    "pages": page_rows,
                }
            )
        runtime = {
            "kind": "runtime",
            "pilot_id": PILOT_ID,
            "schema_version": 1,
            "sources": {"noisy_text": {"root": MINERU_ROOT, "source": "MinerU"}},
            "documents": documents,
            "questions": [{**q, **self.extra_question_fields} for q in self.questions],
        }
        self.pilot_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_manifest.write_text(json.dumps(runtime, indent=1), encoding="utf-8")
        self.write_evaluation({q["question_id"]: {"answers": "twelve months", "evidence_pages": [0]} for q in self.questions})
        (self.pilot_dir / "selection_record.json").write_text("{}", encoding="utf-8")
        (self.pilot_dir / "inspection").mkdir(exist_ok=True)
        (self.pilot_dir / "inspection" / "cases.json").write_text("{}", encoding="utf-8")
        (self.root / "OHR-Bench" / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "OHR-Bench" / "data" / "qas_v2.json").write_text("[]", encoding="utf-8")
        return self

    def write_evaluation(self, questions: dict[str, Any], *, raw: str | None = None) -> None:
        payload = raw if raw is not None else json.dumps({"kind": "evaluation", "pilot_id": PILOT_ID, "questions": questions})
        self.evaluation_manifest.write_text(payload, encoding="utf-8")


def default_project(root: Path) -> Project:
    return Project(
        root,
        {DOC_A: [WARRANTY_A, "Alpha Corp termination clause: either party may end this agreement."], DOC_B: [WARRANTY_B]},
        [
            {"question_id": "q-a", "doc_id": DOC_A, "question": QUESTION_TEXT},
            {"question_id": "q-b", "doc_id": DOC_B, "question": QUESTION_TEXT},
            {"question_id": "q-a2", "doc_id": DOC_A, "question": "How much does the buyer pay?"},
        ],
    ).write()


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return default_project(tmp_path / "project")


def generate(project: Project, name: str = "run-one", **kwargs: Any) -> tuple[pr.RunnerResult, Path]:
    run_dir = project.root / "results" / "engineering" / name
    result = pr.generate_run(project_root=project.root, run_dir=run_dir, pilot_id=PILOT_ID, **kwargs)
    return result, run_dir


def read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]


def snapshot(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


# ---------------------------------------------------------------------------
# R1, L1, L2, L3
# ---------------------------------------------------------------------------


def test_every_evidence_item_belongs_to_the_assigned_document(project: Project) -> None:
    """R1: two documents share vocabulary, and each question sees only its own document."""
    result, run_dir = generate(project)
    assert result.exit_code == pr.EXIT_OK
    records = {r["question_id"]: r for r in read_predictions(run_dir)}
    assert records["q-a"]["status"] == records["q-b"]["status"] == "answered"
    for record in records.values():
        assert record["evidence"], "answered records keep their evidence"
        assert {e["doc_id"] for e in record["evidence"]} == {record["doc_id"]}
        assert all(e["chunk_id"].startswith(record["doc_id"] + "-p") for e in record["evidence"])
    assert "twelve months" in records["q-a"]["answer"]
    assert "twenty four months" in records["q-b"]["answer"]


def test_a_phrase_found_only_in_another_document_is_not_retrieved(project: Project) -> None:
    """R1: a question about text that only document B holds never returns B chunks for document A."""
    project.questions[:] = [{"question_id": "q-x", "doc_id": DOC_A, "question": "How much is 900 dollars for Beta Corp?"}]
    project.write()
    _, run_dir = generate(project)
    (record,) = read_predictions(run_dir)
    assert {e["doc_id"] for e in record["evidence"]} == {DOC_A}
    assert "Beta" not in record["answer"]


def test_generation_succeeds_without_the_evaluation_manifest(project: Project) -> None:
    """L1: nothing in generation needs the evaluation manifest."""
    project.evaluation_manifest.unlink()
    (project.pilot_dir / "selection_record.json").unlink()
    (project.pilot_dir / "inspection" / "cases.json").unlink()
    (project.pilot_dir / "inspection").rmdir()
    (project.root / "OHR-Bench" / "data" / "qas_v2.json").unlink()
    result, run_dir = generate(project)
    assert result.exit_code == pr.EXIT_OK
    assert len(read_predictions(run_dir)) == 3


def test_poisoned_evaluation_files_do_not_change_predictions(tmp_path: Path) -> None:
    """L3: predictions are byte-identical when every evaluation-only file is replaced with poison."""
    clean = default_project(tmp_path / "clean")
    poisoned = default_project(tmp_path / "poisoned")
    poisoned.write_evaluation({}, raw="{ this is not json")
    for name in ("selection_record.json", "inspection/cases.json"):
        (poisoned.pilot_dir / name).write_text("\x00 poison", encoding="utf-8")
    (poisoned.root / "OHR-Bench" / "data" / "qas_v2.json").write_text("poison", encoding="utf-8")
    a, dir_a = generate(clean)
    b, dir_b = generate(poisoned)
    assert (dir_a / "predictions.jsonl").read_bytes() == (dir_b / "predictions.jsonl").read_bytes()
    assert a.summary["fingerprint"] == b.summary["fingerprint"]
    assert a.summary["predictions_sha256"] == b.summary["predictions_sha256"]


def test_evaluation_answer_text_does_not_steer_the_answer(tmp_path: Path) -> None:
    """L3: an evaluation manifest that names a different gold answer changes nothing in generation."""
    plain = default_project(tmp_path / "plain")
    steered = default_project(tmp_path / "steered")
    steered.write_evaluation({q["question_id"]: {"answers": "Beta Corp 900 dollars", "evidence_pages": [1]} for q in steered.questions})
    _, dir_a = generate(plain)
    _, dir_b = generate(steered)
    assert (dir_a / "predictions.jsonl").read_bytes() == (dir_b / "predictions.jsonl").read_bytes()


def _spy_on_file_opens(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    opened: list[tuple[str, str]] = []
    real_open = io.open

    def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append((str(Path(file).resolve()), mode))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    return opened


def _reads_under(opened: list[tuple[str, str]], root: Path) -> set[str]:
    base = str(root.resolve())
    return {path for path, mode in opened if path.startswith(base) and not any(flag in mode for flag in "wax+")}


def test_generation_reads_only_the_runtime_manifest_and_mineru_files(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """L1: file access during generation is limited to the runtime inputs."""
    opened = _spy_on_file_opens(monkeypatch)
    generate(project)
    allowed = {str(project.runtime_manifest.resolve())} | {str(project.mineru_path(d).resolve()) for d in project.docs}
    assert _reads_under(opened, project.root) == allowed


def test_the_file_spy_would_see_an_evaluation_read(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """L1: the spy is live. Scoring, which reads the evaluation manifest, shows up in it."""
    _, run_dir = generate(project)
    install_fake_scoring(monkeypatch)
    opened = _spy_on_file_opens(monkeypatch)
    pr.score_run(project_root=project.root, run_dir=run_dir)
    assert str(project.evaluation_manifest.resolve()) in _reads_under(opened, project.root)


def test_runtime_question_object_has_no_gold_fields() -> None:
    """L2: the runtime question type holds three fields and no room for more."""
    assert pr.RuntimeQuestion.__slots__ == ("question_id", "doc_id", "question")
    question = pr.RuntimeQuestion("q", "d", "text")
    with pytest.raises((AttributeError, TypeError)):
        question.answers = "gold"  # type: ignore[attr-defined]


@pytest.mark.parametrize("field_name", ["answers", "evidence_pages", "evidence_context", "gt_reference"])
def test_a_runtime_question_with_a_gold_field_is_refused(project: Project, field_name: str) -> None:
    """L2: a runtime manifest that carries an evaluation field is refused before any prediction."""
    project.extra_question_fields = {field_name: "leak"}
    project.write()
    with pytest.raises(RunnerRefusal, match="must not see"):
        generate(project)
    assert not (project.root / "results" / "engineering").exists()


@pytest.mark.parametrize("bad_path", ["OHR-Bench/data/qas_v2.json", "results/pilots/fix_v1/evaluation_manifest.json", "../outside.json"])
def test_a_noisy_text_path_outside_the_mineru_root_is_refused(project: Project, bad_path: str) -> None:
    """L2: a manifest cannot point generation at an evaluation file."""
    payload = json.loads(project.runtime_manifest.read_text(encoding="utf-8"))
    payload["documents"][0]["noisy_text"]["path"] = bad_path
    payload["documents"][0]["noisy_text"]["sha256"] = "0" * 64
    project.runtime_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="outside the declared root"):
        generate(project)


# ---------------------------------------------------------------------------
# R2, R3, T1, T2, T3, T4
# ---------------------------------------------------------------------------


def test_a_document_with_only_empty_pages_gives_no_evidence_and_keeps_the_question(tmp_path: Path) -> None:
    """R2: empty and missing pages make no chunk, the question stays, and the condition is counted."""
    project = Project(
        tmp_path / "p",
        {"law/blank": ["", "   \n", None], DOC_A: [WARRANTY_A]},
        [
            {"question_id": "q-blank", "doc_id": "law/blank", "question": QUESTION_TEXT},
            {"question_id": "q-ok", "doc_id": DOC_A, "question": QUESTION_TEXT},
        ],
    ).write()
    result, run_dir = generate(project)
    blank, ok = read_predictions(run_dir)
    assert blank["status"] == "no_evidence"
    assert blank["no_evidence_reason"] == "no_text_chunks"
    assert result.summary["no_evidence_by_reason"] == {"no_text_chunks": 1, "no_text_content": 0, "no_retrieval_tokens": 0, "no_hits": 0}
    assert (blank["answer"], blank["abstained"], blank["evidence"], blank["failure"]) == ("", True, [], None)
    assert blank["ocr_condition"] == {"pages_total": 3, "pages_ok": 0, "pages_empty": 2, "pages_missing": 1, "chunks": 0}
    assert ok["status"] == "answered"
    assert result.summary["counts"] == {"questions": 2, "answered": 1, "no_evidence": 1, "execution_failed": 0, "abstained": 1}
    assert result.exit_code == pr.EXIT_OK


def test_empty_pages_are_counted_but_the_rest_of_the_document_is_searched(tmp_path: Path) -> None:
    """R2: a mixed document keeps its ok pages retrievable and reports the empty page."""
    project = Project(
        tmp_path / "p", {DOC_A: ["", WARRANTY_A, None]}, [{"question_id": "q1", "doc_id": DOC_A, "question": QUESTION_TEXT}]
    ).write()
    _, run_dir = generate(project)
    (record,) = read_predictions(run_dir)
    assert record["ocr_condition"] == {"pages_total": 3, "pages_ok": 1, "pages_empty": 1, "pages_missing": 1, "chunks": 1}
    assert {e["page_idx"] for e in record["evidence"]} == {1}


@pytest.mark.parametrize(
    ("page_text", "reason"),
    [
        # The pilot has this page: MinerU kept only two empty Markdown heading markers.
        ("# \n\n#", "no_text_content"),
        ("- | * |\n\n---", "no_text_content"),
        # Han-only text is content that the engineering tokeniser cannot index.
        ("在审题过程中应该遵循哪些步骤", "no_retrieval_tokens"),
    ],
)
def test_a_document_without_a_retrieval_token_gives_no_evidence(tmp_path: Path, page_text: str, reason: str) -> None:
    """R3: rank_bm25 divides by zero on a corpus with no token; the runner returns no_evidence and says why."""
    project = Project(
        tmp_path / "p", {"law/symbols": [page_text]}, [{"question_id": "q1", "doc_id": "law/symbols", "question": QUESTION_TEXT}]
    ).write()
    result, run_dir = generate(project)
    (record,) = read_predictions(run_dir)
    assert record["status"] == "no_evidence"
    assert record["no_evidence_reason"] == reason
    assert record["ocr_condition"]["chunks"] == 1
    expected = {"no_text_chunks": 0, "no_text_content": 0, "no_retrieval_tokens": 0, "no_hits": 0}
    expected[reason] = 1
    assert result.summary["no_evidence_by_reason"] == expected
    assert result.summary["documents_with_chunks_but_no_retrieval_tokens"] == ["law/symbols"]
    assert result.exit_code == pr.EXIT_OK


def test_zero_hits_from_the_retriever_give_no_evidence_with_reason_no_hits(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """The third reason: chunks with tokens exist but the retriever returns nothing."""
    from faar.retrieval import HybridRetriever

    monkeypatch.setattr(HybridRetriever, "retrieve", lambda self, query, top_k=None: [])
    result, run_dir = generate(project)
    records = read_predictions(run_dir)
    assert {r["status"] for r in records} == {"no_evidence"}
    assert {r["no_evidence_reason"] for r in records} == {"no_hits"}
    assert all(r["ocr_condition"]["chunks"] > 0 for r in records)
    assert result.summary["no_evidence_by_reason"] == {"no_text_chunks": 0, "no_text_content": 0, "no_retrieval_tokens": 0, "no_hits": 3}


def test_no_evidence_reason_is_null_unless_the_status_is_no_evidence(project: Project) -> None:
    """The reason field is null for answered and execution_failed records, and the summary counts stay zero."""
    result, run_dir = generate(project, inject_failures=["q-a"])
    records = read_predictions(run_dir)
    assert {r["status"] for r in records} == {"answered", "execution_failed"}
    assert all(r["no_evidence_reason"] is None for r in records)
    assert result.summary["no_evidence_by_reason"] == {"no_text_chunks": 0, "no_text_content": 0, "no_retrieval_tokens": 0, "no_hits": 0}


def test_run_config_states_that_no_retrieval_tokens_is_a_tokeniser_limit(project: Project) -> None:
    """The config keeps a tokeniser limitation apart from an OCR condition."""
    _, run_dir = generate(project)
    reasons = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["retrieval"]["no_evidence_reasons"]
    assert set(reasons) == {"no_text_chunks", "no_text_content", "no_retrieval_tokens", "no_hits"}
    assert "not missing OCR" in reasons["no_text_content"]
    assert "limitation of the engineering tokeniser, not an OCR condition" in reasons["no_retrieval_tokens"]
    assert "Han-only" in reasons["no_retrieval_tokens"]
    assert "OCR condition" in reasons["no_text_chunks"]


def test_exactly_one_terminal_record_per_question_in_manifest_order(project: Project) -> None:
    """T1: ids in the predictions equal ids in the manifest, in order, each with a terminal status."""
    _, run_dir = generate(project)
    records = read_predictions(run_dir)
    assert [r["question_id"] for r in records] == [q["question_id"] for q in project.questions]
    assert {r["status"] for r in records} <= set(pr.TERMINAL_STATUSES)


def test_a_repeated_question_id_is_refused(project: Project) -> None:
    """T1: a duplicate id cannot enter the run."""
    project.questions.append(dict(project.questions[0]))
    project.write()
    with pytest.raises(RunnerRefusal, match="repeats question_id"):
        generate(project)


def test_a_question_for_an_undeclared_document_is_refused(project: Project) -> None:
    """T1: a question cannot name a document outside the declared page inventory."""
    project.questions.append({"question_id": "q-z", "doc_id": "law/none", "question": QUESTION_TEXT})
    project.write()
    with pytest.raises(RunnerRefusal, match="does not declare"):
        generate(project)


def test_the_terminal_record_check_rejects_duplicates_and_omissions() -> None:
    """T1: the guard behind generation raises on a dropped, repeated or reordered record."""
    questions = [pr.RuntimeQuestion(f"q{i}", "d", "t") for i in range(3)]

    def record(qid: str, status: str = "answered") -> dict[str, Any]:
        return {"question_id": qid, "status": status}

    pr.check_terminal_records(questions, [record("q0"), record("q1"), record("q2", "no_evidence")])
    for bad in (
        [record("q0"), record("q1")],
        [record("q0"), record("q1"), record("q1")],
        [record("q0"), record("q2"), record("q1")],
        [record("q0"), record("q1"), record("q2", "pending")],
    ):
        with pytest.raises(RuntimeError):
            pr.check_terminal_records(questions, bad)


class RaisingBackend:
    """Fails on questions that contain 'BOOM' and answers the rest with the top chunk."""

    def identity(self) -> dict[str, Any]:
        return {"name": "raising_test_backend", "engineering_only": True}

    def answer(self, question: str, hits: Any) -> dict[str, Any]:
        if "BOOM" in question:
            raise ValueError("backend exploded")
        return {"answer": hits[0].chunk.text[:20], "answer_mode": "test"}


def test_an_exception_in_one_question_becomes_a_failure_record_and_the_run_continues(project: Project) -> None:
    """T2: a backend exception is an execution_failed record with stage, type and message."""
    project.questions[1] = {"question_id": "q-b", "doc_id": DOC_B, "question": "BOOM " + QUESTION_TEXT}
    project.write()
    result, run_dir = generate(project, backend=RaisingBackend())
    records = {r["question_id"]: r for r in read_predictions(run_dir)}
    failed = records["q-b"]
    assert failed["status"] == "execution_failed"
    assert failed["answer"] is None and failed["abstained"] is False
    assert failed["failure"] == {"stage": "answer", "type": "ValueError", "message": "backend exploded"}
    assert failed["evidence"], "retrieval finished before the answer stage failed, so its evidence is kept"
    assert records["q-a"]["status"] == records["q-a2"]["status"] == "answered"
    assert result.exit_code == pr.EXIT_EXECUTION_FAILED


def test_a_backend_that_returns_the_wrong_type_is_a_failure_record(project: Project) -> None:
    """T2: a malformed backend result is recorded, not trusted."""

    class BadBackend(RaisingBackend):
        def answer(self, question: str, hits: Any) -> dict[str, Any]:
            return {"answer": 42, "answer_mode": None}

    result, run_dir = generate(project, backend=BadBackend())
    assert {r["failure"]["type"] for r in read_predictions(run_dir)} == {"TypeError"}
    assert result.summary["counts"]["execution_failed"] == 3


def test_a_retrieval_hit_from_another_document_is_a_failure_record(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1: if a hit ever escapes its document, the runner records a failure instead of using it."""
    from faar.retrieval import HybridRetriever

    real = HybridRetriever.retrieve

    def leaky(self: Any, query: str, top_k: int | None = None) -> list[Any]:
        hits = real(self, query, top_k)
        hits[0].chunk.doc_name = "law/elsewhere"
        return hits

    monkeypatch.setattr(HybridRetriever, "retrieve", leaky)
    result, run_dir = generate(project)
    records = read_predictions(run_dir)
    assert {r["failure"]["type"] for r in records} == {"RetrievalScopeError"}
    assert {r["failure"]["stage"] for r in records} == {"retrieve"}
    assert result.exit_code == pr.EXIT_EXECUTION_FAILED


def test_an_injected_failure_stays_in_predictions_summary_config_and_fingerprint(project: Project) -> None:
    """T3, T4: an injected failure is a record, a count, a config entry and part of the fingerprint."""
    baseline, _ = generate(project, "run-base")
    result, run_dir = generate(project, "run-inject", inject_failures=["q-b", "q-a2:retrieve"])
    records = {r["question_id"]: r for r in read_predictions(run_dir)}
    assert len(records) == 3
    assert records["q-b"]["failure"] == {"stage": "answer", "type": "InjectedFailure", "message": records["q-b"]["failure"]["message"]}
    assert records["q-a2"]["failure"]["stage"] == "retrieve"
    assert records["q-a"]["status"] == "answered"
    assert result.summary["counts"]["execution_failed"] == 2
    assert result.summary["execution_failures_by_stage"] == {"answer": 1, "retrieve": 1}
    assert result.summary["status"] == "complete_with_execution_failures"
    assert result.summary["exit_code"] == result.exit_code == pr.EXIT_EXECUTION_FAILED
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["injected_failures"] == [{"question_id": "q-a2", "stage": "retrieve"}, {"question_id": "q-b", "stage": "answer"}]
    assert config["fingerprint"] != baseline.summary["fingerprint"]


@pytest.mark.parametrize("stage", ["load", "retrieve", "answer"])
def test_an_injected_failure_fires_at_every_stage_even_for_no_evidence_documents(tmp_path: Path, stage: str) -> None:
    """T3: the injection is never silently skipped."""
    project = Project(tmp_path / "p", {"law/blank": [""]}, [{"question_id": "q1", "doc_id": "law/blank", "question": QUESTION_TEXT}]).write()
    result, run_dir = generate(project, inject_failures=[f"q1:{stage}"])
    (record,) = read_predictions(run_dir)
    assert record["status"] == "execution_failed"
    assert record["failure"]["stage"] == stage
    assert result.exit_code == pr.EXIT_EXECUTION_FAILED


def test_injecting_a_failure_for_an_unknown_question_or_stage_is_refused(project: Project) -> None:
    """T3: a typo cannot turn into a silent no-op."""
    with pytest.raises(RunnerRefusal, match="no runtime question"):
        generate(project, inject_failures=["missing-id"])
    with pytest.raises(RunnerRefusal, match="stage must be one of"):
        generate(project, "run-two", inject_failures=["q-a:parse"])
    with pytest.raises(RunnerRefusal, match="more than once"):
        generate(project, "run-three", inject_failures=["q-a", "q-a:load"])


# ---------------------------------------------------------------------------
# I1, I2, I3
# ---------------------------------------------------------------------------


def test_a_changed_mineru_file_is_refused_and_nothing_is_written(project: Project) -> None:
    """I1: a file that differs from the manifest hash stops the run."""
    path = project.mineru_path(DOC_A)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="the manifest says"):
        generate(project)
    assert not (project.root / "results" / "engineering").exists()


def test_a_changed_page_hash_is_refused(project: Project) -> None:
    """I1: a per-page hash that no longer matches stops the run even when the file hash was updated."""
    payload = json.loads(project.runtime_manifest.read_text(encoding="utf-8"))
    payload["documents"][0]["pages"][0]["ocr_text_sha256"] = "f" * 64
    project.runtime_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="page 0 reads as"):
        generate(project)


def test_a_page_status_that_disagrees_with_the_text_is_refused(project: Project) -> None:
    """I1: a page the manifest calls empty but the file fills is refused."""
    payload = json.loads(project.runtime_manifest.read_text(encoding="utf-8"))
    payload["documents"][0]["pages"][0]["ocr_status"] = "empty"
    project.runtime_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunnerRefusal):
        generate(project)


def test_page_parsing_and_hashing_match_the_pilot_builder(tmp_path: Path) -> None:
    """I2: run the builder's own inventory reader and hash rule on a file with duplicates, bad rows and non-string text."""
    import audit_assets

    rows = [
        {"page_idx": 0, "text": "first"},
        {"page_idx": 0, "text": "again"},
        {"page_idx": 1, "text": None},
        {"page_idx": 2, "text": "  "},
        {"page_idx": 3, "text": "中文 text"},
        {"page_idx": -1, "text": "negative"},
        {"page_idx": True, "text": "bool"},
        {"text": "no index"},
        "not an object",
    ]
    path = tmp_path / "mineru.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    builder = audit_assets.load_page_inventory(path)["pages"]
    ours = pr.parse_page_inventory(json.loads(path.read_text(encoding="utf-8")))
    assert ours == builder
    assert ours[0] == "first\nagain"
    for text in ours.values():
        assert pr.page_text_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_frozen_inputs_are_unchanged_by_generation_and_scoring(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """I3: hashes of the runtime manifest, MinerU files and evaluation manifest do not move."""
    install_fake_scoring(monkeypatch)
    before = {p: sha256_bytes(p.read_bytes()) for p in project.frozen_files()}
    stat_before = {p: p.stat().st_mtime_ns for p in project.frozen_files()}
    _, run_dir = generate(project)
    pr.score_run(project_root=project.root, run_dir=run_dir)
    assert {p: sha256_bytes(p.read_bytes()) for p in project.frozen_files()} == before
    assert {p: p.stat().st_mtime_ns for p in project.frozen_files()} == stat_before


# ---------------------------------------------------------------------------
# D1, D2, D3, D4
# ---------------------------------------------------------------------------


def test_a_compatible_rerun_is_a_verified_no_op(project: Project) -> None:
    """D1: same inputs, same code, same run directory. Nothing is rewritten."""
    first, run_dir = generate(project)
    before = snapshot(run_dir)
    second, _ = generate(project)
    assert first.wrote and not second.wrote
    assert "already complete, verified identical" in second.message
    assert second.exit_code == pr.EXIT_OK
    assert snapshot(run_dir) == before


def test_a_compatible_rerun_of_a_run_with_failures_keeps_the_failure_exit_code(project: Project) -> None:
    """D1, T4: verifying a run that had execution failures still exits with the failure code."""
    generate(project, inject_failures=["q-a"])
    again, _ = generate(project, inject_failures=["q-a"])
    assert not again.wrote
    assert again.exit_code == pr.EXIT_EXECUTION_FAILED


def test_a_run_directory_holds_a_new_run_only_when_it_is_empty_or_missing(project: Project) -> None:
    """D1: an existing empty directory is accepted as a new run directory."""
    run_dir = project.root / "results" / "engineering" / "run-empty"
    run_dir.mkdir(parents=True)
    result = pr.generate_run(project_root=project.root, run_dir=run_dir, pilot_id=PILOT_ID)
    assert result.wrote
    assert sorted(p.name for p in run_dir.iterdir()) == ["generation_summary.json", "predictions.jsonl", "run_config.json"]


def test_a_run_with_a_different_fingerprint_is_refused(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """D2: changed settings, changed code and changed inputs each refuse reuse of the directory."""
    _, run_dir = generate(project)
    before = snapshot(run_dir)
    with pytest.raises(RunnerRefusal, match="never overwrites"):
        generate(project, inject_failures=["q-a"])
    with pytest.raises(RunnerRefusal, match="never overwrites"):
        generate(project, settings=pr.RetrievalSettings(chunk_size_words=50, chunk_overlap_words=5, top_k=2, embedding_backend="local-hash-v1"))
    monkeypatch.setattr(pr, "_measurement_code_digest", lambda: "changed-code")
    with pytest.raises(RunnerRefusal, match="never overwrites"):
        generate(project)
    monkeypatch.undo()
    project.questions.append({"question_id": "q-new", "doc_id": DOC_A, "question": "New?"})
    project.write()
    with pytest.raises(RunnerRefusal, match="never overwrites"):
        generate(project)
    assert snapshot(run_dir) == before


def test_a_changed_cli_script_changes_the_fingerprint(project: Project, tmp_path: Path) -> None:
    """D2: the CLI script hash is part of the fingerprint."""
    script = tmp_path / "script.py"
    script.write_text("print(1)", encoding="utf-8")
    first, _ = generate(project, "run-s1", cli_script=script)
    script.write_text("print(2)", encoding="utf-8")
    second, _ = generate(project, "run-s2", cli_script=script)
    assert first.summary["fingerprint"] != second.summary["fingerprint"]


def test_a_reused_directory_with_another_run_id_is_refused(project: Project) -> None:
    """D2: the same fingerprint under a different run_id is a different record."""
    _, run_dir = generate(project)
    with pytest.raises(RunnerRefusal, match="names 'other-id'"):
        pr.generate_run(project_root=project.root, run_dir=run_dir, run_id="other-id", pilot_id=PILOT_ID)


@pytest.mark.parametrize("victim", ["generation_summary.json", "run_config.json", "predictions.jsonl"])
def test_a_partial_run_directory_is_refused(project: Project, victim: str) -> None:
    """D2: a directory missing any of the three files is kept as it is."""
    _, run_dir = generate(project)
    (run_dir / victim).unlink()
    before = snapshot(run_dir)
    with pytest.raises(RunnerRefusal, match="partial run"):
        generate(project)
    assert snapshot(run_dir) == before


def test_a_corrupt_run_directory_is_refused(project: Project) -> None:
    """D2: predictions that no longer match the summary hash are refused."""
    _, run_dir = generate(project)
    with (run_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RunnerRefusal, match="corrupt"):
        generate(project)


def test_a_directory_with_unrecognised_files_is_refused(project: Project) -> None:
    """D2: an unrelated non-empty directory is not a place for a run."""
    run_dir = project.root / "results" / "engineering" / "run-junk"
    run_dir.mkdir(parents=True)
    (run_dir / "notes.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="unrecognised"):
        generate(project, "run-junk")
    assert [p.name for p in run_dir.iterdir()] == ["notes.txt"]


def test_a_stray_temporary_file_marks_the_directory_partial(project: Project) -> None:
    """D2: the leftover of an interrupted atomic write is refused, not cleaned up."""
    run_dir = project.root / "results" / "engineering" / "run-crash"
    run_dir.mkdir(parents=True)
    (run_dir / ".predictions.jsonl.abc.tmp").write_text("half", encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="unrecognised"):
        generate(project, "run-crash")


def test_a_nondeterministic_rerun_is_refused_and_writes_nothing(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """D3: same fingerprint but different regenerated bytes."""
    _, run_dir = generate(project)
    before = snapshot(run_dir)
    real = pr.generate_records

    def drifting(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        records = real(*args, **kwargs)
        records[0]["answer"] = "drifted"
        return records

    monkeypatch.setattr(pr, "generate_records", drifting)
    with pytest.raises(RunnerRefusal, match="nondeterministic"):
        generate(project)
    assert snapshot(run_dir) == before


def test_two_fresh_runs_give_identical_predictions(project: Project) -> None:
    """D3: generation is deterministic on the same inputs."""
    _, dir_a = generate(project, "run-a1")
    _, dir_b = generate(project, "run-b1")
    assert (dir_a / "predictions.jsonl").read_bytes() == (dir_b / "predictions.jsonl").read_bytes()
    assert (dir_a / "generation_summary.json").read_bytes().replace(b"run-a1", b"X") == (
        dir_b / "generation_summary.json"
    ).read_bytes().replace(b"run-b1", b"X")


@pytest.mark.parametrize("relative", ["results/pilots/fix_v1/run-x", "results/pilots", "results/pilots/other/deep/run-y"])
def test_a_run_directory_under_results_pilots_is_refused(project: Project, relative: str) -> None:
    """D4: generation never writes into the frozen pilot directories."""
    run_dir = project.root / relative
    existed = run_dir.exists()
    with pytest.raises(RunnerRefusal, match="results/pilots"):
        pr.generate_run(project_root=project.root, run_dir=run_dir, run_id="run-x", pilot_id=PILOT_ID)
    assert run_dir.exists() == existed


def test_a_results_pilots_path_outside_the_project_root_is_also_refused(project: Project, tmp_path: Path) -> None:
    """D4: the rule follows the path components, not only the project root."""
    with pytest.raises(RunnerRefusal, match="results/pilots"):
        pr.generate_run(project_root=project.root, run_dir=tmp_path / "elsewhere" / "results" / "pilots" / "run-z", pilot_id=PILOT_ID)


@pytest.mark.parametrize("relative", ["results/Pilots/ohr_dev_v1/case-run", "RESULTS/pilots/x", "Results/PILOTS"])
def test_a_case_variant_of_results_pilots_is_refused(project: Project, relative: str) -> None:
    """D4: a case variant reaches the frozen directory on a case-insensitive filesystem, so it is refused too."""
    with pytest.raises(RunnerRefusal, match="results/pilots"):
        pr.generate_run(project_root=project.root, run_dir=project.root / relative, run_id="case-run", pilot_id=PILOT_ID)
    assert not (project.root / "results" / "pilots" / "ohr_dev_v1" / "case-run").exists()


@pytest.mark.parametrize(
    "relative", ["config/rogue", "OHR-Bench/data/rogue", "annotation/rogue", "logs/rogue", "experiments/rogue", "results/rogue", "rogue"]
)
def test_a_run_directory_inside_the_project_must_be_under_results_engineering(project: Project, relative: str) -> None:
    """D4: inside the project, only results/engineering/ may receive a run, so no frozen or shared directory can."""
    run_dir = project.root / relative
    existed = run_dir.exists()
    with pytest.raises(RunnerRefusal, match="results/engineering"):
        pr.generate_run(project_root=project.root, run_dir=run_dir, run_id="rogue", pilot_id=PILOT_ID)
    assert run_dir.exists() == existed


def test_a_run_directory_outside_the_project_is_allowed(project: Project, tmp_path: Path) -> None:
    """D4: scratch runs outside the project root stay possible."""
    result = pr.generate_run(project_root=project.root, run_dir=tmp_path / "scratch" / "run-s", pilot_id=PILOT_ID)
    assert result.exit_code == pr.EXIT_OK


@pytest.mark.parametrize("root", ["OHR-Bench/data/retrieval_base", "OHR-Bench/data/retrieval_base/gt", "OHR-Bench/data"])
def test_a_manifest_that_declares_another_noisy_text_root_is_refused(project: Project, root: str) -> None:
    """L3: generation reads noisy text only from the MinerU tree, so a manifest cannot redirect it to gt text."""
    manifest = json.loads(project.runtime_manifest.read_text(encoding="utf-8"))
    manifest["sources"]["noisy_text"]["root"] = root
    project.runtime_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="noisy text root"):
        generate(project)


def test_questions_without_a_retrieval_token_are_counted(tmp_path: Path) -> None:
    """R3: a Han-script question gets no lexical or hashed signal, so its record and the summary say so."""
    project = Project(
        tmp_path / "p",
        {DOC_A: [WARRANTY_A]},
        [
            {"question_id": "q-latin", "doc_id": DOC_A, "question": QUESTION_TEXT},
            {"question_id": "q-han", "doc_id": DOC_A, "question": "保修期是多久？"},
        ],
    ).write()
    result, run_dir = generate(project)
    records = {r["question_id"]: r for r in read_predictions(run_dir)}
    assert records["q-han"]["status"] == "answered"
    assert records["q-han"]["query_retrieval_tokens"] == 0
    assert records["q-latin"]["query_retrieval_tokens"] > 0
    assert result.summary["questions_without_query_tokens"] == {"total": 1, "answered": 1, "no_evidence": 0, "execution_failed": 0}


def test_an_invalid_run_id_is_refused(project: Project) -> None:
    """D4: the run id follows the registry pattern so a later registration accepts it."""
    with pytest.raises(RunnerRefusal, match="run_id"):
        pr.generate_run(project_root=project.root, run_dir=project.root / "out" / "Bad Name", pilot_id=PILOT_ID)


# ---------------------------------------------------------------------------
# O1, O2
# ---------------------------------------------------------------------------

RECORD_KEYS = [
    "schema_version", "question_id", "doc_id", "status", "answer", "abstained", "no_evidence_reason",
    "query_retrieval_tokens", "answer_mode", "evidence", "ocr_condition", "failure",
]
EVIDENCE_KEYS = ["rank", "chunk_id", "doc_id", "page_idx", "fused_score", "bm25_score", "dense_score"]
OCR_KEYS = ["pages_total", "pages_ok", "pages_empty", "pages_missing", "chunks"]


def test_prediction_records_follow_the_contract(tmp_path: Path) -> None:
    """O1: keys, key order, types and values for answered, no_evidence and execution_failed records."""
    project = Project(
        tmp_path / "p",
        {DOC_A: [WARRANTY_A], "law/blank": [""]},
        [
            {"question_id": "q-ans", "doc_id": DOC_A, "question": QUESTION_TEXT},
            {"question_id": "q-none", "doc_id": "law/blank", "question": QUESTION_TEXT},
            {"question_id": "q-fail", "doc_id": DOC_A, "question": QUESTION_TEXT},
        ],
    ).write()
    _, run_dir = generate(project, inject_failures=["q-fail"])
    answered, none, failed = read_predictions(run_dir)
    for record in (answered, none, failed):
        assert list(record) == RECORD_KEYS
        assert record["schema_version"] == 1
        assert list(record["ocr_condition"]) == OCR_KEYS
    assert answered["status"] == "answered" and isinstance(answered["answer"], str) and answered["abstained"] is False
    assert isinstance(answered["answer_mode"], str)
    assert [list(e) for e in answered["evidence"]] == [EVIDENCE_KEYS] * len(answered["evidence"])
    assert [e["rank"] for e in answered["evidence"]] == list(range(1, len(answered["evidence"]) + 1))
    assert all(isinstance(e["page_idx"], int) and isinstance(e["fused_score"], float) for e in answered["evidence"])
    assert (none["answer"], none["abstained"], none["answer_mode"]) == ("", True, None)
    assert (failed["answer"], failed["abstained"], failed["answer_mode"]) == (None, False, None)
    assert list(failed["failure"]) == ["stage", "type", "message"]


def test_output_files_are_strict_json(project: Project) -> None:
    """O1: no NaN or Infinity token in any output file."""
    _, run_dir = generate(project)
    for name in ("predictions.jsonl", "run_config.json", "generation_summary.json"):
        text = (run_dir / name).read_text(encoding="utf-8")
        json.loads(text.splitlines()[0]) if name.endswith("jsonl") else json.loads(text)
        assert "NaN" not in text and "Infinity" not in text


def test_run_config_records_provenance_and_settings(project: Project, tmp_path: Path) -> None:
    """O2: commit, dirty paths, hashes, retrieval settings, backend identity, injected failures and command."""
    repo = tmp_path / "code-repo"
    repo.mkdir()
    git = lambda *args: subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "--allow-empty", "-q", "-m", "init")
    (repo / "tracked.txt").write_text("x", encoding="utf-8")
    git("add", "tracked.txt")
    git("-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-q", "-m", "add")
    (repo / "tracked.txt").write_text("changed", encoding="utf-8")
    (repo / "untracked.txt").write_text("u", encoding="utf-8")
    run_dir = repo / "results" / "engineering" / "run-git"
    result = pr.generate_run(
        project_root=project.root,
        run_dir=run_dir,
        pilot_id=PILOT_ID,
        inject_failures=["q-a"],
        code_root=repo,
        command=["scripts/experiments/run_pilot_offline.py", "generate", "--run-dir", "x"],
    )
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["run_id"] == "run-git" and config["pilot_id"] == PILOT_ID and config["kind"] == "engineering_check"
    assert config["code"]["commit"] == git("rev-parse", "HEAD").stdout.strip()
    assert config["code"]["dirty"] is True
    assert config["code"]["dirty_paths"] == ["tracked.txt", "untracked.txt"], "the run directory itself is not listed"
    assert config["fingerprint"] == result.summary["fingerprint"]
    assert config["runtime_manifest"]["sha256"] == sha256_bytes(project.runtime_manifest.read_bytes())
    docs = {d["doc_id"]: d for d in config["documents"]}
    assert docs[DOC_A]["noisy_text_sha256"] == sha256_bytes(project.mineru_path(DOC_A).read_bytes())
    assert docs[DOC_A]["pages_ok"] == 2 and docs[DOC_A]["chunks"] == 2
    retrieval = config["retrieval"]
    assert retrieval["embedding_backend"] == "local-hash-v1"
    assert (retrieval["chunking"]["chunk_size_words"], retrieval["chunking"]["chunk_overlap_words"], retrieval["top_k"]) == (180, 40, 5)
    assert retrieval["tokenisation"]["retrieval_tokeniser"] == "[a-z0-9%$]+"
    assert "Han" in retrieval["tokenisation"]["han_script"]
    assert "stable argsort" in retrieval["ranking"]["tie_breaking"]
    assert "not the approved scientific protocol" in retrieval["label"]
    assert config["answer_backend"]["engineering_only"] is True and config["answer_backend"]["model_calls"] is False
    assert config["injected_failures"] == [{"question_id": "q-a", "stage": "answer"}]
    assert config["command"][1] == "generate"
    assert "evaluation_manifest.json" in config["inputs_never_read"]


def test_run_config_records_no_absolute_path_under_the_project_root(project: Project) -> None:
    """O2: committed run records name paths relative to the project, so they carry no home directory."""
    run_dir = project.root / "results" / "engineering" / "run-relative"
    pr.generate_run(project_root=project.root, run_dir=run_dir, pilot_id=PILOT_ID, code_root=project.root)
    text = (run_dir / "run_config.json").read_text(encoding="utf-8")
    config = json.loads(text)
    assert config["code"]["code_root"] == "."
    assert str(project.root.resolve()) not in text


def test_git_provenance_is_null_when_unknown(tmp_path: Path) -> None:
    """O2: no repository means unknown values, recorded as null."""
    assert pr.git_provenance(None) == {"commit": None, "dirty": None, "dirty_paths": None}
    assert pr.git_provenance(tmp_path) == {"commit": None, "dirty": None, "dirty_paths": None}


def test_the_runner_writes_no_experiment_registry(project: Project) -> None:
    """The runner leaves experiments/registry.jsonl to the lead."""
    generate(project)
    assert not (project.root / "experiments").exists()


# ---------------------------------------------------------------------------
# S1, S2
# ---------------------------------------------------------------------------


def install_fake_scoring(monkeypatch: pytest.MonkeyPatch, *, calls: list[str] | None = None, version: str = "1") -> types.ModuleType:
    """Put a stand-in for faar.ohr_scoring in sys.modules. It follows the contract signatures and is not a scorer."""
    module = types.ModuleType("faar.ohr_scoring")

    def scorer_identity() -> dict[str, str]:
        return {
            "name": "fake_scorer",
            "upstream_repo": "none",
            "upstream_commit": "none",
            "upstream_path": "none",
            "upstream_sha256": "0" * 64,
            "adapter_version": version,
        }

    def score_predictions(predictions: list[dict[str, Any]], evaluation_questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if calls is not None:
            calls.append("score_predictions")
        ids = [p["question_id"] for p in predictions]
        if len(set(ids)) != len(ids) or set(ids) != set(evaluation_questions):
            raise ValueError("prediction ids and evaluation ids differ or repeat")
        rows = []
        for p in predictions:
            if p["status"] == "execution_failed":
                rows.append({"question_id": p["question_id"], "status": p["status"], "em": 0, "f1": 0.0, "scored_by": "failure_as_zero"})
            else:
                em = int(p["answer"] == evaluation_questions[p["question_id"]]["answers"])
                rows.append({"question_id": p["question_id"], "status": p["status"], "em": em, "f1": float(em), "scored_by": "official"})
        counts = {s: sum(p["status"] == s for p in predictions) for s in ("answered", "no_evidence", "execution_failed")}
        n = len(predictions)
        return {
            "scorer": scorer_identity(),
            "rows": rows,
            "counts": {"questions": n, **counts, "abstained": counts["no_evidence"]},
            "aggregates": {
                "all_questions": {"denominator": n, "em": sum(r["em"] for r in rows) / n, "f1": sum(r["f1"] for r in rows) / n, "policy": "fake"},
                "answered_only": {"denominator": counts["answered"], "em": None, "f1": None},
            },
        }

    module.scorer_identity = scorer_identity  # type: ignore[attr-defined]
    module.scorer_dependencies = lambda: {"jieba": "fake", "regex": "fake"}  # type: ignore[attr-defined]
    module.score_predictions = score_predictions  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faar.ohr_scoring", module)
    return module


def test_scoring_writes_scores_and_a_summary_with_identities(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: scores.jsonl and score_summary.json carry the scorer identity and the evaluation manifest hash."""
    install_fake_scoring(monkeypatch)
    gen, run_dir = generate(project)
    result = pr.score_run(project_root=project.root, run_dir=run_dir)
    assert result.wrote and result.exit_code == pr.EXIT_OK
    rows = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["question_id"] for r in rows] == [q["question_id"] for q in project.questions]
    summary = json.loads((run_dir / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["scorer"]["name"] == "fake_scorer"
    assert summary["evaluation_manifest"]["sha256"] == sha256_bytes(project.evaluation_manifest.read_bytes())
    assert summary["generation_fingerprint"] == gen.summary["fingerprint"]
    assert summary["predictions_sha256"] == gen.summary["predictions_sha256"]
    assert summary["counts"]["questions"] == 3
    assert summary["scorer_environment"]["jieba"] == "fake" and summary["scorer_environment"]["regex"] == "fake"
    assert summary["scorer_environment"]["unicodedata"] == unicodedata.unidata_version


def test_scoring_with_the_real_scorer_keeps_every_question(project: Project) -> None:
    """S1, S2: the real faar.ohr_scoring joins the saved predictions; failures stay in the denominator."""
    pytest.importorskip("jieba")
    _, run_dir = generate(project, inject_failures=["q-a"])
    result = pr.score_run(project_root=project.root, run_dir=run_dir)
    assert result.exit_code == pr.EXIT_EXECUTION_FAILED
    summary = json.loads((run_dir / "score_summary.json").read_text(encoding="utf-8"))
    assert summary["scorer"]["name"] == "ohr-bench-official-qa"
    assert summary["counts"]["questions"] == 3 and summary["counts"]["execution_failed"] == 1
    assert summary["aggregates"]["all_questions"]["denominator"] == 3
    assert set(summary["scorer_environment"]) >= {"jieba", "regex", "unicodedata"}


def test_scoring_never_overwrites_and_verifies_identical_scores(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: a second scoring with the same result is a no-op. A different result is refused."""
    install_fake_scoring(monkeypatch)
    _, run_dir = generate(project)
    pr.score_run(project_root=project.root, run_dir=run_dir)
    before = snapshot(run_dir)
    again = pr.score_run(project_root=project.root, run_dir=run_dir)
    assert not again.wrote and "already scored, verified identical" in again.message
    install_fake_scoring(monkeypatch, version="2")
    with pytest.raises(RunnerRefusal, match="differ from a fresh scoring"):
        pr.score_run(project_root=project.root, run_dir=run_dir)
    assert snapshot(run_dir) == before


def test_partial_score_files_are_refused(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: one score file without the other is kept and refused."""
    install_fake_scoring(monkeypatch)
    _, run_dir = generate(project)
    (run_dir / "scores.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RunnerRefusal, match="partial"):
        pr.score_run(project_root=project.root, run_dir=run_dir)


def test_scoring_keeps_execution_failures_visible(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: an execution_failed prediction is scored by the scorer's failure rule, counted, and reflected in the exit code."""
    install_fake_scoring(monkeypatch)
    _, run_dir = generate(project, inject_failures=["q-a"])
    result = pr.score_run(project_root=project.root, run_dir=run_dir)
    assert result.exit_code == pr.EXIT_EXECUTION_FAILED
    rows = {json.loads(line)["question_id"]: json.loads(line) for line in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()}
    assert rows["q-a"]["status"] == "execution_failed" and rows["q-a"]["scored_by"] == "failure_as_zero"
    assert result.summary["counts"]["execution_failed"] == 1


def test_scoring_refuses_a_join_the_scorer_rejects(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: an evaluation manifest with other question ids stops scoring and writes nothing."""
    install_fake_scoring(monkeypatch)
    _, run_dir = generate(project)
    project.write_evaluation({"other": {"answers": "x"}})
    with pytest.raises(RunnerRefusal, match="scoring refused the join"):
        pr.score_run(project_root=project.root, run_dir=run_dir)
    assert not (run_dir / "scores.jsonl").exists()


def test_scoring_refuses_an_evaluation_manifest_for_another_pilot(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: pilot ids must match."""
    install_fake_scoring(monkeypatch)
    _, run_dir = generate(project)
    project.write_evaluation({}, raw=json.dumps({"pilot_id": "other", "questions": {}}))
    with pytest.raises(RunnerRefusal, match="is for pilot"):
        pr.score_run(project_root=project.root, run_dir=run_dir)


def test_scoring_refuses_when_the_scoring_module_is_missing(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: the scoring step names the missing module instead of scoring with something else."""
    _, run_dir = generate(project)
    monkeypatch.setitem(sys.modules, "faar.ohr_scoring", None)
    with pytest.raises(RunnerRefusal, match="faar.ohr_scoring is not available"):
        pr.score_run(project_root=project.root, run_dir=run_dir)


def test_scoring_needs_a_complete_generation_run(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S1: scoring does not run on a missing or partial run directory."""
    install_fake_scoring(monkeypatch)
    with pytest.raises(RunnerRefusal, match="no complete generation run"):
        pr.score_run(project_root=project.root, run_dir=project.root / "results" / "engineering" / "absent")
    _, run_dir = generate(project)
    (run_dir / "generation_summary.json").unlink()
    with pytest.raises(RunnerRefusal, match="partial run"):
        pr.score_run(project_root=project.root, run_dir=run_dir)


def test_generation_never_calls_the_scorer(project: Project, monkeypatch: pytest.MonkeyPatch) -> None:
    """S2: generation does not import or call the scoring module."""
    calls: list[str] = []
    install_fake_scoring(monkeypatch, calls=calls)
    generate(project)
    assert calls == []


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------


def cli_args(project: Project, name: str, *extra: str) -> list[str]:
    return [
        "generate",
        "--project-root",
        str(project.root),
        "--pilot-id",
        PILOT_ID,
        "--run-dir",
        str(project.root / "results" / "engineering" / name),
        *extra,
    ]


def test_cli_exit_codes_separate_success_failure_and_refusal(project: Project, capsys: pytest.CaptureFixture[str]) -> None:
    """C1, T4: 0 for a clean run, 2 for execution failures, 1 for a refusal."""
    assert cli.main(cli_args(project, "run-c0")) == 0
    assert "3 questions: 3 answered, 0 no_evidence, 0 execution_failed" in capsys.readouterr().out
    assert cli.main(cli_args(project, "run-c2", "--inject-failure", "q-a")) == 2
    assert "Status: complete_with_execution_failures" in capsys.readouterr().out
    assert cli.main(cli_args(project, "run-c0", "--inject-failure", "q-a")) == 1
    assert "refused:" in capsys.readouterr().err
    assert cli.main(cli_args(project, "run-c0")) == 0
    assert "already complete, verified identical" in capsys.readouterr().out
    assert cli.main([*cli_args(project, "run-c3")[:-1], str(project.root / "results" / "pilots" / "run-c3")]) == 1


def test_cli_reports_an_unexpected_error_with_its_own_exit_code(
    project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """C1: an internal error is not reported as a clean refusal."""

    def broken(**kwargs: Any) -> None:
        raise KeyError("pilot_id")

    monkeypatch.setattr(cli.pilot_runner, "generate_run", broken)
    assert cli.main(cli_args(project, "run-err")) == 3
    assert "KeyError" in capsys.readouterr().err


def test_cli_usage_errors_exit_with_the_refusal_code(capsys: pytest.CaptureFixture[str]) -> None:
    """C1: argparse errors use code 1, not argparse's default 2, which means execution failures here."""
    for argv in ([], ["generate"], ["bogus"], ["generate", "--run-dir"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 1
    capsys.readouterr()


def test_cli_help_documents_exit_codes_and_the_overwrite_policy(capsys: pytest.CaptureFixture[str]) -> None:
    """C1: --help states the exit codes, the overwrite policy and the injection option."""
    for argv in (["--help"], ["generate", "--help"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 0
        text = capsys.readouterr().out
        assert "exit codes" in text and "execution_failed" in text
        assert "overwrite policy" in text and "verified" in text
        assert "results/pilots/" in text
    with pytest.raises(SystemExit):
        cli.main(["generate", "--help"])
    assert "--inject-failure" in capsys.readouterr().out


def test_cli_scores_a_generated_run(project: Project, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """C1: the score subcommand runs against a saved run and exits 0."""
    install_fake_scoring(monkeypatch)
    assert cli.main(cli_args(project, "run-sc")) == 0
    run_dir = str(project.root / "results" / "engineering" / "run-sc")
    assert cli.main(["score", "--project-root", str(project.root), "--run-dir", run_dir]) == 0
    assert "scores written" in capsys.readouterr().out
    assert cli.main(["score", "--project-root", str(project.root), "--run-dir", run_dir]) == 0
    assert "already scored, verified identical" in capsys.readouterr().out


def test_cli_refuses_a_faar_package_from_another_checkout(project: Project, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The fingerprint hashes the imported package, so a mismatched checkout is refused."""
    monkeypatch.setattr(cli, "SRC", tmp_path / "other" / "src")
    assert cli.main(cli_args(project, "run-x1")) == 1
