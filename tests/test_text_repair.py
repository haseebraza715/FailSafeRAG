"""Tests for the text repairs, written from their failure list.

T1. ``exclude-seen-v1`` returns a chunk the initial answer saw, or more than ``k`` chunks.
T2. Offline retrieval that differs from the recorded evidence is not refused.
T3. Correction changes block order or ids, or miscounts changed blocks.
T4. A question with no text evidence, nothing new to retrieve or nothing corrected is sent.
T5. The repair prompt uses another template than the no-recovery run, or the old evidence.
T6. The shared runner sends a text repair with images, sends a skip record, or exports a skip as a failure.
T7. ByT5 runs without an authorised download.
T8. Prepare opens an evaluation-only file.
T9. The keep-top3 rule loses initial evidence, repeats seen chunks, exceeds the budget, or fails to skip without new evidence.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import run_text_repair as rtr
import run_visual_repair as rvr
from test_visual_repair import Project, ScriptedProvider

from faar import text_repair
from faar.answer_prompt import TEMPLATE_V2
from faar.live_contract import EvidenceBlock
from faar.text_repair import TextRepairError, corrected_blocks, corrector, replay_retrieval, retry_hits


def hit(cid: str, page: int, text: str, score: float = 0.5) -> Any:
    return SimpleNamespace(chunk=SimpleNamespace(chunk_id=cid, doc_name="d/a", page_id=page, text=text), fused_score=score)


DOC_A = [hit("d/a-p1-c0", 1, "Ann signed it."), hit("d/a-p0-c0", 0, "Title page."), hit("d/a-p1-c1", 1, "Date: 4 May."),
         hit("d/a-p3-c0", 3, "Appendix one."), hit("d/a-p2-c0", 2, "Notes."), hit("d/a-p3-c1", 3, "Appendix two.")]


class FakeRetriever:
    def __init__(self, ranked: list[Any]) -> None:
        self.ranked = ranked
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[Any]:
        self.queries.append(query)
        return self.ranked[: top_k or 5]


def fresh(path: Path) -> Project:
    path.mkdir()
    return Project(path)


def fake_replay(project: Project, ranked: list[Any] = DOC_A) -> dict[str, Any]:
    retriever = FakeRetriever(ranked)
    return {"hits": {"q1": ranked[:2], "q2": ranked[:2], "q3": []},
            "indexes": {"d/a": SimpleNamespace(retriever=lambda: retriever), "d/b": SimpleNamespace(retriever=lambda: None)},
            "settings": SimpleNamespace(top_k=2), "tokenizer": "multilingual-v1", "retrieval": {"top_k": 2}, "runtime_sha256": "x"}


def prepare(project: Project, monkeypatch: pytest.MonkeyPatch, repair: str, method: str, **kw: Any) -> list[dict[str, Any]]:
    monkeypatch.setattr(rtr, "replay_retrieval", lambda root, pilot, reqs: fake_replay(project, **kw))
    records, _ = rtr.prepare(root=project.root, repair=repair, method=method, pilot_id="pilot_t", baseline_run_dir=project.baseline,
                             run_dir=project.run_dir(repair), run_id=repair, resend_rule="none", provider_config=project.provider_config)
    return records


# --------------------------------------------------------------------------- T1


def test_exclude_seen_returns_only_unseen_chunks_up_to_k() -> None:
    retriever = FakeRetriever(DOC_A)
    out = retry_hits("exclude-seen-v1", "Who signed?", DOC_A[:2], retriever, 2)
    assert [h.chunk.chunk_id for h in out] == ["d/a-p1-c1", "d/a-p3-c0"]
    assert retriever.queries == ["Who signed?"]
    anchored = retry_hits("append-anchor-words-v0", "Who signed?", DOC_A[:2], FakeRetriever(DOC_A), 2)
    assert len(anchored) == 2
    with pytest.raises(TextRepairError):
        retry_hits("gold-pages", "Who signed?", DOC_A[:2], retriever, 2)


# --------------------------------------------------------------------------- T9


def test_keep_top3_adds_best_unseen_within_budget() -> None:
    initial = DOC_A[:5]
    extra = hit("extra", 4, "Extra.")
    ranked = [DOC_A[4], DOC_A[5], DOC_A[1], extra, DOC_A[0], DOC_A[2], DOC_A[3]]
    retriever = FakeRetriever(ranked)
    out = retry_hits("keep-top3-add-next2-v1", "Who signed?", initial, retriever, 5)
    assert [h.chunk.chunk_id for h in out] == [h.chunk.chunk_id for h in initial[:3]] + [DOC_A[5].chunk.chunk_id, "extra"]
    assert retriever.queries == ["Who signed?"]
    assert retry_hits("keep-top3-add-next2-v1", "Q", initial, FakeRetriever(DOC_A), 5) == initial[:3] + [DOC_A[5]]
    assert retry_hits("keep-top3-add-next2-v1", "Q", initial, FakeRetriever(initial), 5) == []


def test_keep_top3_prepare_skips_when_no_new_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path)
    monkeypatch.setattr(rtr, "replay_retrieval", lambda root, pilot, reqs: {
        **fake_replay(project), "hits": {"q1": DOC_A[:5], "q2": DOC_A[:5], "q3": []},
        "indexes": {"d/a": SimpleNamespace(retriever=lambda: FakeRetriever(DOC_A[:5]))},
        "settings": SimpleNamespace(top_k=5),
    })
    records, identity = rtr.prepare(root=project.root, repair="retry_retrieval", method="keep-top3-add-next2-v1",
                                    pilot_id="pilot_t", baseline_run_dir=project.baseline, run_dir=project.run_dir("retry"),
                                    run_id="retry", resend_rule="none", provider_config=project.provider_config)
    assert records[0]["action"] == "skip" and records[0]["skip_reason"] == "nothing_new_retrieved"
    assert identity["method"] == {"name": "keep-top3-add-next2-v1", "top_k": 5}


# --------------------------------------------------------------------------- T2


def test_replay_refuses_retrieval_that_differs_from_the_recorded_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from faar import pilot_runner

    outcome = SimpleNamespace(question=SimpleNamespace(question_id="q1"), hits=(DOC_A[0],))
    fake_run = SimpleNamespace(questions=(outcome,), loaded_documents={}, retrieval={}, runtime_manifest_sha256="x")
    monkeypatch.setattr(pilot_runner, "retrieve_runtime_questions", lambda **kw: fake_run)
    good = {"q1": {"action": "send", "evidence": [{"rank": 1, "chunk_id": "d/a-p1-c0", "text_sha256": hashlib.sha256(b"Ann signed it.").hexdigest()}]}}
    assert replay_retrieval(tmp_path, "pilot_t", good)["hits"]["q1"] == [DOC_A[0]]
    bad = {"q1": {"action": "send", "evidence": [{"rank": 1, "chunk_id": "d/a-p1-c0", "text_sha256": "0" * 64}]}}
    with pytest.raises(TextRepairError, match="differs"):
        replay_retrieval(tmp_path, "pilot_t", bad)


# --------------------------------------------------------------------------- T3, T7


def test_correction_keeps_order_and_counts_changes() -> None:
    blocks = [EvidenceBlock(1, "c1", "d", 0, "teh cat"), EvidenceBlock(2, "c2", "d", 0, "dog"), EvidenceBlock(3, "c3", "d", 1, "营收")]
    out, changed = corrected_blocks(blocks, lambda t: t.replace("teh", "the"))
    assert [(b.rank, b.chunk_id, b.page_idx) for b in out] == [(1, "c1", 0), (2, "c2", 0), (3, "c3", 1)]
    assert [b.text for b in out] == ["the cat", "dog", "营收"] and changed == 1
    fix, identity = corrector("symspell-en-82765")
    assert fix("营收 2005") == "营收 2005" and identity["dictionary_sha256"]
    with pytest.raises(TextRepairError, match="not been authorised"):
        corrector("byt5-small")


# --------------------------------------------------------------------------- T4, T5


def test_retry_skips_what_it_cannot_serve_and_prompts_with_the_new_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path)
    records = {r["question_id"]: r for r in prepare(project, monkeypatch, "retry_retrieval", "exclude-seen-v1")}
    assert records["q3"]["action"] == "skip" and records["q3"]["skip_reason"] == "no_text_evidence"
    q1 = records["q1"]
    assert q1["action"] == "send" and [e["chunk_id"] for e in q1["evidence"]] == ["d/a-p1-c1", "d/a-p3-c0"]
    assert q1["system"] == TEMPLATE_V2.system and "Date: 4 May." in q1["user"] and "Ann signed it." not in q1["user"]
    nothing_new = {r["question_id"]: r for r in prepare(fresh(tmp_path / "b"), monkeypatch, "retry_retrieval", "exclude-seen-v1", ranked=DOC_A[:2])}
    assert nothing_new["q1"]["skip_reason"] == "nothing_new_retrieved"


def test_correction_skips_unchanged_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path)
    records = {r["question_id"]: r for r in prepare(project, monkeypatch, "correct_text", "symspell-en-82765",
                                                    ranked=[hit("d/a-p1-c0", 1, "Teh agreemnet was sigend."), hit("d/a-p0-c0", 0, "Title page.")])}
    assert records["q1"]["action"] == "send" and records["q1"]["changed_blocks"] >= 1
    assert "agreement" in records["q1"]["user"]
    clean = {r["question_id"]: r for r in prepare(fresh(tmp_path / "c"), monkeypatch, "correct_text", "symspell-en-82765",
                                                  ranked=[hit("d/a-p1-c0", 1, "the page"), hit("d/a-p0-c0", 0, "the title")])}
    assert clean["q1"]["skip_reason"] == "correction_changed_nothing"


# --------------------------------------------------------------------------- T6


def test_the_shared_runner_sends_text_only_and_exports_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path)
    prepare(project, monkeypatch, "retry_retrieval", "exclude-seen-v1")
    provider = ScriptedProvider()
    project.run(provider, name="retry_retrieval")
    assert sorted(provider.sent()) == ["q1", "q2"]
    assert all(isinstance(r.messages[1]["content"], str) for r in provider.requests)
    rows = {r["question_id"]: r for r in rvr.export(project.run_dir("retry_retrieval"))}
    assert rows["q3"]["status"] == "skipped" and rows["q1"]["status"] == "answered"


# --------------------------------------------------------------------------- T8


def test_prepare_never_opens_evaluation_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(tmp_path)
    opened: list[str] = []
    real_open = io.open

    def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(Path(file).resolve().as_posix())
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    prepare(project, monkeypatch, "correct_text", "symspell-en-82765")
    assert opened and not [p for p in opened if "evaluation_manifest" in p or "retrieval_base/gt" in p or "scores.jsonl" in p]
    assert text_repair.EXPANSION_RULES == ("append-anchor-words-v0", "exclude-seen-v1", "keep-top3-add-next2-v1")
