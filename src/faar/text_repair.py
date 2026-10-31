"""Text repairs for the FAAR comparison: ``retry_retrieval`` and ``correct_text``.

Both repairs answer again with the shared prompt (``faar-answer-v2``) from new evidence blocks. They read only the
question, the retrieved chunks and the same document's MinerU text (study brief section 4). No ground-truth text is
ever a correction target or a retrieval source.

Query expansion rules for ``retry_retrieval`` (one is chosen on development data, analysis plan amendment 3):

``append-anchor-words-v0``  the current rule (``recovery.semantic_backtrack``): the question plus the first 24 words
                            of the retrieved text, then the top ``k`` chunks. It can repeat the first error.
``exclude-seen-v1``         the question unchanged, retrieving ``2k`` chunks and keeping the first ``k`` that the
                            initial answer did not see. The model always gets evidence it has not seen.

Correctors for ``correct_text`` (one is chosen on development data):

``symspell-en-82765``  SymSpell closest-word lookup with the English frequency dictionary shipped with symspellpy
                       (``frequency_dictionary_en_82_765.txt``). Alphabetic tokens only; other scripts pass unchanged.
``byt5-small``         the ByT5 corrector of ``recovery.ByT5Corrector``. Needs a model download that is not
                       authorised yet.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .live_contract import EvidenceBlock
from .recovery import semantic_backtrack

EXPANSION_RULES = ("append-anchor-words-v0", "exclude-seen-v1")
CORRECTORS = ("symspell-en-82765", "byt5-small")


class TextRepairError(Exception):
    """A text repair input is invalid or unavailable. Nothing was sent."""


def blocks_from_hits(hits: Sequence[Any]) -> list[EvidenceBlock]:
    """Evidence blocks in rank order, as the no-recovery run builds them."""
    return [EvidenceBlock(rank, h.chunk.chunk_id, h.chunk.doc_name, h.chunk.page_id, h.chunk.text) for rank, h in enumerate(hits, start=1)]


def retry_hits(rule: str, question: str, initial_hits: Sequence[Any], retriever: Any, top_k: int) -> list[Any]:
    """The hits of the second retrieval under ``rule``. Empty when the rule finds nothing new."""
    if rule == "append-anchor-words-v0":
        return list(retriever.retrieve(semantic_backtrack(question, list(initial_hits)), top_k=top_k))
    if rule == "exclude-seen-v1":
        seen = {h.chunk.chunk_id for h in initial_hits}
        return [h for h in retriever.retrieve(question, top_k=2 * top_k) if h.chunk.chunk_id not in seen][:top_k]
    raise TextRepairError(f"unknown expansion rule {rule!r}; known: {EXPANSION_RULES}")


def symspell_dictionary() -> Path:
    import symspellpy

    path = Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt"
    if not path.is_file():
        raise TextRepairError(f"the SymSpell dictionary {path} is missing")
    return path


def corrector(name: str) -> tuple[Callable[[str], str], dict[str, Any]]:
    """A text-to-text corrector and the identity to record for it."""
    if name == "symspell-en-82765":
        from .symspell import correct_text

        path = symspell_dictionary()
        identity = {"name": name, "dictionary": path.name, "dictionary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return (lambda text: correct_text(text, path)), identity
    if name == "byt5-small":
        raise TextRepairError("byt5-small needs a model download that has not been authorised; nothing was corrected")
    raise TextRepairError(f"unknown corrector {name!r}; known: {CORRECTORS}")


def corrected_blocks(blocks: Sequence[EvidenceBlock], correct: Callable[[str], str]) -> tuple[list[EvidenceBlock], int]:
    """Each block with corrected text, in the same order, and how many blocks changed."""
    out, changed = [], 0
    for b in blocks:
        text = correct(b.text)
        changed += text != b.text
        out.append(EvidenceBlock(b.rank, b.chunk_id, b.doc_id, b.page_idx, text))
    return out, changed


def replay_retrieval(project_root: Path, pilot_id: str, baseline_requests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Rerun the no-recovery retrieval offline and check it hit for hit against the run's recorded evidence.

    Returns the hits per question and a retriever per document (for ``retry_retrieval``). Refuses when any question's
    chunks or chunk texts differ from what the no-recovery run sent, because the repair must start from the same
    evidence as the initial answer.
    """
    from .pilot_runner import (
        DEFAULT_TEXT_POLICY,
        _DocumentIndex,
        engineering_retrieval_settings,
        retrieve_runtime_questions,
    )

    settings = engineering_retrieval_settings()
    run = retrieve_runtime_questions(project_root=project_root, pilot_id=pilot_id, settings=settings, text_policy=DEFAULT_TEXT_POLICY)
    hits: dict[str, list[Any]] = {}
    for outcome in run.questions:
        qid = outcome.question.question_id
        recorded = baseline_requests[qid]
        got = [(h.chunk.chunk_id, hashlib.sha256(h.chunk.text.encode("utf-8")).hexdigest()) for h in outcome.hits]
        want = [(e["chunk_id"], e["text_sha256"]) for e in sorted(recorded["evidence"], key=lambda e: e["rank"])] if recorded["action"] == "send" else []
        if recorded["action"] == "send" and got != want:
            raise TextRepairError(f"{qid}: offline retrieval differs from the evidence the no-recovery run sent")
        hits[qid] = list(outcome.hits)
    indexes = {doc_id: _DocumentIndex(loaded, settings, DEFAULT_TEXT_POLICY) for doc_id, loaded in run.loaded_documents.items()}
    return {"hits": hits, "indexes": indexes, "settings": settings, "tokenizer": DEFAULT_TEXT_POLICY.tokenizer,
            "retrieval": dict(run.retrieval), "runtime_sha256": run.runtime_manifest_sha256}
