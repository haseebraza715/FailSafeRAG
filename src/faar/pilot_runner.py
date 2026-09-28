"""Offline pilot runner: runtime manifest, retrieval, saved predictions, scoring join.

This module is an engineering path for the frozen pilot. It measures nothing
scientific. Its retrieval settings and answer backend are engineering
settings, not the approved study protocol.

Generation reads exactly two kinds of file: the pilot's ``runtime_manifest.json``
and the MinerU JSON file that each document names. The scoring step
(:func:`score_run`) is the only code here that opens ``evaluation_manifest.json``.
Generation never calls it and never receives its path.

Data flow::

    runtime_manifest.json + MinerU JSON  (hash-verified)
      -> per-document chunks and hybrid retrieval (local-hash embeddings)
      -> one terminal record per question (answered, no_evidence, execution_failed)
      -> predictions.jsonl, run_config.json, generation_summary.json
      -> score_run: join to evaluation_manifest.json -> scores.jsonl, score_summary.json

Run directory contents::

    run_config.json          settings, fingerprint and provenance
    predictions.jsonl        one record per runtime question, manifest order
    generation_summary.json  counts and status; written last, so it marks a complete run
    scores.jsonl             after scoring only
    score_summary.json       after scoring only

Overwrite policy: the runner never overwrites a run.

- An empty or missing directory receives a new run.
- A complete run with a different fingerprint is refused.
- A complete run with the same fingerprint is regenerated in memory. If the
  bytes match, nothing is rewritten and the result reports "already complete,
  verified identical". If they differ, the run is refused as nondeterministic.
- A partial, corrupt or unrecognised directory is refused.

Exit codes are defined by ``EXIT_OK``, ``EXIT_REFUSED`` and ``EXIT_EXECUTION_FAILED``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .run_io import (
    FINGERPRINT_SCHEMA_VERSION,
    _measurement_code_digest,
    atomic_write_text,
    canonical_digest,
)
from .settings import RetrievalSettings
from .types import Chunk, RetrievalHit

RUN_SCHEMA_VERSION = 1
RUN_KIND = "engineering_check"
ENGINEERING_LABEL = (
    "Engineering settings for an offline pipeline check. They are not the approved scientific protocol."
)

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_EXECUTION_FAILED = 2

STAGES = ("load", "retrieve", "answer")
STATUS_ANSWERED = "answered"
STATUS_NO_EVIDENCE = "no_evidence"
STATUS_EXECUTION_FAILED = "execution_failed"
TERMINAL_STATUSES = (STATUS_ANSWERED, STATUS_NO_EVIDENCE, STATUS_EXECUTION_FAILED)
NO_TEXT_CHUNKS = "no_text_chunks"
NO_RETRIEVAL_TOKENS = "no_retrieval_tokens"
NO_HITS = "no_hits"
NO_EVIDENCE_REASONS = (NO_TEXT_CHUNKS, NO_RETRIEVAL_TOKENS, NO_HITS)

RUN_CONFIG_NAME = "run_config.json"
PREDICTIONS_NAME = "predictions.jsonl"
GENERATION_SUMMARY_NAME = "generation_summary.json"
SCORES_NAME = "scores.jsonl"
SCORE_SUMMARY_NAME = "score_summary.json"
GENERATION_FILES = (RUN_CONFIG_NAME, PREDICTIONS_NAME, GENERATION_SUMMARY_NAME)
SCORE_FILES = (SCORES_NAME, SCORE_SUMMARY_NAME)

RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")
OCR_STATUSES = ("ok", "empty", "missing")
RUNTIME_QUESTION_KEYS = frozenset({"question_id", "doc_id", "question"})
FAILURE_MESSAGE_LIMIT = 500

_TOKENISER_PATTERN = "[a-z0-9%$]+"
_WORD_PATTERN = r"\S+"


class RunnerRefusal(Exception):
    """The runner declined to proceed. The message says why and what to do."""


class InjectedFailure(RuntimeError):
    """Raised on purpose by ``--inject-failure`` to exercise the failure path."""


class RetrievalScopeError(RuntimeError):
    """A retrieval hit did not belong to the question's assigned document."""


@dataclass(frozen=True)
class RunnerResult:
    """What a command did, for the CLI to print and turn into an exit status."""

    exit_code: int
    message: str
    summary: dict[str, Any] | None = None
    wrote: bool = False


# ---------------------------------------------------------------------------
# Runtime inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuntimeQuestion:
    """A question as generation sees it. It has no gold, evidence or scoring field."""

    question_id: str
    doc_id: str
    question: str


@dataclass(frozen=True, slots=True)
class DeclaredPage:
    page_idx: int
    pdf_page_number: int
    ocr_status: str
    ocr_text_sha256: str | None


@dataclass(frozen=True)
class RuntimeDocument:
    doc_id: str
    noisy_path: str | None
    noisy_sha256: str | None
    noisy_status: str
    pages: tuple[DeclaredPage, ...]


@dataclass(frozen=True)
class RuntimeManifest:
    pilot_id: str
    noisy_root: str
    documents: tuple[RuntimeDocument, ...]
    questions: tuple[RuntimeQuestion, ...]


@dataclass
class LoadedDocument:
    """A document whose MinerU text passed the hash checks."""

    doc: RuntimeDocument
    page_texts: dict[int, str]
    page_status: dict[int, str]
    undeclared_ocr_pages: int
    chunks: list[Chunk] = field(default_factory=list)
    chunks_with_tokens: int = 0

    @property
    def ocr_condition(self) -> dict[str, int]:
        counts = {status: 0 for status in OCR_STATUSES}
        for status in self.page_status.values():
            counts[status] += 1
        return {
            "pages_total": len(self.page_status),
            "pages_ok": counts["ok"],
            "pages_empty": counts["empty"],
            "pages_missing": counts["missing"],
            "chunks": len(self.chunks),
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerRefusal(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_runtime_manifest(payload: Any, *, expected_pilot_id: str | None = None) -> RuntimeManifest:
    """Validate the runtime manifest and keep only the fields generation may use.

    A question object with any key beyond ``question_id``, ``doc_id`` and
    ``question`` is refused, so a manifest that carries a gold field cannot
    reach generation.
    """
    _require(isinstance(payload, dict), "runtime manifest is not a JSON object")
    _require(payload.get("kind") == "runtime", f"runtime manifest kind is {payload.get('kind')!r}, expected 'runtime'")
    pilot_id = payload.get("pilot_id")
    _require(isinstance(pilot_id, str) and pilot_id != "", "runtime manifest has no pilot_id")
    if expected_pilot_id is not None:
        _require(pilot_id == expected_pilot_id, f"runtime manifest pilot_id {pilot_id!r} differs from {expected_pilot_id!r}")
    sources = payload.get("sources")
    noisy_source = sources.get("noisy_text") if isinstance(sources, dict) else None
    noisy_root = noisy_source.get("root") if isinstance(noisy_source, dict) else None
    _require(isinstance(noisy_root, str) and noisy_root != "", "runtime manifest declares no sources.noisy_text.root")

    documents: list[RuntimeDocument] = []
    seen_docs: set[str] = set()
    raw_documents = payload.get("documents")
    _require(isinstance(raw_documents, list) and raw_documents, "runtime manifest has no documents")
    for raw in raw_documents:
        _require(isinstance(raw, dict), "runtime manifest document is not an object")
        doc_id = raw.get("doc_id")
        _require(isinstance(doc_id, str) and doc_id != "", "runtime manifest document has no doc_id")
        _require(doc_id not in seen_docs, f"runtime manifest repeats doc_id {doc_id!r}")
        seen_docs.add(doc_id)
        noisy = raw.get("noisy_text")
        _require(isinstance(noisy, dict), f"{doc_id}: noisy_text is not an object")
        status = noisy.get("status")
        _require(status in ("present", "missing"), f"{doc_id}: noisy_text.status is {status!r}")
        path = noisy.get("path")
        sha = noisy.get("sha256")
        if status == "present":
            _require(isinstance(path, str) and isinstance(sha, str), f"{doc_id}: a present noisy text needs a path and a sha256")
        else:
            path, sha = None, None
        raw_pages = raw.get("pages")
        _require(isinstance(raw_pages, list), f"{doc_id}: pages is not a list")
        pages: list[DeclaredPage] = []
        seen_idx: set[int] = set()
        for page in raw_pages:
            _require(isinstance(page, dict), f"{doc_id}: page entry is not an object")
            idx = page.get("page_idx")
            _require(_is_int(idx) and idx >= 0, f"{doc_id}: invalid page_idx {idx!r}")
            _require(idx not in seen_idx, f"{doc_id}: repeats page_idx {idx}")
            seen_idx.add(idx)
            _require(page.get("pdf_page_number") == idx + 1, f"{doc_id}: page {idx} has pdf_page_number {page.get('pdf_page_number')!r}")
            ocr_status = page.get("ocr_status")
            _require(ocr_status in OCR_STATUSES, f"{doc_id}: page {idx} has ocr_status {ocr_status!r}")
            page_sha = page.get("ocr_text_sha256")
            _require(
                (ocr_status == "missing") == (page_sha is None),
                f"{doc_id}: page {idx} has ocr_status {ocr_status!r} with ocr_text_sha256 {page_sha!r}",
            )
            pages.append(DeclaredPage(idx, idx + 1, ocr_status, page_sha))
        _require(bool(pages), f"{doc_id}: declares no pages")
        documents.append(RuntimeDocument(doc_id, path, sha, status, tuple(sorted(pages, key=lambda p: p.page_idx))))

    questions: list[RuntimeQuestion] = []
    seen_questions: set[str] = set()
    raw_questions = payload.get("questions")
    _require(isinstance(raw_questions, list) and raw_questions, "runtime manifest has no questions")
    for raw in raw_questions:
        _require(isinstance(raw, dict), "runtime manifest question is not an object")
        extra = sorted(set(raw) - RUNTIME_QUESTION_KEYS)
        _require(not extra, f"runtime question carries fields that generation must not see: {extra}")
        missing = sorted(RUNTIME_QUESTION_KEYS - set(raw))
        _require(not missing, f"runtime question lacks fields: {missing}")
        question_id, doc_id, text = raw["question_id"], raw["doc_id"], raw["question"]
        _require(all(isinstance(v, str) for v in (question_id, doc_id, text)), "runtime question fields must be strings")
        _require(question_id != "", "runtime question has an empty question_id")
        _require(question_id not in seen_questions, f"runtime manifest repeats question_id {question_id!r}")
        _require(doc_id in seen_docs, f"question {question_id} names doc_id {doc_id!r}, which the manifest does not declare")
        seen_questions.add(question_id)
        questions.append(RuntimeQuestion(question_id, doc_id, text))
    return RuntimeManifest(pilot_id, noisy_root, tuple(documents), tuple(questions))


def load_runtime_manifest(path: Path, *, expected_pilot_id: str | None = None) -> tuple[RuntimeManifest, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunnerRefusal(f"cannot read the runtime manifest {path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"runtime manifest {path} is not valid JSON: {exc}") from exc
    return parse_runtime_manifest(payload, expected_pilot_id=expected_pilot_id), sha256_bytes(raw)


def parse_page_inventory(payload: Any) -> dict[int, str]:
    """Read a MinerU page list the way ``scripts/data/audit_assets.py`` did when it built the manifest.

    Entries without a valid ``page_idx`` are skipped. A non-string ``text``
    counts as empty. A repeated ``page_idx`` joins its texts with a newline.
    """
    _require(isinstance(payload, list), "MinerU payload is not a list")
    pages: dict[int, str] = {}
    for row in payload:
        if not isinstance(row, dict) or "page_idx" not in row:
            continue
        idx = row["page_idx"]
        if not _is_int(idx) or idx < 0:
            continue
        text = row.get("text")
        text = text if isinstance(text, str) else ""
        pages[idx] = pages[idx] + "\n" + text if idx in pages else text
    return pages


def page_text_sha256(text: str) -> str:
    """The per-page hash the pilot builder stored: SHA-256 of the UTF-8 page text."""
    return sha256_bytes(text.encode("utf-8"))


def _resolve_noisy_path(project_root: Path, noisy_root: str, relative: str, doc_id: str) -> Path:
    root = (project_root / noisy_root).resolve()
    resolved = (project_root / relative).resolve()
    _require(resolved.is_relative_to(root), f"{doc_id}: noisy text path {relative!r} is outside the declared root {noisy_root!r}")
    return resolved


def load_document(project_root: Path, noisy_root: str, doc: RuntimeDocument) -> LoadedDocument:
    """Read one document's MinerU file and verify the file hash and every declared page hash.

    A mismatch raises :class:`RunnerRefusal`. Text that no longer matches the
    frozen manifest must not produce predictions.
    """
    texts: dict[int, str] = {}
    if doc.noisy_status == "present":
        assert doc.noisy_path is not None and doc.noisy_sha256 is not None
        path = _resolve_noisy_path(project_root, noisy_root, doc.noisy_path, doc.doc_id)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RunnerRefusal(f"{doc.doc_id}: cannot read {doc.noisy_path}: {exc}") from exc
        actual = sha256_bytes(raw)
        _require(actual == doc.noisy_sha256, f"{doc.doc_id}: {doc.noisy_path} has sha256 {actual}, the manifest says {doc.noisy_sha256}")
        try:
            texts = parse_page_inventory(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, RunnerRefusal) as exc:
            raise RunnerRefusal(f"{doc.doc_id}: {doc.noisy_path} is not a MinerU page list: {exc}") from exc
    declared = {page.page_idx for page in doc.pages}
    page_texts: dict[int, str] = {}
    page_status: dict[int, str] = {}
    for page in doc.pages:
        if page.page_idx not in texts:
            status, digest = "missing", None
        else:
            text = texts[page.page_idx]
            status = "empty" if not text.strip() else "ok"
            digest = page_text_sha256(text)
            page_texts[page.page_idx] = text
        _require(
            status == page.ocr_status and digest == page.ocr_text_sha256,
            f"{doc.doc_id}: page {page.page_idx} reads as {status!r} with hash {digest}, "
            f"the manifest says {page.ocr_status!r} with hash {page.ocr_text_sha256}",
        )
        page_status[page.page_idx] = status
    return LoadedDocument(doc, page_texts, page_status, sum(1 for idx in texts if idx not in declared))


# ---------------------------------------------------------------------------
# Retrieval and answer backend
# ---------------------------------------------------------------------------


def engineering_retrieval_settings() -> RetrievalSettings:
    """Chunking and retrieval settings for this offline path.

    The local-hash backend needs no model download. The chunk size, overlap and
    ``top_k`` repeat the repository defaults and are engineering settings.
    """
    from .retrieval import LOCAL_HASH_BACKEND

    return RetrievalSettings(
        chunk_size_words=180,
        chunk_overlap_words=40,
        top_k=5,
        embedding_backend=LOCAL_HASH_BACKEND,
    )


def describe_retrieval(settings: RetrievalSettings) -> dict[str, Any]:
    """The retrieval settings as recorded in ``run_config.json`` and hashed into the fingerprint."""
    from .retrieval import LOCAL_HASH_BACKEND, LocalHashEmbedder

    _require(settings.embedding_backend == LOCAL_HASH_BACKEND, "the offline runner only supports the local-hash embedding backend")
    return {
        "label": ENGINEERING_LABEL,
        "scope": "one index per document, built from every declared page; a question searches only its own document",
        "embedding_backend": settings.embedding_backend,
        "embedding_dimensions": LocalHashEmbedder.dimensions,
        "embedding": "signed feature hashing of blake2b token digests, L2-normalised, no model download",
        "reranker": None,
        "chunking": {
            "function": "faar.chunking.build_page_chunks",
            "word_tokeniser": _WORD_PATTERN,
            "chunk_size_words": settings.chunk_size_words,
            "chunk_overlap_words": settings.chunk_overlap_words,
            "chunk_id": "<doc_id>-p<page_idx>-c<chunk_index>",
            "pages_without_text": "empty or missing OCR gives no chunk and is counted in ocr_condition",
        },
        "tokenisation": {
            "retrieval_tokeniser": _TOKENISER_PATTERN,
            "applied_to": "lower-cased chunk and query text, for BM25 and for the hashing embedder",
            "han_script": "the tokeniser matches only [a-z0-9%$], so Han-script text gets no lexical or hashed-embedding signal",
        },
        "ranking": {
            "lexical": "BM25Okapi from rank_bm25, default parameters",
            "fusion": "0.45 * min-max dense + 0.35 * min-max bm25 + 0.20 * reciprocal-rank component (k=60)",
            "dense_note": "faiss returns the dense scores of the top_k chunks only; other chunks enter the fusion with a dense score of 0",
            "order": "descending fused score",
            "tie_breaking": "numpy stable argsort: equal scores keep chunk order (page_idx ascending, then chunk_index)",
        },
        "no_evidence_reasons": {
            NO_TEXT_CHUNKS: "every declared page has empty or missing OCR, so the document has zero chunks (an OCR condition)",
            NO_RETRIEVAL_TOKENS: (
                "chunks exist but none holds a [a-z0-9%$] token, so BM25 cannot index the document (rank_bm25 divides by zero). "
                "This is a limitation of the engineering tokeniser, not an OCR condition. Han-only and symbol-only text lands here. "
                "The document is listed in generation_summary.json"
            ),
            NO_HITS: "the retriever returned zero hits",
        },
        "top_k": settings.top_k,
        "max_chunks": settings.max_chunks,
        "evidence_limit": "at most top_k hits per question, all saved in predictions.jsonl with their scores",
    }


class AnswerBackend(Protocol):
    """The answer step behind an explicit interface. Backends here make no model call."""

    def identity(self) -> dict[str, Any]: ...

    def answer(self, question: str, hits: Sequence[RetrievalHit]) -> Mapping[str, Any]:
        """Return ``{"answer": str, "answer_mode": str | None}``."""
        ...


class RuleBasedExtractiveBackend:
    """Engineering-only backend that wraps the rule-based ``faar.answering.answer_from_hits``."""

    name = "rule_based_extractive"

    def identity(self) -> dict[str, Any]:
        from . import answering

        return {
            "name": self.name,
            "implementation": "faar.answering.answer_from_hits",
            "implementation_sha256": sha256_file(Path(answering.__file__)),
            "engineering_only": True,
            "model_calls": False,
        }

    def answer(self, question: str, hits: Sequence[RetrievalHit]) -> Mapping[str, Any]:
        from .answering import answer_from_hits

        return answer_from_hits(question, list(hits))


class _DocumentIndex:
    """Chunks for one document, built when the index is created, and a retriever built on first use.

    Chunking is deterministic and cheap, so every record of a document reports
    the same chunk count whatever the question order.
    """

    def __init__(self, loaded: LoadedDocument, settings: RetrievalSettings) -> None:
        from .chunking import build_page_chunks
        from .retrieval import _tokenize

        self._settings = settings
        self._loaded = loaded
        self._built = False
        self._error: Exception | None = None
        self._retriever: Any = None
        chunks: list[Chunk] = []
        try:
            for page_idx in sorted(loaded.page_texts):
                chunks.extend(
                    build_page_chunks(
                        example_id=loaded.doc.doc_id,
                        doc_name=loaded.doc.doc_id,
                        page_id=page_idx,
                        page_text=loaded.page_texts[page_idx],
                        settings=settings,
                    )
                )
        except Exception as exc:
            self._error, self._built, chunks = exc, True, []
        loaded.chunks = chunks
        loaded.chunks_with_tokens = sum(1 for chunk in chunks if _tokenize(chunk.text))

    @property
    def empty_reason(self) -> str | None:
        """Why the document cannot be searched: ``no_text_chunks``, ``no_retrieval_tokens`` or ``None``."""
        if not self._loaded.chunks:
            return NO_TEXT_CHUNKS
        if not self._loaded.chunks_with_tokens:
            return NO_RETRIEVAL_TOKENS
        return None

    def retriever(self) -> Any:
        """Return the document's retriever, or ``None`` when no chunk holds a retrieval token.

        ``rank_bm25`` raises ``ZeroDivisionError`` on a corpus without a single
        token, so a document whose chunks match no ``[a-z0-9%$]`` token (for
        example a page of only ``#`` marks, or only Han script) has nothing to
        index. Its questions get ``no_evidence``. ``run_config.json`` lists
        such documents.
        """
        if not self._built:
            self._built = True
            try:
                if self._loaded.chunks_with_tokens:
                    from .retrieval import HybridRetriever

                    self._retriever = HybridRetriever(self._loaded.chunks, self._settings)
            except Exception as exc:
                self._error = exc
        if self._error is not None:
            raise self._error
        return self._retriever


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectedFailureSpec:
    question_id: str
    stage: str

    def as_dict(self) -> dict[str, str]:
        return {"question_id": self.question_id, "stage": self.stage}


def parse_injected_failures(specs: Sequence[str], known_question_ids: set[str]) -> tuple[InjectedFailureSpec, ...]:
    """Turn ``QUESTION_ID`` or ``QUESTION_ID:STAGE`` strings into specs. The default stage is ``answer``."""
    parsed: dict[str, InjectedFailureSpec] = {}
    for spec in specs:
        question_id, _, stage = spec.partition(":")
        stage = stage or "answer"
        _require(stage in STAGES, f"--inject-failure {spec!r}: stage must be one of {', '.join(STAGES)}")
        _require(question_id in known_question_ids, f"--inject-failure {spec!r}: no runtime question has that id")
        _require(question_id not in parsed, f"--inject-failure names question {question_id!r} more than once")
        parsed[question_id] = InjectedFailureSpec(question_id, stage)
    return tuple(sorted(parsed.values(), key=lambda s: s.question_id))


def _finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _evidence_record(rank: int, hit: RetrievalHit) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": hit.chunk.chunk_id,
        "doc_id": hit.chunk.doc_name,
        "page_idx": hit.chunk.page_id,
        "fused_score": float(hit.fused_score),
        "bm25_score": float(hit.bm25_score),
        "dense_score": float(hit.dense_score),
    }


def _record(question: RuntimeQuestion, status: str, loaded: LoadedDocument, **fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "question_id": question.question_id,
        "doc_id": question.doc_id,
        "status": status,
        "answer": None,
        "abstained": False,
        "no_evidence_reason": None,
        "answer_mode": None,
        "evidence": [],
        "ocr_condition": loaded.ocr_condition,
        "failure": None,
    }
    record.update(fields)
    return record


def answer_question(
    question: RuntimeQuestion,
    loaded: LoadedDocument,
    index: _DocumentIndex,
    backend: AnswerBackend,
    injection: InjectedFailureSpec | None,
) -> dict[str, Any]:
    """Produce the one terminal record for a question. Any exception becomes ``execution_failed``."""
    stage = "load"
    evidence: list[dict[str, Any]] = []

    def inject(at: str) -> None:
        if injection is not None and injection.stage == at:
            raise InjectedFailure(f"injected failure at stage {at!r} for question {question.question_id}")

    try:
        inject("load")
        retriever = index.retriever()
        stage = "retrieve"
        inject("retrieve")
        hits: list[RetrievalHit] = []
        if retriever is not None:
            hits = list(retriever.retrieve(question.question))
        for hit in hits:
            if hit.chunk.doc_name != question.doc_id:
                raise RetrievalScopeError(
                    f"hit {hit.chunk.chunk_id!r} belongs to {hit.chunk.doc_name!r}, not to {question.doc_id!r}"
                )
            if not all(_finite(v) for v in (hit.fused_score, hit.bm25_score, hit.dense_score)):
                raise ValueError(f"hit {hit.chunk.chunk_id!r} has a non-finite score")
        evidence = [_evidence_record(rank, hit) for rank, hit in enumerate(hits, start=1)]
        stage = "answer"
        inject("answer")
        if not hits:
            reason = index.empty_reason or NO_HITS
            return _record(question, STATUS_NO_EVIDENCE, loaded, answer="", abstained=True, no_evidence_reason=reason)
        result = backend.answer(question.question, hits)
        answer, mode = result.get("answer"), result.get("answer_mode")
        if not isinstance(answer, str) or not (mode is None or isinstance(mode, str)):
            raise TypeError("answer backend must return a str answer and a str or None answer_mode")
        return _record(question, STATUS_ANSWERED, loaded, answer=answer, answer_mode=mode, evidence=evidence)
    except Exception as exc:
        failure = {"stage": stage, "type": type(exc).__name__, "message": str(exc)[:FAILURE_MESSAGE_LIMIT]}
        return _record(question, STATUS_EXECUTION_FAILED, loaded, evidence=evidence, failure=failure)


def check_terminal_records(questions: Sequence[RuntimeQuestion], records: Sequence[Mapping[str, Any]]) -> None:
    """Raise unless there is exactly one terminal record per question, in manifest order."""
    ids = [record["question_id"] for record in records]
    expected = [question.question_id for question in questions]
    if ids != expected:
        raise RuntimeError("predictions do not hold exactly one record per runtime question in manifest order")
    bad = [r["question_id"] for r in records if r["status"] not in TERMINAL_STATUSES]
    if bad:
        raise RuntimeError(f"non-terminal statuses for questions {bad[:3]}")


def generate_records(
    manifest: RuntimeManifest,
    loaded_documents: Mapping[str, LoadedDocument],
    settings: RetrievalSettings,
    backend: AnswerBackend,
    injections: Sequence[InjectedFailureSpec] = (),
) -> list[dict[str, Any]]:
    """Answer every runtime question. The evaluation manifest plays no part."""
    indexes = {doc_id: _DocumentIndex(loaded, settings) for doc_id, loaded in loaded_documents.items()}
    by_question = {spec.question_id: spec for spec in injections}
    records = [
        answer_question(q, loaded_documents[q.doc_id], indexes[q.doc_id], backend, by_question.get(q.question_id))
        for q in manifest.questions
    ]
    check_terminal_records(manifest.questions, records)
    return records


def _dumps(payload: Any, *, indent: int | None = 2) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=False, allow_nan=False)


def render_predictions(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_dumps(record, indent=None) + "\n" for record in records)


def summarise_generation(
    *,
    run_id: str,
    pilot_id: str,
    fingerprint: str,
    records: Sequence[Mapping[str, Any]],
    loaded_documents: Mapping[str, LoadedDocument],
    predictions_text: str,
    injections: Sequence[InjectedFailureSpec],
) -> dict[str, Any]:
    counts = {status: 0 for status in TERMINAL_STATUSES}
    stages: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for record in records:
        counts[record["status"]] += 1
        if record["no_evidence_reason"] is not None:
            reasons[record["no_evidence_reason"]] = reasons.get(record["no_evidence_reason"], 0) + 1
        if record["failure"] is not None:
            stages[record["failure"]["stage"]] = stages.get(record["failure"]["stage"], 0) + 1
    failed = counts[STATUS_EXECUTION_FAILED]
    pages = {status: 0 for status in OCR_STATUSES}
    for loaded in loaded_documents.values():
        for status in loaded.page_status.values():
            pages[status] += 1
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "pilot_id": pilot_id,
        "kind": RUN_KIND,
        "label": ENGINEERING_LABEL,
        "status": "complete" if failed == 0 else "complete_with_execution_failures",
        "exit_code": EXIT_OK if failed == 0 else EXIT_EXECUTION_FAILED,
        "all_questions_terminal": True,
        "counts": {
            "questions": len(records),
            "answered": counts[STATUS_ANSWERED],
            "no_evidence": counts[STATUS_NO_EVIDENCE],
            "execution_failed": failed,
            "abstained": counts[STATUS_NO_EVIDENCE],
        },
        "no_evidence_by_reason": {reason: reasons.get(reason, 0) for reason in NO_EVIDENCE_REASONS},
        "execution_failures_by_stage": dict(sorted(stages.items())),
        "injected_failures": [spec.as_dict() for spec in injections],
        "documents": len(loaded_documents),
        "pages": {"total": sum(pages.values()), "ok": pages["ok"], "empty": pages["empty"], "missing": pages["missing"]},
        "chunks": sum(len(loaded.chunks) for loaded in loaded_documents.values()),
        "documents_with_chunks_but_no_retrieval_tokens": sorted(
            doc_id for doc_id, loaded in loaded_documents.items() if loaded.chunks and not loaded.chunks_with_tokens
        ),
        "fingerprint": fingerprint,
        "predictions_sha256": sha256_bytes(predictions_text.encode("utf-8")),
    }


# ---------------------------------------------------------------------------
# Provenance and run directory
# ---------------------------------------------------------------------------


def git_provenance(code_root: Path | None, exclude: Sequence[Path] = ()) -> dict[str, Any]:
    """Commit, dirty flag and dirty paths of the checkout that holds the code. Unknown values are null."""
    unknown: dict[str, Any] = {"commit": None, "dirty": None, "dirty_paths": None}
    if code_root is None:
        return unknown
    try:
        commit = subprocess.run(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(code_root), "status", "--porcelain=v1", "-z", "-uall"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return unknown
    top = subprocess.run(
        ["git", "-C", str(code_root), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
    ).stdout.strip()
    top_path = Path(top).resolve() if top else code_root.resolve()
    excluded = [p.resolve() for p in exclude]
    entries = status.split("\0")
    paths: list[str] = []
    skip_next = False
    for entry in entries:
        if skip_next:
            skip_next = False
            continue
        if len(entry) < 4:
            continue
        code, path = entry[:2], entry[3:]
        if "R" in code or "C" in code:
            skip_next = True
        absolute = (top_path / path).resolve()
        if any(absolute == e or e in absolute.parents for e in excluded):
            continue
        paths.append(path)
    return {"commit": commit, "dirty": bool(paths), "dirty_paths": sorted(paths)}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in ("numpy", "faiss-cpu", "rank-bm25"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def refuse_pilot_directory(run_dir: Path, project_root: Path) -> None:
    resolved = run_dir.resolve()
    frozen = (project_root / "results" / "pilots").resolve()
    parts = resolved.parts
    in_pilots = resolved == frozen or frozen in resolved.parents
    in_pilots = in_pilots or any(parts[i] == "results" and parts[i + 1] == "pilots" for i in range(len(parts) - 1))
    _require(not in_pilots, f"refusing {run_dir}: runs must not be written under results/pilots/. Use results/engineering/<run_id>/.")


def _read_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ExistingRun:
    config: dict[str, Any]
    summary: dict[str, Any]
    predictions_text: str
    summary_text: str
    score_files: tuple[str, ...]


def inspect_run_dir(run_dir: Path) -> ExistingRun | None:
    """Return the run stored in ``run_dir``, or ``None`` for a missing or empty directory.

    Raises :class:`RunnerRefusal` for a partial, corrupt or unrecognised directory.
    """
    if not run_dir.exists():
        return None
    _require(run_dir.is_dir(), f"{run_dir} exists and is not a directory")
    names = {entry.name for entry in run_dir.iterdir()}
    if not names:
        return None
    unknown = sorted(names - set(GENERATION_FILES) - set(SCORE_FILES))
    _require(not unknown, f"{run_dir} holds unrecognised entries {unknown}; it is not a complete run directory. Pick a new run directory.")
    missing = [name for name in GENERATION_FILES if name not in names]
    _require(
        not missing,
        f"{run_dir} holds a partial run (missing {', '.join(missing)}). The runner never overwrites. "
        "Keep the directory as a record of the interrupted attempt and choose a new run directory.",
    )
    try:
        config = _read_json_file(run_dir / RUN_CONFIG_NAME)
        summary_text = (run_dir / GENERATION_SUMMARY_NAME).read_text(encoding="utf-8")
        summary = json.loads(summary_text)
        predictions_text = (run_dir / PREDICTIONS_NAME).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"{run_dir} holds a corrupt run: {exc}") from exc
    consistent = (
        isinstance(config, dict)
        and isinstance(summary, dict)
        and summary.get("predictions_sha256") == sha256_bytes(predictions_text.encode("utf-8"))
        and summary.get("fingerprint") == config.get("fingerprint")
        and summary.get("run_id") == config.get("run_id")
    )
    _require(consistent, f"{run_dir} holds a corrupt run: predictions.jsonl, run_config.json and generation_summary.json disagree")
    return ExistingRun(config, summary, predictions_text, summary_text, tuple(sorted(names & set(SCORE_FILES))))


def validate_run_id(run_id: str) -> str:
    _require(bool(RUN_ID_PATTERN.match(run_id)), f"run_id {run_id!r} must match {RUN_ID_PATTERN.pattern}")
    return run_id


def generation_fingerprint(
    *,
    pilot_id: str,
    runtime_manifest_sha256: str,
    document_hashes: Mapping[str, str | None],
    retrieval: Mapping[str, Any],
    backend: Mapping[str, Any],
    injections: Sequence[InjectedFailureSpec],
    cli_script_sha256: str | None,
) -> str:
    """Bind the code, inputs and settings that decide the predictions.

    ``code_identity`` hashes every ``src/faar/*.py`` file through the existing
    ``faar.run_io`` helper, so a code change gives a new fingerprint.
    """
    return canonical_digest(
        {
            "runner_schema_version": RUN_SCHEMA_VERSION,
            "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            "code_identity": _measurement_code_digest(),
            "cli_script_sha256": cli_script_sha256,
            "kind": RUN_KIND,
            "pilot_id": pilot_id,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "document_noisy_text_sha256": dict(document_hashes),
            "retrieval": dict(retrieval),
            "answer_backend": dict(backend),
            "injected_failures": [spec.as_dict() for spec in injections],
        }
    )


def generate_run(
    *,
    project_root: Path,
    run_dir: Path,
    run_id: str | None = None,
    pilot_id: str = "ohr_dev_v1",
    runtime_manifest_path: Path | None = None,
    inject_failures: Sequence[str] = (),
    backend: AnswerBackend | None = None,
    settings: RetrievalSettings | None = None,
    cli_script: Path | None = None,
    code_root: Path | None = None,
    command: Sequence[str] | None = None,
) -> RunnerResult:
    """Run the offline pipeline once and save the predictions. See the module docstring for the overwrite policy.

    Raises :class:`RunnerRefusal` when the run must not start or continue.
    """
    project_root = project_root.resolve()
    run_dir = run_dir if run_dir.is_absolute() else Path.cwd() / run_dir
    refuse_pilot_directory(run_dir, project_root)
    run_id = validate_run_id(run_id if run_id is not None else run_dir.name)
    manifest_path = runtime_manifest_path or project_root / "results" / "pilots" / pilot_id / "runtime_manifest.json"
    manifest, manifest_sha = load_runtime_manifest(manifest_path, expected_pilot_id=pilot_id)
    injections = parse_injected_failures(inject_failures, {q.question_id for q in manifest.questions})
    loaded_documents = {doc.doc_id: load_document(project_root, manifest.noisy_root, doc) for doc in manifest.documents}
    backend = backend or RuleBasedExtractiveBackend()
    settings = settings or engineering_retrieval_settings()
    retrieval = describe_retrieval(settings)
    backend_identity = dict(backend.identity())
    _require(backend_identity.get("engineering_only") is True, "the answer backend must declare engineering_only: true")
    script_sha = sha256_file(cli_script) if cli_script is not None else None
    fingerprint = generation_fingerprint(
        pilot_id=manifest.pilot_id,
        runtime_manifest_sha256=manifest_sha,
        document_hashes={doc.doc_id: doc.noisy_sha256 for doc in manifest.documents},
        retrieval=retrieval,
        backend=backend_identity,
        injections=injections,
        cli_script_sha256=script_sha,
    )

    existing = inspect_run_dir(run_dir)
    if existing is not None:
        _require(
            existing.config.get("fingerprint") == fingerprint,
            f"{run_dir} holds a complete run with fingerprint {existing.config.get('fingerprint')}, "
            f"this invocation has fingerprint {fingerprint}. Code, inputs or settings changed. "
            "The runner never overwrites; use a new run directory.",
        )
        _require(
            existing.config.get("run_id") == run_id,
            f"{run_dir} holds run_id {existing.config.get('run_id')!r}, this invocation names {run_id!r}",
        )

    records = generate_records(manifest, loaded_documents, settings, backend, injections)
    predictions_text = render_predictions(records)
    summary = summarise_generation(
        run_id=run_id,
        pilot_id=manifest.pilot_id,
        fingerprint=fingerprint,
        records=records,
        loaded_documents=loaded_documents,
        predictions_text=predictions_text,
        injections=injections,
    )
    summary_text = _dumps(summary) + "\n"
    counts = summary["counts"]
    tally = f"{counts['questions']} questions: {counts['answered']} answered, {counts['no_evidence']} no_evidence, {counts['execution_failed']} execution_failed"

    if existing is not None:
        _require(
            existing.predictions_text == predictions_text and existing.summary_text == summary_text,
            f"{run_dir} holds a complete run with the same fingerprint, but regenerating it gives different output. "
            "The pipeline is nondeterministic or the saved files were altered. Nothing was written.",
        )
        return RunnerResult(
            summary["exit_code"],
            f"already complete, verified identical: run {run_id}, fingerprint {fingerprint}, {tally}. Nothing rewritten.",
            summary,
            wrote=False,
        )

    git = git_provenance(code_root, exclude=[run_dir])
    config = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "pilot_id": manifest.pilot_id,
        "kind": RUN_KIND,
        "label": ENGINEERING_LABEL,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "fingerprint": fingerprint,
        "fingerprint_inputs": {
            "code_identity": _measurement_code_digest(),
            "cli_script_sha256": script_sha,
        },
        "code": {
            **git,
            "code_root": _display_path(code_root, project_root) if code_root else None,
            "faar_package": _display_path(Path(__file__).resolve().parent, code_root or project_root),
        },
        "command": list(command) if command is not None else None,
        "run_dir": _display_path(run_dir, project_root),
        "runtime_manifest": {"path": _display_path(manifest_path, project_root), "sha256": manifest_sha},
        "documents": [
            {
                "doc_id": doc.doc_id,
                "noisy_text_path": doc.noisy_path,
                "noisy_text_sha256": doc.noisy_sha256,
                "undeclared_ocr_pages": loaded_documents[doc.doc_id].undeclared_ocr_pages,
                "chunks_with_retrieval_tokens": loaded_documents[doc.doc_id].chunks_with_tokens,
                **loaded_documents[doc.doc_id].ocr_condition,
            }
            for doc in manifest.documents
        ],
        "retrieval": retrieval,
        "answer_backend": backend_identity,
        "injected_failures": [spec.as_dict() for spec in injections],
        "environment": _package_versions(),
        "inputs_read": [
            "runtime manifest",
            "MinerU JSON file of each declared document",
        ],
        "inputs_never_read": [
            "evaluation_manifest.json",
            "selection_record.json",
            "inspection/",
            "OHR-Bench/data/qas*.json",
            "gt text",
            "annotations",
            "audit manifest",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / PREDICTIONS_NAME, predictions_text)
    atomic_write_text(run_dir / RUN_CONFIG_NAME, _dumps(config) + "\n")
    atomic_write_text(run_dir / GENERATION_SUMMARY_NAME, summary_text)
    return RunnerResult(summary["exit_code"], f"run {run_id} written to {run_dir}: {tally}. Status: {summary['status']}.", summary, wrote=True)


# ---------------------------------------------------------------------------
# Scoring join
# ---------------------------------------------------------------------------


def load_predictions(predictions_text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in predictions_text.splitlines() if line.strip()]


def score_run(
    *,
    project_root: Path,
    run_dir: Path,
    evaluation_manifest_path: Path | None = None,
) -> RunnerResult:
    """Join saved predictions to the evaluation manifest and write ``scores.jsonl`` and ``score_summary.json``.

    Existing score files are never overwritten. If both exist and a fresh
    scoring gives identical bytes, the result says so and writes nothing.
    """
    project_root = project_root.resolve()
    run_dir = run_dir if run_dir.is_absolute() else Path.cwd() / run_dir
    refuse_pilot_directory(run_dir, project_root)
    existing = inspect_run_dir(run_dir)
    _require(existing is not None, f"{run_dir} holds no complete generation run. Run the generate subcommand first.")
    assert existing is not None
    _require(len(existing.score_files) in (0, 2), f"{run_dir} holds only {existing.score_files}; scoring is partial. Keep it and choose a new run directory.")
    pilot_id = existing.config["pilot_id"]
    path = evaluation_manifest_path or project_root / "results" / "pilots" / pilot_id / "evaluation_manifest.json"
    try:
        evaluation_bytes = path.read_bytes()
        evaluation = json.loads(evaluation_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerRefusal(f"cannot read the evaluation manifest {path}: {exc}") from exc
    _require(isinstance(evaluation, dict) and isinstance(evaluation.get("questions"), dict), f"{path} has no questions object")
    _require(evaluation.get("pilot_id") == pilot_id, f"{path} is for pilot {evaluation.get('pilot_id')!r}, the run is for {pilot_id!r}")
    try:
        scoring = importlib.import_module("faar.ohr_scoring")
    except ImportError as exc:
        raise RunnerRefusal(f"faar.ohr_scoring is not available in this checkout: {exc}") from exc

    predictions = load_predictions(existing.predictions_text)
    try:
        result = scoring.score_predictions(predictions, evaluation["questions"])
    except ValueError as exc:
        raise RunnerRefusal(f"scoring refused the join: {exc}") from exc
    scores_text = "".join(_dumps(row, indent=None) + "\n" for row in result["rows"])
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": existing.config["run_id"],
        "pilot_id": pilot_id,
        "kind": RUN_KIND,
        "label": ENGINEERING_LABEL,
        "scorer": scoring.scorer_identity(),
        "evaluation_manifest": {"path": _display_path(path, project_root), "sha256": sha256_bytes(evaluation_bytes)},
        "generation_fingerprint": existing.config["fingerprint"],
        "predictions_sha256": existing.summary["predictions_sha256"],
        "counts": result["counts"],
        "aggregates": result["aggregates"],
        "scores_sha256": sha256_bytes(scores_text.encode("utf-8")),
    }
    summary_text = _dumps(summary) + "\n"
    failed = result["counts"].get("execution_failed", 0)
    exit_code = EXIT_OK if failed == 0 else EXIT_EXECUTION_FAILED
    aggregate = result["aggregates"]["all_questions"]
    line = (
        f"{result['counts']['questions']} questions, execution_failed {failed}; "
        f"all-question EM {aggregate['em']}, F1 {aggregate['f1']}"
    )
    if existing.score_files:
        same = (
            (run_dir / SCORES_NAME).read_text(encoding="utf-8") == scores_text
            and (run_dir / SCORE_SUMMARY_NAME).read_text(encoding="utf-8") == summary_text
        )
        _require(same, f"{run_dir} already holds scores that differ from a fresh scoring. Nothing was written.")
        return RunnerResult(exit_code, f"already scored, verified identical: {line}. Nothing rewritten.", summary, wrote=False)
    atomic_write_text(run_dir / SCORES_NAME, scores_text)
    atomic_write_text(run_dir / SCORE_SUMMARY_NAME, summary_text)
    return RunnerResult(exit_code, f"scores written to {run_dir}: {line}.", summary, wrote=True)
