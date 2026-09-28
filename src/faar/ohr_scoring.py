"""Score answers with the official OHR-Bench exact-match and F1 metrics.

Provenance
----------
The OHR-Bench evaluator scores each generated answer in
``QuestAnswer.scoring`` (``src/tasks/quest_answer.py``), which calls
``exact_match_score`` and ``f1_score`` from ``src/metric/common.py``. This
module reimplements those two functions and the helpers they use
(``normalize_answer``, ``has_chn_character``, ``catch_all_exceptions``) so the
project can score without upstream's heavy imports (``evaluate``, ``text2vec``,
``loguru``, ``numpy``).

* Repository: https://github.com/opendatalab/OHR-Bench
* Commit: 1f421eb428f9f5b8ac0bc8064d6ad1f13fab7af7
* ``src/metric/common.py`` SHA-256: 9fe7eb527762d1be8a00d9a98a775b850bb42e9441a09db84e88996b94ef5a35
* ``src/tasks/quest_answer.py`` SHA-256: 1feaee1780ed7e3ea597559c3c3addfeb3596cb2b679c11fba885ba40ca21b56

The upstream file header credits Shichao Song, and the OHR-Bench README says the
evaluation framework is based on CRUD_RAG (https://github.com/IAAR-Shanghai/CRUD_RAG).
At that commit the GitHub repository has no LICENSE file, and GitHub reports no
licence for it. The Hugging Face dataset card for ``opendatalab/OHR-Bench``
declares ``cc-by-4.0`` for the dataset repository. This module credits the
authors and pins the source; it does not claim upstream grants code reuse
rights beyond that. The functions are short and follow the well-known SQuAD
answer-normalisation layout. Check the terms before redistributing this module
outside the project.

What the module preserves
-------------------------
The metric keeps upstream's behaviour, including these quirks:

* ``normalize_answer`` lowercases, then deletes ASCII ``string.punctuation``
  characters without inserting a space, then replaces the words ``a``, ``an``
  and ``the`` with a space, then collapses whitespace. Deleting punctuation
  first means ``1,000`` and ``1000`` are equal, ``3.5`` and ``35`` are equal,
  and ``A,B`` becomes the single token ``ab`` while ``A, B`` stays two tokens.
* Non-ASCII punctuation (``。``, ``，``, ``“``), accents and full-width forms are
  kept.
* ``exact_match_score`` compares the two normalised strings for equality. Two
  strings that normalise to the empty string are equal.
* ``f1_score`` returns 0 when either normalised side is ``yes``, ``no`` or
  ``noanswer`` and the two sides differ. It splits on whitespace, unless either
  raw string contains a character whose Unicode name contains ``CJK``, in which
  case it tokenises both normalised strings with ``jieba.lcut``. Kana, Hangul
  and CJK punctuation do not trigger that path. Whitespace tokens from
  ``jieba.lcut`` count as tokens. Two empty normalised strings score 0, not 1.
* The upstream ``catch_all_exceptions`` decorator returns -1 when the wrapped
  function raises. ``exact_match_score`` and ``f1_score`` here do the same, so
  they match upstream on any input. ``score_predictions`` never reaches that
  path: it raises ``TypeError`` for a non-string answer or reference before it
  calls either metric, and it asserts that no metric result is negative.
  Upstream averages a -1 into its overall mean; this adapter refuses instead.

Adapter policy
--------------
``score_predictions`` joins prediction records to the evaluation manifest by
``question_id``. Every row stays in the all-question denominator.
``execution_failed`` rows score 0 (``scored_by: "failure_as_zero"``).
``answered`` rows and ``no_evidence`` rows score with the official metric on
the answer string (``scored_by: "official"``); a ``no_evidence`` answer must be
the empty string. The official metric gives an empty abstention EM 1 against a
reference that normalises to empty (5 of the 8498 ``qas_v2.json`` references,
none of the 70 in the first pilot). Read the aggregates together with ``counts``.

The metric is a token-overlap score on the answer string. It does not measure
faithfulness to the document.
"""

from __future__ import annotations

import logging
import math
import string
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import metadata
from typing import Any

import jieba
import regex

logger = logging.getLogger(__name__)

ADAPTER_VERSION = "1"
SCORER_NAME = "ohr-bench-official-qa"
UPSTREAM_REPO = "https://github.com/opendatalab/OHR-Bench"
UPSTREAM_COMMIT = "1f421eb428f9f5b8ac0bc8064d6ad1f13fab7af7"
UPSTREAM_PATH = "src/metric/common.py"
UPSTREAM_SHA256 = "9fe7eb527762d1be8a00d9a98a775b850bb42e9441a09db84e88996b94ef5a35"
UPSTREAM_CALLER_PATH = "src/tasks/quest_answer.py"
UPSTREAM_CALLER_SHA256 = "1feaee1780ed7e3ea597559c3c3addfeb3596cb2b679c11fba885ba40ca21b56"

STATUSES = ("answered", "no_evidence", "execution_failed")
ALL_QUESTIONS_POLICY = (
    "execution_failed scored as 0; no_evidence scored by the official metric on the empty abstention"
)

_PUNCTUATION = frozenset(string.punctuation)
_ARTICLES = regex.compile(r"\b(a|an|the)\b")
_YES_NO = ("yes", "no", "noanswer")


def _catch_all_exceptions(func):
    """Return -1 when ``func`` raises, as upstream's ``catch_all_exceptions`` does."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning(repr(exc))
            return -1

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def normalize_answer(s: str) -> str:
    """Lowercase, delete ASCII punctuation, replace a/an/the with a space, collapse whitespace."""
    lowered = s.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punctuation)
    return " ".join(no_articles.split())


def has_chn_character(s: str) -> bool:
    """Return True if any character's Unicode name contains ``CJK``."""
    for char in s:
        try:
            if "CJK" in unicodedata.name(char):
                return True
        except ValueError:
            continue
    return False


@_catch_all_exceptions
def exact_match_score(prediction: str, ground_truth: str) -> int:
    """Return 1 if the normalised strings are equal, else 0 (-1 if upstream would raise)."""
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


@_catch_all_exceptions
def f1_score(prediction: str, ground_truth: str) -> float:
    """Return the official token-overlap F1 (-1 if upstream would raise)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = 0

    if normalized_prediction in _YES_NO and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in _YES_NO and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    if has_chn_character(prediction) or has_chn_character(ground_truth):
        prediction_tokens = jieba.lcut(normalized_prediction)
        ground_truth_tokens = jieba.lcut(normalized_ground_truth)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def scorer_identity() -> dict[str, str]:
    """Identify the scorer and the upstream file it reproduces."""
    return {
        "name": SCORER_NAME,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_path": UPSTREAM_PATH,
        "upstream_sha256": UPSTREAM_SHA256,
        "adapter_version": ADAPTER_VERSION,
    }


def scorer_dependencies() -> dict[str, str]:
    """Return the installed versions of the packages that affect scores.

    ``jieba`` sets the Chinese tokenisation and ``regex`` sets the article
    pattern. Upstream pins ``jieba==0.42.1`` and ``regex==2024.7.24``. Record this
    beside ``scorer_identity()`` in a run summary.
    """
    return {name: metadata.version(name) for name in ("jieba", "regex")}


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a string, got {type(value).__name__}")
    return value


def _check_ids(predictions: Sequence[Mapping[str, Any]], evaluation_questions: Mapping[str, Any]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, record in enumerate(predictions):
        question_id = record.get("question_id")
        if not isinstance(question_id, str):
            raise ValueError(f"prediction {index} has no string question_id")
        if question_id in seen:
            duplicates.append(question_id)
        seen.add(question_id)
    if duplicates:
        raise ValueError(f"duplicate prediction question_ids: {sorted(set(duplicates))[:5]}")
    expected = set(evaluation_questions)
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(
            "prediction and evaluation question_ids differ: "
            f"{len(missing)} missing from predictions (first: {missing[:3]}), "
            f"{len(extra)} not in the evaluation manifest (first: {extra[:3]})"
        )


def _score_record(record: Mapping[str, Any], reference: str) -> tuple[int, float, str]:
    status = record["status"]
    if status == "execution_failed":
        return 0, 0.0, "failure_as_zero"
    answer = _require_str(record.get("answer"), f"answer of {status} record {record['question_id']}")
    if status == "no_evidence" and answer != "":
        raise ValueError(f"no_evidence record {record['question_id']} has a non-empty answer")
    em = exact_match_score(answer, reference)
    f1 = f1_score(answer, reference)
    if em < 0 or f1 < 0:
        raise AssertionError(f"metric returned a failure value for {record['question_id']}: em={em}, f1={f1}")
    return em, float(f1), "official"


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    evaluation_questions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Join predictions to references on ``question_id`` and score them.

    ``predictions`` are the terminal records of one run. ``evaluation_questions``
    is ``evaluation_manifest.json["questions"]``; the reference string is in its
    ``answers`` field. Rows keep the order of ``predictions``.

    Raises ``ValueError`` when the id sets differ, a prediction id repeats, a
    status is unknown, a ``doc_id`` differs between the two sides, or a
    ``no_evidence`` answer is not empty. Raises ``TypeError`` when an answer or
    reference is not a string.
    """
    if not predictions:
        raise ValueError("no predictions to score")
    _check_ids(predictions, evaluation_questions)

    rows: list[dict[str, Any]] = []
    counts = {"questions": len(predictions), "answered": 0, "no_evidence": 0, "execution_failed": 0, "abstained": 0}
    for record in predictions:
        question_id = record["question_id"]
        status = record.get("status")
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r} for {question_id}")
        evaluation = evaluation_questions[question_id]
        record_doc, evaluation_doc = record.get("doc_id"), evaluation.get("doc_id")
        if record_doc is not None and evaluation_doc is not None and record_doc != evaluation_doc:
            raise ValueError(f"doc_id mismatch for {question_id}: {record_doc!r} != {evaluation_doc!r}")
        reference = _require_str(evaluation.get("answers"), f"reference answer of {question_id}")
        em, f1, scored_by = _score_record(record, reference)
        counts[status] += 1
        if record.get("abstained") is True:
            counts["abstained"] += 1
        rows.append({"question_id": question_id, "status": status, "em": em, "f1": f1, "scored_by": scored_by})

    answered = [row for row in rows if row["status"] == "answered"]
    return {
        "scorer": scorer_identity(),
        "rows": rows,
        "counts": counts,
        "aggregates": {
            "all_questions": {
                "denominator": len(rows),
                "em": _mean([row["em"] for row in rows]),
                "f1": _mean([row["f1"] for row in rows]),
                "policy": ALL_QUESTIONS_POLICY,
            },
            "answered_only": {
                "denominator": len(answered),
                "em": _mean([row["em"] for row in answered]) if answered else None,
                "f1": _mean([row["f1"] for row in answered]) if answered else None,
            },
        },
    }
