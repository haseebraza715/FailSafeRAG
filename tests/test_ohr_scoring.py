"""Tests for the OHR-Bench answer scorer in ``faar.ohr_scoring``.

Failure list, written before the scorer. Each item is a way this scorer could
be wrong, and each is covered by a test group below.

1. Wrong helper. The adapter uses ``em``, ``em_strict``, ``exact_match_strict_score``,
   ``f1_en`` or ``f1_zh`` instead of the ``exact_match_score`` and ``f1_score``
   that ``QuestAnswer.scoring`` calls. ``em`` is a substring test and ``f1_zh``
   counts characters, so the answers differ on inputs the fixture covers.
2. Lost normalisation quirk. The order (lower, punctuation, articles, spaces)
   changes; punctuation removal becomes Unicode-aware or inserts a space (so
   ``1,000`` no longer equals ``1000``, and ``A,B`` no longer glues to ``ab``);
   accents or full-width forms get folded; the article pattern stops matching
   after punctuation removal.
3. Lost F1 quirk. The yes/no/noanswer rule is applied to the raw string or on
   one side only; empty against empty stops giving EM 1 with F1 0; a
   zero-overlap case returns something other than 0.
4. Wrong CJK path. ``has_chn_character`` tests the normalised string instead of
   the raw one, tests one side only, or accepts kana, Hangul or CJK
   punctuation (whose Unicode names do not contain ``CJK``). ``jieba.lcut`` is
   skipped, or run on the raw string, or replaced by character splitting.
5. The -1 path. Non-string input makes the upstream wrapper return -1, and a -1
   averaged into a mean silently corrupts it. The adapter must refuse
   non-string answers and references before scoring, so -1 never reaches a row.
6. Join errors. Prediction and evaluation ids differ, a prediction id repeats,
   a record has an unknown status, a ``doc_id`` differs between the two sides,
   or the reference is not a string. Each must raise ``ValueError`` and score
   nothing.
7. Denominator errors. ``execution_failed`` rows drop out of the all-question
   denominator or score with the official metric; ``no_evidence`` rows score
   as 0 without the metric (an empty abstention against an empty-normalising
   reference gives EM 1 upstream); ``answered_only`` includes other statuses,
   or reports 0.0 instead of ``None`` when nothing was answered; the counts of
   answered, no_evidence, execution_failed and abstained are conflated.
8. Identity drift. The recorded upstream sha256 stops matching the vendored
   file, or ``scorer_identity()`` gains or loses a contract key.
9. Side effects. Scoring mutates its inputs, reorders rows, or writes files.
10. Old-normaliser claim. ``faar.metrics.normalize_text`` erases everything
    outside ``[a-z0-9]``; the study brief must not describe when that yields a
    perfect score wrongly. The tests pin the exact condition.

Expected values for the upstream functions come from
``tests/fixtures/ohr_scoring/upstream_cases.json``. ``scripts/experiments/ohr_scoring_parity.py``
records them by loading the vendored upstream module with stubs for
``evaluate``, ``text2vec`` and ``loguru``. The tests never import that script
or the vendored module's heavy dependencies; only the hash test reads the
vendored file as bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from faar import metrics, ohr_scoring
from faar.ohr_scoring import (
    exact_match_score,
    f1_score,
    has_chn_character,
    normalize_answer,
    score_predictions,
    scorer_identity,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/ohr_scoring/upstream_cases.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _typed(value):
    return [type(value).__name__, value]


def _normalized_or_error(value):
    try:
        return normalize_answer(value)
    except Exception as exc:
        return f"<raises {type(exc).__name__}>"


# 1, 2, 3, 4, 5: agreement with recorded upstream outputs


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_matches_recorded_upstream_output(case):
    expected = case["upstream"]
    prediction, reference = case["prediction"], case["reference"]
    assert _normalized_or_error(prediction) == expected["normalize_prediction"]
    assert _normalized_or_error(reference) == expected["normalize_reference"]
    assert _typed(exact_match_score(prediction, reference)) == expected["em"]
    assert _typed(f1_score(prediction, reference)) == expected["f1"]


def test_fixture_covers_the_required_behaviour_groups():
    ids = {case["id"] for case in CASES}
    required = {
        "identical", "numeric_comma", "decimal_point_deleted", "percent_sign", "currency", "unit_spaced_vs_glued",
        "yes_same", "noanswer_two_words", "empty_empty", "empty_prediction", "empty_reference", "zh_identical",
        "mixed_reordered", "kana_identical", "list_numbers_spaced", "none_prediction", "leading_article",
    }
    assert required <= ids


# 2: normalisation quirks stated directly, not through the fixture


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("The  Eiffel\tTower", "eiffel tower"),
        ("1,000", "1000"),
        ("1, 000", "1 000"),
        ("3.5", "35"),
        ("A,B", "ab"),
        ("A, B", "b"),  # "a" is an article once punctuation is gone
        ("well-known", "wellknown"),
        ("the-cat", "thecat"),
        ("Bob's", "bobs"),
        ("snake_case", "snakecase"),
        ("café", "café"),
        ("“quoted”", "“quoted”"),
        ("ＡＢＣ", "ａｂｃ"),
        ("a—b", "—b"),
        ("another theory", "another theory"),
        ("", ""),
        ("the a an", ""),
    ],
)
def test_normalize_answer_keeps_upstream_quirks(raw, normalized):
    assert normalize_answer(raw) == normalized


def test_article_pattern_uses_unicode_word_boundaries_like_regex_module():
    # stdlib re treats a combining accent as a non-word character and would strip "the".
    assert normalize_answer("the\u0301 cat") == "the\u0301 cat"
    assert normalize_answer("the\u200d cat") == "the\u200d cat"


def test_list_answers_are_glued_or_split_by_the_comma_spacing():
    reference = "341195,339502,339909"
    assert exact_match_score(reference, reference) == 1
    assert exact_match_score("341195, 339502, 339909", reference) == 0
    assert f1_score("341195, 339502, 339909", reference) == 0
    assert exact_match_score("341195339502339909", reference) == 1


# 3: F1 quirks


def test_empty_against_empty_is_exact_match_but_zero_f1():
    assert exact_match_score("", "") == 1
    assert f1_score("", "") == 0
    assert exact_match_score("the", "...") == 1
    assert f1_score("the", "...") == 0


# 4: CJK path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("北京", True),
        ("Beijing 北京", True),
        ("\U00020000", True),
        ("", False),
        ("Beijing", False),
        ("こんにちは", False),  # hiragana
        ("안녕", False),  # hangul
        ("，", False),  # CJK punctuation: fullwidth comma
        ("\ud800", False),  # lone surrogate has no Unicode name
    ],
)
def test_has_chn_character_tests_unicode_names(text, expected):
    assert has_chn_character(text) is expected


def test_chinese_punctuation_is_not_removed():
    assert normalize_answer("上海，北京。") == "上海，北京。"
    assert exact_match_score("北京。", "北京") == 0


# 1: the adapter calls the official functions


def _question(question_id, answer, doc_id="doc/a"):
    return {"answers": answer, "doc_id": doc_id, "answer_form": "String"}


def _record(question_id, status, answer, doc_id="doc/a", abstained=False):
    return {
        "schema_version": 1,
        "question_id": question_id,
        "doc_id": doc_id,
        "status": status,
        "answer": answer,
        "abstained": abstained,
    }


def test_score_predictions_routes_through_the_official_functions(monkeypatch):
    monkeypatch.setattr(ohr_scoring, "exact_match_score", lambda p, r: 1)
    monkeypatch.setattr(ohr_scoring, "f1_score", lambda p, r: 0.25)
    result = score_predictions([_record("q1", "answered", "zzz")], {"q1": _question("q1", "abc")})
    assert (result["rows"][0]["em"], result["rows"][0]["f1"]) == (1, 0.25)


# 5: the -1 path


@pytest.mark.parametrize(("status", "answer"), [("answered", None), ("no_evidence", 5)])
def test_score_predictions_refuses_non_string_answers(status, answer):
    with pytest.raises(TypeError):
        score_predictions([_record("q1", status, answer)], {"q1": _question("q1", "abc")})


def test_score_predictions_refuses_non_string_reference():
    with pytest.raises(TypeError):
        score_predictions([_record("q1", "answered", "a")], {"q1": _question("q1", 5)})
    with pytest.raises(TypeError):
        score_predictions([_record("q1", "answered", "a")], {"q1": {"doc_id": "doc/a"}})


def test_no_evidence_answer_must_be_empty():
    with pytest.raises(ValueError, match="non-empty"):
        score_predictions([_record("q1", "no_evidence", "guess", abstained=True)], {"q1": _question("q1", "abc")})


# 6: join errors


def test_missing_prediction_is_refused():
    evaluation = {"q1": _question("q1", "a"), "q2": _question("q2", "b")}
    with pytest.raises(ValueError, match="1 missing from predictions"):
        score_predictions([_record("q1", "answered", "a")], evaluation)


def test_extra_prediction_is_refused():
    with pytest.raises(ValueError, match="1 not in the evaluation manifest"):
        score_predictions([_record("q1", "answered", "a"), _record("q9", "answered", "a")], {"q1": _question("q1", "a")})


def test_duplicate_prediction_is_refused_even_when_the_id_sets_match():
    evaluation = {"q1": _question("q1", "a")}
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions([_record("q1", "answered", "a"), _record("q1", "answered", "b")], evaluation)


def test_unknown_status_and_missing_id_are_refused():
    evaluation = {"q1": _question("q1", "a")}
    with pytest.raises(ValueError, match="unknown status"):
        score_predictions([_record("q1", "skipped", "a")], evaluation)
    with pytest.raises(ValueError, match="question_id"):
        score_predictions([{"status": "answered", "answer": "a"}], evaluation)


def test_doc_id_mismatch_is_refused():
    with pytest.raises(ValueError, match="doc_id mismatch"):
        score_predictions([_record("q1", "answered", "a", doc_id="doc/other")], {"q1": _question("q1", "a")})


def test_empty_prediction_list_is_refused():
    with pytest.raises(ValueError, match="no predictions"):
        score_predictions([], {})


# 7 and 9: denominators, counts, order, purity


def _mixed_run():
    evaluation = {
        "q1": _question("q1", "Paris"),
        "q2": _question("q2", "842"),
        "q3": _question("q3", "Kenneth Turpin"),
        "q4": _question("q4", "yes"),
        "q5": _question("q5", "[***]"),  # normalises to empty
    }
    predictions = [
        _record("q4", "execution_failed", None),
        _record("q1", "answered", "The Paris"),
        _record("q2", "answered", "The answer is 842."),
        _record("q3", "no_evidence", "", abstained=True),
        _record("q5", "no_evidence", "", abstained=True),
    ]
    return predictions, evaluation


def test_all_question_and_answered_only_aggregates():
    predictions, evaluation = _mixed_run()
    result = score_predictions(predictions, evaluation)

    assert [row["question_id"] for row in result["rows"]] == ["q4", "q1", "q2", "q3", "q5"]
    rows = {row["question_id"]: row for row in result["rows"]}
    assert rows["q4"] == {"question_id": "q4", "status": "execution_failed", "em": 0, "f1": 0.0, "scored_by": "failure_as_zero"}
    assert (rows["q1"]["em"], rows["q1"]["f1"], rows["q1"]["scored_by"]) == (1, 1.0, "official")
    assert (rows["q2"]["em"], rows["q2"]["f1"]) == (0, 0.5)
    assert (rows["q3"]["em"], rows["q3"]["f1"], rows["q3"]["scored_by"]) == (0, 0.0, "official")
    # The official metric gives an empty abstention EM 1 against a reference that normalises to empty.
    assert (rows["q5"]["em"], rows["q5"]["f1"], rows["q5"]["scored_by"]) == (1, 0.0, "official")

    assert result["counts"] == {"questions": 5, "answered": 2, "no_evidence": 2, "execution_failed": 1, "abstained": 2}
    all_questions = result["aggregates"]["all_questions"]
    assert all_questions["denominator"] == 5
    assert all_questions["em"] == pytest.approx(2 / 5)
    assert all_questions["f1"] == pytest.approx(1.5 / 5)
    assert all_questions["policy"] == ohr_scoring.ALL_QUESTIONS_POLICY
    answered_only = result["aggregates"]["answered_only"]
    assert answered_only["denominator"] == 2
    assert answered_only["em"] == pytest.approx(0.5)
    assert answered_only["f1"] == pytest.approx(0.75)
    assert result["scorer"] == scorer_identity()


def test_execution_failed_stays_in_the_denominator_whatever_its_answer_field():
    evaluation = {"q1": _question("q1", "Paris")}
    record = _record("q1", "execution_failed", "Paris")  # a matching answer must not be scored
    result = score_predictions([record], evaluation)
    assert (result["rows"][0]["em"], result["rows"][0]["f1"]) == (0, 0.0)
    assert result["aggregates"]["all_questions"]["denominator"] == 1
    assert result["aggregates"]["all_questions"]["em"] == 0.0


def test_answered_only_is_none_when_nothing_was_answered():
    evaluation = {"q1": _question("q1", "a"), "q2": _question("q2", "b")}
    predictions = [_record("q1", "execution_failed", None), _record("q2", "no_evidence", "", abstained=True)]
    result = score_predictions(predictions, evaluation)
    assert result["aggregates"]["answered_only"] == {"denominator": 0, "em": None, "f1": None}
    assert result["aggregates"]["all_questions"]["denominator"] == 2


def test_abstained_is_counted_from_the_flag_separately_from_no_evidence():
    evaluation = {"q1": _question("q1", "a"), "q2": _question("q2", "b")}
    evaluation["q3"] = _question("q3", "c")
    predictions = [
        _record("q1", "answered", "a", abstained=True),
        _record("q2", "no_evidence", "", abstained=False),
        _record("q3", "answered", "c", abstained=False),
    ]
    counts = score_predictions(predictions, evaluation)["counts"]
    assert (counts["answered"], counts["no_evidence"], counts["abstained"]) == (2, 1, 1)


def test_upstream_valid_only_drops_blank_answers_like_evaluator_remove_invalid():
    """Upstream evaluator.py keeps only results whose generated text is non-blank before compute_overall."""
    evaluation = {q: _question(q, "abc") for q in ("q1", "q2", "q3", "q4")}
    predictions = [
        _record("q1", "answered", "abc"),
        _record("q2", "answered", "  "),
        _record("q3", "no_evidence", "", abstained=True),
        _record("q4", "execution_failed", None),
    ]
    aggregates = score_predictions(predictions, evaluation)["aggregates"]
    assert aggregates["answered_only"]["denominator"] == 2
    assert aggregates["upstream_valid_only"]["denominator"] == 1
    assert aggregates["upstream_valid_only"]["em"] == 1.0
    assert "remove_invalid" in aggregates["upstream_valid_only"]["policy"]
    assert aggregates["all_questions"]["denominator"] == 4


def test_answered_empty_string_is_scored_and_stays_answered():
    result = score_predictions([_record("q1", "answered", "")], {"q1": _question("q1", "abc")})
    assert result["rows"][0]["scored_by"] == "official"
    assert result["counts"]["answered"] == 1
    assert result["aggregates"]["answered_only"]["denominator"] == 1


def test_score_predictions_does_not_mutate_inputs():
    predictions, evaluation = _mixed_run()
    before = copy.deepcopy((predictions, evaluation))
    score_predictions(predictions, evaluation)
    assert (predictions, evaluation) == before


def test_result_keys_follow_the_contract():
    predictions, evaluation = _mixed_run()
    result = score_predictions(predictions, evaluation)
    assert set(result) == {"scorer", "rows", "counts", "aggregates"}
    assert set(result["aggregates"]) == {"all_questions", "answered_only", "upstream_valid_only"}
    assert set(result["aggregates"]["upstream_valid_only"]) == {"denominator", "em", "f1", "policy"}
    assert set(result["aggregates"]["all_questions"]) == {"denominator", "em", "f1", "policy"}
    assert set(result["aggregates"]["answered_only"]) == {"denominator", "em", "f1"}
    assert all(set(row) == {"question_id", "status", "em", "f1", "scored_by"} for row in result["rows"])
    assert json.loads(json.dumps(result)) == result


# 8: identity


def test_scorer_identity_has_exactly_the_contract_keys():
    identity = scorer_identity()
    assert set(identity) == {"name", "upstream_repo", "upstream_commit", "upstream_path", "upstream_sha256", "adapter_version"}
    assert all(isinstance(value, str) and value for value in identity.values())
    assert identity["upstream_commit"] == "1f421eb428f9f5b8ac0bc8064d6ad1f13fab7af7"


def test_fixture_records_the_same_upstream_as_the_scorer_identity():
    upstream = json.loads(FIXTURE.read_text(encoding="utf-8"))["upstream"]
    identity = scorer_identity()
    assert upstream == {
        "repo": identity["upstream_repo"],
        "commit": identity["upstream_commit"],
        "path": identity["upstream_path"],
        "sha256": identity["upstream_sha256"],
    }


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("OHR-Bench/src/metric/common.py", ohr_scoring.UPSTREAM_SHA256),
        ("OHR-Bench/src/tasks/quest_answer.py", ohr_scoring.UPSTREAM_CALLER_SHA256),
    ],
)
def test_recorded_upstream_hashes_match_the_vendored_files(relative, expected):
    path = ROOT / relative
    if not path.exists():
        pytest.skip("OHR-Bench is not checked out")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


# 10: the old FAAR normaliser


def test_old_normalizer_gives_a_perfect_score_only_when_both_sides_normalize_to_empty():
    # Two different Chinese strings, or a Chinese reference and an empty prediction, score perfectly.
    assert metrics.exact_match("上海", "北京") == 1.0
    assert metrics.token_f1("上海", "北京") == 1.0
    assert metrics.exact_match("", "北京") == 1.0
    # A prediction with any ASCII letter or digit does not: the reference normalises to empty, the prediction does not.
    assert metrics.exact_match("Beijing", "北京") == 0.0
    assert metrics.token_f1("Beijing", "北京") == 0.0
    assert metrics.exact_match("5", "北京") == 0.0
    # A Chinese reference that contains ASCII digits keeps them, so it is not a wildcard.
    assert metrics.exact_match("上海", "4人") == 0.0
    assert metrics.exact_match("4", "4人") == 1.0
    # Accented letters are erased, not folded.
    assert metrics.exact_match("café", "caf") == 1.0


def test_official_metric_does_not_treat_chinese_as_empty():
    assert (exact_match_score("上海", "北京"), f1_score("上海", "北京")) == (0, 0)
    assert exact_match_score("", "北京") == 0
