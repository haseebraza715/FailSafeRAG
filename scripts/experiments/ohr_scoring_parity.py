"""Compare ``faar.ohr_scoring`` with the vendored upstream OHR-Bench metric.

The script loads ``OHR-Bench/src/metric/common.py`` with stub modules for
``evaluate``, ``text2vec`` and ``loguru`` (upstream imports them at module
level and the QA metrics do not use them), then scores the same pairs with
both implementations and compares ``normalize_answer``, ``exact_match_score``
and ``f1_score`` for exact equality of value and type.

Pairs compared:

* the hand-written edge cases in ``EDGE_CASES``;
* every reference in ``OHR-Bench/data/qas_v2.json`` against itself, against the
  empty string, as a reference for an empty prediction, and against
  deterministic perturbations of itself, plus cross pairs with other
  references (this is where partial F1 overlaps show up);
* the same pairs for the references of a pilot's ``evaluation_manifest.json``,
  with every pilot reference scored against every other.

It also prints reference statistics: how many references normalise to the
empty string under ``faar.metrics.normalize_text`` and under the official
normaliser, and how many contain CJK characters.

Run it offline with ``jieba==0.42.1`` and ``regex==2024.7.24`` importable:

    PYTHONPATH=src python scripts/experiments/ohr_scoring_parity.py
    PYTHONPATH=src python scripts/experiments/ohr_scoring_parity.py --write-fixture
    PYTHONPATH=src python scripts/experiments/ohr_scoring_parity.py --check-fixture

``--write-fixture`` records the upstream outputs for ``EDGE_CASES`` in
``tests/fixtures/ohr_scoring/upstream_cases.json``. ``--check-fixture`` verifies
that file against the vendored upstream. The unit tests read only the fixture,
so they do not need the vendored module or its stubs; they never import this
script. Exit status is 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import types
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar import ohr_scoring  # noqa: E402
from faar.metrics import normalize_text as old_normalize_text  # noqa: E402

UPSTREAM_MODULE = ROOT / "OHR-Bench/src/metric/common.py"
QAS_PATH = ROOT / "OHR-Bench/data/qas_v2.json"
PILOT_MANIFEST = ROOT / "results/pilots/ohr_dev_v1/evaluation_manifest.json"
FIXTURE_PATH = ROOT / "tests/fixtures/ohr_scoring/upstream_cases.json"
PINNED = {"jieba": "0.42.1", "regex": "2024.7.24"}

# (id, prediction, reference). The strings after the comment name the behaviour under test.
EDGE_CASES: list[tuple[str, Any, Any]] = [
    # identity, case, articles
    ("identical", "Paris", "Paris"),
    # reviewer mutants: articles replaced by a space, not deleted; noanswer in the yes/no rule; lower, not casefold
    ("article_between_curly_quotes", "x\u201ca\u201dy", "x\u201c\u201dy"),
    ("noanswer_with_extra_words", "noanswer here", "noanswer"),
    ("sharp_s_lower_not_casefold", "Stra\u00dfe", "STRASSE"),
    ("ligature_lower_not_casefold", "\ufb01le", "file"),
    ("case", "paris", "PARIS"),
    ("leading_article", "The Eiffel Tower", "Eiffel Tower"),
    ("inner_articles", "a cat and an owl", "cat and owl"),
    ("article_only_prediction", "the the the", "a"),
    ("article_glued_by_hyphen", "the-cat", "cat"),
    ("article_in_word", "another theory", "another theory"),
    ("articles_not_words", "athens", "thens"),
    # punctuation is deleted, not replaced by a space
    ("dotted_acronym", "U.S.A.", "USA"),
    ("hyphen_glues", "well-known", "wellknown"),
    ("hyphen_vs_space", "well-known", "well known"),
    ("apostrophe", "Bob's car", "Bobs car"),
    ("underscore_is_punctuation", "snake_case", "snakecase"),
    ("all_ascii_punctuation", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", ""),
    # numbers
    ("numeric_comma", "1,000", "1000"),
    ("numeric_comma_with_space", "1, 000", "1000"),
    ("thousands_millions", "3,736,704", "3736704"),
    ("decimal_point_deleted", "3.5", "35"),
    ("decimal_trailing_zero", "3.5", "3.50"),
    ("percent_sign", "35.93%", "3593"),
    ("percent_word", "35.93%", "35.93 percent"),
    ("currency", "$10.00", "1000"),
    ("currency_vs_plain", "$10.00", "10"),
    ("unit_spaced_vs_glued", "5 km", "5km"),
    ("unit_glued_vs_spaced", "5km", "5 km"),
    ("negative_number", "-5", "5"),
    ("letters_digits_glued", "A1", "a 1"),
    # list-like answer strings in the actual reference representation
    ("list_no_space_vs_space", "Peak Clipping, Valley Filling, Load Shifting", "Peak Clipping,Valley Filling,Load Shifting"),
    ("list_no_space_exact", "Peak Clipping,Valley Filling,Load Shifting", "Peak Clipping,Valley Filling,Load Shifting"),
    ("list_numbers_spaced", "341195, 339502, 339909", "341195,339502,339909"),
    ("list_numbers_glued", "341195,339502,339909", "341195,339502,339909"),
    ("list_mixed_units", "3,736,704 shares and 35.93%", "3,736,704 shares, 35.93%"),
    ("list_json_style", '["A", "B", "C"]', "A,B,C"),
    # containment and partial overlap
    ("prediction_contains_reference", "Paris, France", "Paris"),
    ("sentence_around_number", "The answer is 842.", "842"),
    ("reference_contains_prediction", "Paris", "Paris, France"),
    ("repeated_tokens", "x x x y", "x y"),
    ("no_overlap", "alpha", "beta"),
    ("whitespace_variants", "a  b\t c", "b c"),
    ("newline_variants", "line one\nline two", "line one line two"),
    # yes / no / noanswer
    ("yes_same", "Yes", "yes"),
    ("yes_punct", "Yes.", "Yes"),
    ("yes_with_tail", "Yes, it is", "Yes"),
    ("reference_yes_prediction_yes_tail", "Yes", "Yes, it is"),
    ("no_vs_yes", "No", "Yes"),
    ("no_same", "no", "no"),
    ("noanswer_two_words", "no answer", "noanswer"),
    ("noanswer_glued_by_punct", "no-answer", "noanswer"),
    ("noanswer_same", "NoAnswer", "noanswer"),
    ("not_answerable_vs_yes", "Not answerable", "Yes"),
    ("not_answerable_vs_text", "Not answerable", "Kenneth Turpin"),
    ("yes_vs_chinese", "yes", "是"),
    # empty and degenerate
    ("empty_empty", "", ""),
    ("empty_prediction", "", "abc"),
    ("empty_reference", "abc", ""),
    ("blank_vs_empty", "   ", ""),
    ("punct_vs_empty", "...", ""),
    ("article_vs_empty", "the", ""),
    ("punct_same", "?", "?"),
    ("article_article", "the", "a"),
    ("empty_prediction_chinese_reference", "", "北京"),
    ("chinese_prediction_empty_reference", "北京", ""),
    # non-ASCII text that ASCII-only handling keeps
    ("accent_kept", "café", "cafe"),
    ("accent_case", "Café", "café"),
    ("fullwidth_kept", "ＡＢＣ", "ABC"),
    ("curly_quotes_kept", "“quoted”", "quoted"),
    ("em_dash_kept", "a—b", "ab"),
    ("ellipsis_kept", "wait…", "wait"),
    ("dotted_capital_i", "İstanbul", "istanbul"),
    ("combining_accent_after_article", "the\u0301 cat", "cat"),
    ("zero_width_joiner_after_article", "the\u200d cat", "cat"),
    ("article_before_accented_letter", "the école", "école"),
    ("non_breaking_space", "a\u00a0b", "b"),
    # Chinese
    ("zh_identical", "北京", "北京"),
    ("zh_longer_prediction", "北京市", "北京"),
    ("zh_punct_removed_not", "他们中有4人担任村党支部副书记，10人当选为村党支部委员。", "他们中有4人担任村党支部副书记10人当选为村党支部委员"),
    ("zh_list_enumeration_comma", "发行债券、上市融资和设立新型农村金融机构。", "发行债券、上市融资、设立新型农村金融机构"),
    ("zh_fullwidth_comma_only", "，", "，"),
    ("zh_full_stop_only", "。", "。"),
    ("zh_vs_en", "北京", "Beijing"),
    ("zh_space_variant", "北京 上海", "北京上海"),
    ("zh_space_identical", "北京 上海", "北京 上海"),
    ("zh_article_prefix", "the 北京", "北京"),
    ("zh_digit_spacing", "4人", "4 人"),
    ("zh_partial_sentence", "汉斯·斯隆在牙买加岛进行了植物调查", "汉斯·斯隆和杰纳斯·哈洛在牙买加岛进行了植物调查，汉斯·斯隆出版了《牙买加博物志》第1卷。"),
    ("zh_extension_b_ideograph", "\U00020000", "\U00020000"),
    # mixed language
    ("mixed_reordered", "Beijing 北京", "北京 Beijing"),
    ("mixed_extra_tokens", "iPhone 15 手机", "iPhone 15"),
    ("mixed_spacing", "GDP增长5%", "GDP 增长 5%"),
    ("mixed_english_prediction_zh_reference", "The capital is Beijing", "北京"),
    ("mixed_zh_prediction_en_reference", "北京", "The capital is Beijing"),
    # scripts whose Unicode names do not contain CJK
    ("kana_identical", "こんにちは", "こんにちは"),
    ("kana_vs_kana_tail", "こんにちは", "こんにちは 世界"),
    ("kana_only_tail", "こんにちは", "こんにちは せかい"),
    ("hangul_tail", "안녕하세요", "안녕하세요 세계"),
    ("katakana_halfwidth", "ｶﾅ", "ｶﾅ"),
    # non-string input reaches the -1 path
    ("none_prediction", None, "a"),
    ("none_reference", "a", None),
    ("int_prediction", 5, "5"),
    ("int_reference", "5", 5),
    ("list_prediction", ["a"], "a"),
]

# Cases the JSON fixture cannot hold (a lone surrogate does not survive UTF-8).
SCRIPT_ONLY_CASES: list[tuple[str, Any, Any]] = [
    ("lone_surrogate", "\ud800", "\ud800"),
    ("lone_surrogate_prefix", "a\ud800b", "ab"),
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_upstream(path: Path = UPSTREAM_MODULE) -> types.ModuleType:
    """Import the vendored upstream ``common.py`` with stubs for its unused heavy imports."""
    stubs = {
        "evaluate": types.ModuleType("evaluate"),
        "text2vec": types.ModuleType("text2vec"),
        "loguru": types.ModuleType("loguru"),
    }
    stubs["text2vec"].Similarity = object
    stubs["loguru"].logger = types.SimpleNamespace(warning=lambda *a, **k: None, error=lambda *a, **k: None)
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("ohr_upstream_metric_common", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def outputs(module: Any, prediction: Any, reference: Any) -> dict[str, Any]:
    """Run the three functions; keep type with value so 0 and 0.0 stay distinguishable."""
    norm_p = _try(module.normalize_answer, prediction)
    norm_r = _try(module.normalize_answer, reference)
    return {
        "normalize_prediction": norm_p,
        "normalize_reference": norm_r,
        "em": _typed(module.exact_match_score(prediction, reference)),
        "f1": _typed(module.f1_score(prediction, reference)),
    }


def _try(func: Callable[[Any], Any], value: Any) -> Any:
    try:
        return func(value)
    except Exception as exc:  # the non-string cases raise; record the class
        return f"<raises {type(exc).__name__}>"


def _typed(value: Any) -> list[Any]:
    return [type(value).__name__, value]


def perturbations(reference: str) -> dict[str, str]:
    tokens = reference.split()
    return {
        "self": reference,
        "dropped_last_token": " ".join(tokens[:-1]),
        "added_article": "The " + reference,
        "removed_commas": reference.replace(",", "").replace("，", ""),
        "uppercased": reference.upper(),
        "wrapped_sentence": f"The answer is {reference}.",
    }


def build_pairs(references: list[str], cross_offsets: tuple[int, ...]) -> list[tuple[str, str, str]]:
    """Return (kind, prediction, reference) triples for a list of reference strings."""
    pairs: list[tuple[str, str, str]] = []
    for index, reference in enumerate(references):
        for name, prediction in perturbations(reference).items():
            pairs.append((name, prediction, reference))
        pairs.append(("empty_prediction", "", reference))
        pairs.append(("empty_reference", reference, ""))
        for offset in cross_offsets:
            pairs.append((f"cross_{offset}", references[(index + offset) % len(references)], reference))
    return pairs


def compare(mine: Any, upstream: Any, pairs: list[tuple[str, Any, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for kind, prediction, reference in pairs:
        got = outputs(mine, prediction, reference)
        want = outputs(upstream, prediction, reference)
        if got != want:
            mismatches.append({"kind": kind, "prediction": prediction, "reference": reference, "mine": got, "upstream": want})
    return {"pairs": len(pairs), "mismatches": len(mismatches), "first_mismatches": mismatches[:5]}


def reference_statistics(references: list[str]) -> dict[str, Any]:
    old_empty = [r for r in references if old_normalize_text(r) == ""]
    official_empty = [r for r in references if ohr_scoring.normalize_answer(r) == ""]
    cjk = [r for r in references if ohr_scoring.has_chn_character(r)]
    return {
        "references": len(references),
        "empty_under_old_normalizer": len(old_empty),
        "empty_under_official_normalizer": len(official_empty),
        "contains_cjk": len(cjk),
        "cjk_and_empty_under_old_normalizer": sum(1 for r in cjk if old_normalize_text(r) == ""),
        "cjk_and_nonempty_under_old_normalizer": sum(1 for r in cjk if old_normalize_text(r) != ""),
        "empty_under_old_but_not_cjk": sum(1 for r in old_empty if not ohr_scoring.has_chn_character(r)),
        "yes_no_noanswer_under_official_normalizer": sum(
            1 for r in references if ohr_scoring.normalize_answer(r) in ("yes", "no", "noanswer")
        ),
        "empty_under_official_examples": official_empty[:5],
        "empty_under_old_not_cjk_examples": [r for r in old_empty if not ohr_scoring.has_chn_character(r)][:5],
    }


def fixture_payload(upstream: Any) -> dict[str, Any]:
    return {
        "description": "Upstream OHR-Bench outputs for edge cases. Regenerate with scripts/experiments/ohr_scoring_parity.py --write-fixture.",
        "upstream": {
            "repo": ohr_scoring.UPSTREAM_REPO,
            "commit": ohr_scoring.UPSTREAM_COMMIT,
            "path": ohr_scoring.UPSTREAM_PATH,
            "sha256": file_sha256(UPSTREAM_MODULE),
        },
        "recorded_with": {
            "python": sys.version.split()[0],
            "jieba": metadata.version("jieba"),
            "regex": metadata.version("regex"),
        },
        "value_format": "[python type name, value]",
        "cases": [
            {"id": case_id, "prediction": prediction, "reference": reference, "upstream": outputs(upstream, prediction, reference)}
            for case_id, prediction, reference in EDGE_CASES
        ],
    }


def format_fixture(payload: dict[str, Any]) -> str:
    """Write the header indented and one case per line, so a diff shows changed cases."""
    head = {key: value for key, value in payload.items() if key != "cases"}
    text = json.dumps(head, indent=1, ensure_ascii=False)[:-2]
    lines = ",\n".join("  " + json.dumps(case, ensure_ascii=False) for case in payload["cases"])
    return f'{text},\n "cases": [\n{lines}\n ]\n}}\n'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-fixture", action="store_true", help="record upstream outputs for EDGE_CASES")
    mode.add_argument("--check-fixture", action="store_true", help="verify the committed fixture against upstream")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--qas", type=Path, default=QAS_PATH)
    parser.add_argument("--pilot-manifest", type=Path, default=PILOT_MANIFEST)
    args = parser.parse_args()

    if file_sha256(UPSTREAM_MODULE) != ohr_scoring.UPSTREAM_SHA256:
        print("vendored upstream file differs from the sha256 recorded in faar.ohr_scoring", file=sys.stderr)
        return 1
    upstream = load_upstream()

    if args.write_fixture:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(format_fixture(fixture_payload(upstream)), encoding="utf-8")
        print(f"wrote {args.fixture}")
        return 0
    if args.check_fixture:
        committed = json.loads(args.fixture.read_text(encoding="utf-8"))
        fresh = json.loads(json.dumps(fixture_payload(upstream), ensure_ascii=False))
        same = committed["cases"] == fresh["cases"] and committed["upstream"] == fresh["upstream"]
        print("fixture matches upstream" if same else "fixture differs from upstream")
        return 0 if same else 1

    versions = {name: metadata.version(name) for name in PINNED}
    report: dict[str, Any] = {
        "upstream_sha256": file_sha256(UPSTREAM_MODULE),
        "versions": versions,
        "pinned_versions_in_use": versions == PINNED,
    }
    edge = [(case_id, p, r) for case_id, p, r in EDGE_CASES + SCRIPT_ONLY_CASES]
    report["edge_cases"] = compare(ohr_scoring, upstream, edge)

    qas = json.loads(args.qas.read_text(encoding="utf-8"))
    qas_refs = [item["answers"] for item in qas]
    report["qas_v2"] = compare(ohr_scoring, upstream, build_pairs(qas_refs, cross_offsets=(1, 7, 101)))
    report["qas_v2"]["reference_statistics"] = reference_statistics(qas_refs)

    if args.pilot_manifest.exists():
        questions = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))["questions"]
        pilot_refs = [question["answers"] for question in questions.values()]
        pilot_pairs = build_pairs(pilot_refs, cross_offsets=())
        pilot_pairs += [("all_cross", p, r) for p in pilot_refs for r in pilot_refs]
        report["pilot_references"] = compare(ohr_scoring, upstream, pilot_pairs)
        report["pilot_references"]["reference_statistics"] = reference_statistics(pilot_refs)

    report["kinds_compared"] = dict(Counter(kind for kind, _, _ in build_pairs(qas_refs[:1], (1, 7, 101))))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    total = sum(section["mismatches"] for key, section in report.items() if isinstance(section, dict) and "mismatches" in section)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
