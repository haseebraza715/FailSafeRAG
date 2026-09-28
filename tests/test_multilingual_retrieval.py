"""Multilingual retrieval for the offline engineering path (tokenizer ``multilingual-v1``).

Ways the change could fail, written before the code. Each test names the item it covers.

Tokens
  K1. A Chinese or mixed question has no token, so its hits are ranked by page position.
  K2. A number, ``%`` or ``$`` is lost or split, or full-width forms do not fold.
  K3. English text tokenizes differently from before without a reason.
  K4. Queries and documents are tokenized by different rules.
  K5. Symbol-only or empty text raises, or gets a token.
Chunks
  C1. An unspaced Chinese page is one unbounded chunk.
  C2. A window skips text, or the loop does not end (overlap at or above the size, size 1).
  C3. English pages chunk differently from before.
  C4. The default of a shared entry point changes, so ``graph`` and ``benchmarks`` change.
Retrieval
  R1. The relevant page or chunk is not ranked first in Chinese, mixed or English documents.
  R2. Repeated terms, numbers or symbol-only chunks upset the ranking.
  R3. A hit comes from another document.
  R4. Order or scores differ between runs or between hash seeds.
Runner
  N1. Han text is reported as an OCR condition or as ``no_retrieval_tokens``, or the reverse.
  N2. The policy ids are missing from ``run_config.json`` or from the fingerprint.
  N3. Generation needs an evaluation input.

The tests in "Reproduction" failed on the ASCII-only tokenizer, in 8 cases: the six
Chinese questions had zero query tokens, five of their relevant pages and the tie case
(same Latin words on two pages) were not ranked first, and an unspaced Chinese page
stayed one chunk. They call only the runner entry point that existed before the change.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_pilot_runner import DOC_A, PILOT_ID, Project

from faar import pilot_runner as pr
from faar.chunking import build_chunks, build_page_chunks
from faar.retrieval import HybridRetriever, LocalHashEmbedder
from faar.settings import RetrievalSettings
from faar.text_units import (
    CJK_CHARS_PER_WORD,
    LEGACY_CHUNK_POLICY,
    LEGACY_TEXT_POLICY,
    LEGACY_TOKENIZER,
    MULTILINGUAL_CHUNK_POLICY,
    MULTILINGUAL_TEXT_POLICY,
    MULTILINGUAL_TOKENIZER,
    TextPolicy,
    chunk_spans,
    tokenize,
)
from faar.types import Chunk

SRC = Path(__file__).resolve().parents[1] / "src"

# ---------------------------------------------------------------------------
# Synthetic documents. Every page of a document shares some vocabulary with the others.
# ---------------------------------------------------------------------------

ZH_DOC = "news/zh_report"
ZH_PAGES = [
    "仔猪沙门氏菌病主要通过污染的饲料和饮水传播，急性病例表现为高热和腹泻。研究报告建议养殖场每周消毒圈舍。",
    "我国三大产棉区包括黄河流域棉区、长江流域棉区和西北内陆棉区，其中新疆的棉花产量最高。研究报告显示产量逐年上升。",
    "汉斯·斯隆在牙买加岛采集了八百种植物标本，并于1707年出版了考察报告。考察报告记录了当地的气候。",
    "在审题过程中，应当先通读全题，再圈出关键词，最后检查答案是否完整。这些方法有助于提高答题质量。",
    "长江是中国最长的河流，全长约六千三百公里，流经十一个省级行政区。研究报告指出其流域人口最多。",
    "黄河流域的农业研究报告显示，小麦和玉米是主要作物，华北平原的作物熟制为一年两熟。",
]
ZH_QUESTIONS = [
    ("zh-pig", "沙门氏菌病是如何传播的？", 0),
    ("zh-cotton", "我国三大产棉区包括哪些地区？", 1),
    ("zh-botanist", "汉斯·斯隆在牙买加岛采集了多少种植物标本？", 2),
    ("zh-exam", "在审题过程中应该遵循哪些步骤？", 3),
    ("zh-river", "长江全长大约多少公里？", 4),
    ("zh-crop", "华北平原的作物熟制是什么？", 5),
]

MIXED_DOC = "finance/mixed_report"
MIXED_PAGES = [
    "2022年公司营收为3.2亿元，同比增长3%。主要业务为国内零售，Widget X 的销量下降。",
    "2023年公司营收为4.8亿元，同比增长12%。主要增长来自 Widget X 出口，欧洲市场需求强劲。",
    "2023 annual report: net income was $5.1 million, up 8% year over year, mainly from exports.",
    "Gadget Y 的国内零售收入为1.1亿元，占营收的23%。",
]
MIXED_QUESTIONS = [
    ("mx-growth", "2023年营收同比增长多少%？", 1),
    ("mx-old", "2022年营收同比增长了多少？", 0),
    ("mx-english", "What was the net income in 2023?", 2),
    ("mx-export", "Widget X出口的增长来自哪个市场？", 1),
    ("mx-gadget", "Gadget Y 的零售收入占营收多少？", 3),
]

# The Latin words and the % sign are the same on both pages; only the Chinese words tell them apart.
TIE_DOC = "finance/tie_report"
TIE_PAGES = [
    "Widget X 的国内零售下降了5%，主要原因是门店关闭。",
    "Widget X 的出口增长了12%，主要来自欧洲市场。",
]
TIE_QUESTIONS = [
    ("mx-tie-export", "Widget X 出口增长了多少%？", 1),
    ("mx-tie-retail", "Widget X 国内零售为什么下降？", 0),
]

EN_DOC = "law/en_contract"
EN_PAGES = [
    "The supplier delivers the goods within thirty days of the purchase order. Late delivery is reported in writing.",
    "The warranty period is twelve months from the delivery date, and repairs are free of charge during the warranty.",
    "Either party may terminate this agreement with sixty days written notice. The notice must be in writing.",
    "The buyer pays 500 dollars per unit. Payment is due within thirty days of the invoice.",
]
EN_QUESTIONS = [
    ("en-warranty", "What is the warranty period from the delivery date?", 1),
    ("en-notice", "How many days written notice ends the agreement?", 2),
    ("en-delivery", "Within how many days does the supplier deliver the goods?", 0),
    ("en-price", "How much does the buyer pay per unit?", 3),
]

DOCUMENTS = {
    ZH_DOC: (ZH_PAGES, ZH_QUESTIONS),
    MIXED_DOC: (MIXED_PAGES, MIXED_QUESTIONS),
    TIE_DOC: (TIE_PAGES, TIE_QUESTIONS),
    EN_DOC: (EN_PAGES, EN_QUESTIONS),
}
ALL_QUESTIONS = [(qid, doc, text, page) for doc, (_, group) in DOCUMENTS.items() for qid, text, page in group]
LOCAL_HASH = RetrievalSettings(embedding_backend="local-hash-v1", top_k=5)


def make_project(tmp_path: Path, name: str = "project") -> Project:
    docs = {doc: list(pages) for doc, (pages, _) in DOCUMENTS.items()}
    questions = [{"question_id": qid, "doc_id": doc, "question": text} for qid, doc, text, _ in ALL_QUESTIONS]
    return Project(tmp_path / name, docs, questions).write()


def run(project: Project, name: str = "run-ml", **kwargs: Any) -> tuple[pr.RunnerResult, list[dict[str, Any]]]:
    run_dir = project.root / "results" / "engineering" / name
    result = pr.generate_run(project_root=project.root, run_dir=run_dir, pilot_id=PILOT_ID, **kwargs)
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return result, [json.loads(line) for line in lines]


def retriever_for(pages: list[str], policy: TextPolicy, settings: RetrievalSettings = LOCAL_HASH) -> HybridRetriever:
    chunks: list[Chunk] = []
    for page_id, text in enumerate(pages):
        chunks.extend(
            build_page_chunks(
                example_id="doc", doc_name="doc", page_id=page_id, page_text=text, settings=settings, chunk_policy=policy.chunk_policy
            )
        )
    return HybridRetriever(chunks, settings, tokenizer=policy.tokenizer)


# ---------------------------------------------------------------------------
# Reproduction: these failed on the ASCII-only tokenizer
# ---------------------------------------------------------------------------


def test_repro_chinese_and_mixed_questions_have_query_tokens(tmp_path: Path) -> None:
    """K1: on the old tokenizer every question in ZH_QUESTIONS had zero tokens."""
    result, records = run(make_project(tmp_path))
    by_id = {r["question_id"]: r for r in records}
    for qid, *_ in ALL_QUESTIONS:
        assert by_id[qid]["query_retrieval_tokens"] > 0, qid
    assert result.summary["questions_without_query_tokens"]["total"] == 0


@pytest.mark.parametrize(("qid", "doc", "text", "page"), ALL_QUESTIONS, ids=[q[0] for q in ALL_QUESTIONS])
def test_repro_the_relevant_page_ranks_first(tmp_path: Path, qid: str, doc: str, text: str, page: int) -> None:
    """R1: on the old tokenizer the Chinese questions ranked pages by position, so only zh-pig (page 0) passed."""
    del doc, text
    _, records = run(make_project(tmp_path))
    record = next(r for r in records if r["question_id"] == qid)
    assert record["status"] == "answered"
    assert record["evidence"][0]["page_idx"] == page


def test_repro_an_unspaced_chinese_page_is_split_into_bounded_chunks(tmp_path: Path) -> None:
    """C1: on the old chunker this page was one chunk of about 2,500 characters and the answer was the whole page."""
    page = "".join(f"第{i}号规定：养殖户必须在每个季度末提交第{i}份防疫记录并由县畜牧局备案。" for i in range(1, 60))
    assert len(page) > 2000 and " " not in page
    project = Project(
        tmp_path / "long", {"news/long": [page]}, [{"question_id": "q1", "doc_id": "news/long", "question": "第42号规定要求什么？"}]
    ).write()
    _, (record,) = run(project)
    assert record["ocr_condition"]["chunks"] > 1
    assert len(record["answer"]) <= 400


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def multi(text: str) -> list[str]:
    return tokenize(text, MULTILINGUAL_TOKENIZER)


def test_han_runs_become_overlapping_bigrams_and_a_lone_character_stays() -> None:
    """K1."""
    assert multi("沙门氏菌病") == ["沙门", "门氏", "氏菌", "菌病"]
    assert multi("猪") == ["猪"]
    assert multi("年，月") == ["年", "月"], "punctuation ends a run"


def test_latin_words_numbers_and_symbols_survive_next_to_han_text() -> None:
    """K2."""
    assert multi("2019年GDP增长5%，售价$30") == ["2019", "年", "gdp", "增长", "5%", "售价", "$30"]
    assert multi("Widget X出口") == ["widget", "x", "出口"]
    assert multi("a_b c-d") == ["a", "b", "c", "d"], "underscore and hyphen separate tokens as before"


def test_full_width_forms_fold_to_ascii() -> None:
    """K2: NFKC turns full-width digits, percent and letters into the ASCII tokens."""
    assert multi("１２％ ＧＤＰ") == ["12%", "gdp"]
    assert multi("１２％ ＧＤＰ") == multi("12% GDP")


def test_whitespace_between_han_characters_is_ignored() -> None:
    """K4: OCR line breaks and letter-spaced text put spaces inside Chinese words."""
    assert multi("考 察笔记") == multi("考察笔记")
    assert multi("温\n\n度") == multi("温度")
    assert multi("控 制 也 有") == multi("控制也有")
    assert multi("洁如 hello 建军") == ["洁如", "hello", "建军"], "a space next to a Latin word still separates"


def test_kana_is_tokenized_like_han_and_hangul_words_stay_whole() -> None:
    assert multi("テスト") == ["テス", "スト"]
    assert multi("서울 시장") == ["서울", "시장"]


def test_accented_and_cased_letters_are_kept_and_folded() -> None:
    """K3: the old tokenizer cut ``café`` to ``caf``; the new one keeps the word."""
    assert tokenize("Café Straße", LEGACY_TOKENIZER) == ["caf", "stra", "e"]
    assert multi("Café Straße") == ["café", "strasse"]


@pytest.mark.parametrize("text", ["", "   ", "\n\t", "#", "# \n\n#", "- | * |\n\n---", "？！。，、", "___", "…"])
def test_symbol_only_and_empty_text_give_no_token(text: str) -> None:
    """K5."""
    assert multi(text) == []
    assert tokenize(text, LEGACY_TOKENIZER) == []


def test_ascii_text_tokenizes_exactly_as_before() -> None:
    """K3: the two tokenizers agree on every ASCII string, so English results cannot move."""
    rng = random.Random(20260929)
    alphabet = string.ascii_letters + string.digits + "%$_-.,;:'\"()[]{}<>/\\|!?@#^&*+=~` \t\n"
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        assert multi(text) == tokenize(text, LEGACY_TOKENIZER), repr(text)
    english = " ".join(EN_PAGES + [text for _, text, _ in EN_QUESTIONS])
    assert multi(english) == tokenize(english, LEGACY_TOKENIZER)


def test_every_letter_and_digit_but_a_few_arabic_diacritics_gives_a_token() -> None:
    """N1: no_retrieval_tokens is left for a stand-alone Arabic diacritic letter and nothing else."""
    exceptions = set(range(0xFC5E, 0xFC64)) | set(range(0xFE70, 0xFE7F))
    missing = [
        cp
        for cp in range(sys.maxunicode + 1)
        if not 0xD800 <= cp <= 0xDFFF
        and unicodedata.category(chr(cp))[0] in "LN"
        and not multi(chr(cp))
        and cp not in exceptions
    ]
    assert missing == []


def test_unknown_tokenizer_and_policy_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer"):
        tokenize("x", "no-such-tokenizer")
    with pytest.raises(ValueError, match="unknown tokenizer"):
        LocalHashEmbedder("no-such-tokenizer")
    with pytest.raises(ValueError, match="unknown tokenizer"):
        HybridRetriever([Chunk("c", "d", "d", 0, "text")], LOCAL_HASH, tokenizer="no-such-tokenizer")
    with pytest.raises(ValueError, match="unknown chunk policy"):
        TextPolicy(LEGACY_TOKENIZER, "no-such-policy")
    with pytest.raises(ValueError, match="unknown chunk policy"):
        build_page_chunks(example_id="e", doc_name="d", page_id=0, page_text="x", settings=LOCAL_HASH, chunk_policy="no-such-policy")


# ---------------------------------------------------------------------------
# Local hashed embeddings
# ---------------------------------------------------------------------------


def encode(text: str, tokenizer: str) -> np.ndarray:
    return LocalHashEmbedder(tokenizer).encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]


def test_the_hashing_embedder_gives_chinese_text_a_signal_only_under_the_new_tokenizer() -> None:
    """K1, K4."""
    query, related, unrelated = "沙门氏菌病如何传播", "仔猪沙门氏菌病主要通过饲料传播", "长江全长约六千三百公里"
    assert not encode(query, LEGACY_TOKENIZER).any()
    new = {text: encode(text, MULTILINGUAL_TOKENIZER) for text in (query, related, unrelated)}
    assert all(vector.any() for vector in new.values())
    assert float(new[query] @ new[related]) > float(new[query] @ new[unrelated])


def test_the_default_embedder_is_unchanged() -> None:
    """C4: ``LocalHashEmbedder()`` hashes the original ASCII tokens into the same buckets as before."""
    from hashlib import blake2b

    expected = np.zeros(LocalHashEmbedder.dimensions, dtype=np.float32)
    for token in re.findall(r"[a-z0-9%$]+", "warranty period 12% 保修期".lower()):
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        expected[int.from_bytes(digest[:4], "big") % LocalHashEmbedder.dimensions] += 1.0 if digest[4] & 1 else -1.0
    expected /= np.linalg.norm(expected)
    got = LocalHashEmbedder().encode(["warranty period 12% 保修期"], normalize_embeddings=True, convert_to_numpy=True)[0]
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------


def settings_for(size: int, overlap: int) -> RetrievalSettings:
    return RetrievalSettings(chunk_size_words=size, chunk_overlap_words=overlap, embedding_backend="local-hash-v1")


def page_chunks(text: str, settings: RetrievalSettings, policy: str) -> list[Chunk]:
    return build_page_chunks(example_id="e", doc_name="d", page_id=3, page_text=text, settings=settings, chunk_policy=policy)


def test_an_unspaced_han_page_splits_into_chunks_of_at_most_two_characters_per_word() -> None:
    """C1: with the engineering settings (180 words, overlap 40) a chunk holds at most 360 characters and overlaps by 80."""
    settings = settings_for(180, 40)
    text = "".join(chr(0x4E00 + (i * 7919) % 20000) for i in range(3001))
    chunks = page_chunks(text, settings, MULTILINGUAL_CHUNK_POLICY)
    limit = settings.chunk_size_words * CJK_CHARS_PER_WORD
    assert len(chunks) > 8
    assert all(0 < len(chunk.text) <= limit for chunk in chunks)
    assert [len(chunk.text) for chunk in chunks[:-1]] == [limit] * (len(chunks) - 1)
    step = (settings.chunk_size_words - settings.chunk_overlap_words) * CJK_CHARS_PER_WORD
    assert [chunk.text for chunk in chunks] == [text[i : i + limit] for i in range(0, len(text), step)][: len(chunks)]
    assert chunks[-1].text.endswith(text[-10:]), "the last chunk reaches the end of the page"
    assert [chunk.chunk_id for chunk in chunks] == [f"e-p3-c{i}" for i in range(len(chunks))]
    # The original policy left the same page as one chunk.
    assert len(page_chunks(text, settings, LEGACY_CHUNK_POLICY)) == 1


def test_letter_spaced_han_text_stays_within_twice_the_bound() -> None:
    """C1: whitespace between characters is kept as one space, so the worst case is 2 * 360 - 1 characters."""
    settings = settings_for(180, 40)
    text = " ".join("控制也有不可估量" * 200)
    chunks = page_chunks(text, settings, MULTILINGUAL_CHUNK_POLICY)
    assert len(chunks) > 1
    assert max(len(chunk.text) for chunk in chunks) <= 2 * 360 - 1


def test_mixed_text_counts_a_latin_word_as_two_han_characters() -> None:
    settings = settings_for(3, 1)
    text = "aa bb 一二三四 cc"  # units: aa bb 一 二 三 四 cc, weights 2 2 1 1 1 1 2, window 6, step 4
    texts = [chunk.text for chunk in page_chunks(text, settings, MULTILINGUAL_CHUNK_POLICY)]
    assert texts == ["aa bb 一二", "一二三四 cc"]


def test_text_without_han_chunks_exactly_as_before() -> None:
    """C3: same ids, same text, over random whitespace, sizes and overlaps."""
    rng = random.Random(7)
    spaces = [" ", "  ", "\n", "\t", " ", "\r\n", " "]
    for _ in range(300):
        words = ["".join(rng.choice(string.ascii_letters + "%$.,éü") for _ in range(rng.randint(1, 9))) for _ in range(rng.randint(0, 60))]
        text = "".join(rng.choice(spaces) + word for word in words) + rng.choice(["", " ", "\n"])
        size = rng.randint(1, 12)
        settings = settings_for(size, rng.randint(0, size - 1))
        old = page_chunks(text, settings, LEGACY_CHUNK_POLICY)
        new = page_chunks(text, settings, MULTILINGUAL_CHUNK_POLICY)
        assert [(c.chunk_id, c.text) for c in new] == [(c.chunk_id, c.text) for c in old], repr(text)


def test_windows_cover_every_unit_and_always_end() -> None:
    """C2: any text, size and overlap, including size 1 and an overlap at or above the size.

    ``RetrievalSettings`` refuses an overlap at or above the size, but ``chunk_spans`` takes plain integers.
    """
    rng = random.Random(11)
    pieces = ["中", "文", "字", "テ", "ab", "cd", "12%", " ", "\n", "。", "　"]
    for _ in range(400):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 90)))
        size, overlap = rng.randint(1, 8), rng.randint(0, 12)
        spans = chunk_spans(text, size, overlap)
        units = [m for m in re.finditer(r"[一-鿿ぁ-ヿ]|[^\s一-鿿ぁ-ヿ]+", text)]
        assert bool(spans) == bool(units)
        for unit in units:
            assert any(s <= unit.start() and unit.end() <= e for s, e in spans), (text, size, overlap, unit)
        if spans:
            assert spans[0][0] == units[0].start() and spans[-1][1] == units[-1].end()
            assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(spans, spans[1:])), "windows move forward"
        for start, end in spans:
            weight = sum(1 if re.match(r"[一-鿿ぁ-ヿ]", m.group()) else 2 for m in re.finditer(
                r"[一-鿿ぁ-ヿ]|[^\s一-鿿ぁ-ヿ]+", text[start:end]))
            assert weight <= size * CJK_CHARS_PER_WORD


def test_the_shared_defaults_keep_the_original_behavior() -> None:
    """C4: ``graph`` and ``benchmarks`` call these entry points with no policy argument."""
    import inspect

    from faar import benchmarks, graph

    assert inspect.signature(build_page_chunks).parameters["chunk_policy"].default == LEGACY_CHUNK_POLICY
    assert inspect.signature(HybridRetriever.__init__).parameters["tokenizer"].default == LEGACY_TOKENIZER
    assert inspect.signature(LocalHashEmbedder.__init__).parameters["tokenizer"].default == LEGACY_TOKENIZER
    for module in (graph, benchmarks):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "chunk_policy" not in source and "tokenizer" not in source, module.__name__
    # RetrievalSettings feeds faar.run_io.run_fingerprint through model_dump(). Any new field would change
    # the fingerprint of every existing profile, so the policy ids live outside it.
    assert not {"tokenizer", "chunk_policy", "text_policy"} & set(RetrievalSettings.model_fields)


def test_the_default_chunker_and_retriever_still_give_han_text_no_signal() -> None:
    """C4: unchanged shared behavior, stated so nobody mistakes it for a regression."""
    settings = settings_for(180, 40)
    (chunk,) = page_chunks(ZH_PAGES[0], settings, LEGACY_CHUNK_POLICY)
    english = page_chunks(EN_PAGES[1], settings, LEGACY_CHUNK_POLICY)
    retriever = HybridRetriever([chunk, *english], settings)
    assert retriever.tokenizer == LEGACY_TOKENIZER
    assert tokenize("沙门氏菌病", retriever.tokenizer) == []
    example_chunks = build_chunks(
        type("E", (), {"example_id": "e", "doc_name": "d", "ocr_text": ZH_PAGES[0], "page_ids": [0], "metadata": {}})(), settings
    )
    assert [c.text for c in example_chunks] == [chunk.text]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("qid", "doc", "text", "page"), ALL_QUESTIONS, ids=[q[0] for q in ALL_QUESTIONS])
def test_the_relevant_page_ranks_first_with_the_retriever_alone(qid: str, doc: str, text: str, page: int) -> None:
    """R1: retriever level, so the answer step plays no part."""
    del qid
    hits = retriever_for(DOCUMENTS[doc][0], MULTILINGUAL_TEXT_POLICY).retrieve(text)
    assert hits[0].chunk.page_id == page
    assert hits[0].bm25_score == 1.0 or hits[0].dense_score == 1.0, "the winner leads at least one signal"


def test_the_english_results_equal_the_old_tokenizer_results() -> None:
    """K3, C3: same hits, same order and same scores on the English fixture."""
    old = retriever_for(EN_PAGES, LEGACY_TEXT_POLICY)
    new = retriever_for(EN_PAGES, MULTILINGUAL_TEXT_POLICY)
    for _, text, _ in EN_QUESTIONS:
        old_hits, new_hits = old.retrieve(text), new.retrieve(text)
        assert [h.chunk.chunk_id for h in old_hits] == [h.chunk.chunk_id for h in new_hits]
        assert [(h.bm25_score, h.dense_score, h.fused_score) for h in old_hits] == [
            (h.bm25_score, h.dense_score, h.fused_score) for h in new_hits
        ]


def test_the_old_tokenizer_cannot_rank_the_chinese_fixture() -> None:
    """The contrast that gives the passing tests meaning: with the old policy the order is page position."""
    retriever = retriever_for(ZH_PAGES, LEGACY_TEXT_POLICY)
    for _, text, _ in ZH_QUESTIONS:
        assert [h.chunk.page_id for h in retriever.retrieve(text)] == [0, 1, 2, 3, 4]


def test_the_relevant_chunk_wins_inside_a_long_chinese_page() -> None:
    """R1, C1: an unspaced page of 16 sentences makes several chunks, and the one holding the sentence ranks first."""
    sentences = [
        "青岛港的集装箱吞吐量在去年突破了两千万标准箱。",
        "兰州拉面的制作需要经过和面、揉面、拉面和煮面四个步骤。",
        "敦煌莫高窟现存洞窟七百三十五个，壁画面积四万五千平方米。",
        "深圳湾大桥全长五点五公里，是连接深圳与香港的跨海通道。",
        "西湖龙井的采摘时间集中在清明节前后的二十天之内。",
        "内蒙古草原的羊群数量在春季接产期达到全年最高值。",
        "哈尔滨的冰雪大世界每年十二月中旬开园，持续到次年二月。",
        "景德镇的青花瓷烧制温度约为一千三百摄氏度。",
        "三峡大坝的总装机容量为二千二百五十万千瓦。",
        "拉萨布达拉宫始建于七世纪，海拔三千七百米。",
        "苏州园林中的拙政园占地约五点二公顷。",
        "海南的椰子产量占全国总产量的九成以上。",
        "桂林漓江的游览线路全长八十三公里。",
        "云南普洱茶的发酵工艺分为生茶和熟茶两类。",
        "西安兵马俑一号坑东西长二百三十米，南北宽六十二米。",
        "武汉长江大桥于一九五七年建成通车。",
    ]
    page = "".join(sentences)
    assert " " not in page
    retriever = retriever_for([page], MULTILINGUAL_TEXT_POLICY, settings_for(30, 6))
    assert len(retriever.chunks) >= 3
    for target in (sentences[1], sentences[8], sentences[14], sentences[15]):
        question = {
            sentences[1]: "兰州拉面的制作有哪些步骤？",
            sentences[8]: "三峡大坝的总装机容量是多少？",
            sentences[14]: "兵马俑一号坑东西长多少米？",
            sentences[15]: "武汉长江大桥是哪一年建成通车的？",
        }[target]
        top = retriever.retrieve(question)[0]
        assert target in top.chunk.text, question


def test_numbers_and_percent_signs_decide_between_similar_pages() -> None:
    """K2, R1: the pages differ only in year and percentage."""
    pages = ["2022年营收同比增长3%。", "2023年营收同比增长12%。", "2024年营收同比增长8%。"]
    retriever = retriever_for(pages, MULTILINGUAL_TEXT_POLICY)
    assert retriever.retrieve("增长12%")[0].chunk.page_id == 1
    assert retriever.retrieve("2024年增长")[0].chunk.page_id == 2
    assert retriever.retrieve("3% 2022")[0].chunk.page_id == 0


def test_repeated_query_terms_do_not_change_the_winner() -> None:
    """R2."""
    retriever = retriever_for(ZH_PAGES, MULTILINGUAL_TEXT_POLICY)
    assert retriever.retrieve("棉区 棉区 棉区 棉区 地区")[0].chunk.page_id == 1
    english = retriever_for(EN_PAGES, MULTILINGUAL_TEXT_POLICY)
    assert english.retrieve("warranty warranty warranty period")[0].chunk.page_id == 1


def test_a_symbol_only_page_is_never_the_top_hit() -> None:
    """R2, K5: a chunk with no token stays in the index and ranks last."""
    retriever = retriever_for(["# \n\n#", ZH_PAGES[1], "---"], MULTILINGUAL_TEXT_POLICY)
    hits = retriever.retrieve("三大产棉区包括哪些地区？")
    assert hits[0].chunk.page_id == 1
    assert {h.chunk.page_id for h in hits} == {0, 1, 2}


def test_an_empty_or_symbol_only_query_ranks_by_position_and_does_not_raise() -> None:
    """K5: the record's query_retrieval_tokens is 0, which the summary counts."""
    retriever = retriever_for(ZH_PAGES, MULTILINGUAL_TEXT_POLICY)
    for query in ("", "？！", "..."):
        assert [h.chunk.page_id for h in retriever.retrieve(query)] == [0, 1, 2, 3, 4]


def test_hits_never_cross_documents_in_chinese(tmp_path: Path) -> None:
    """R3: two Chinese documents share most words; each question sees only its own document."""
    twin = [page.replace("沙门氏菌病", "口蹄疫") for page in ZH_PAGES]
    project = Project(
        tmp_path / "twins",
        {"news/zh_a": list(ZH_PAGES), "news/zh_b": twin},
        [
            {"question_id": "qa", "doc_id": "news/zh_a", "question": "沙门氏菌病是如何传播的？"},
            {"question_id": "qb", "doc_id": "news/zh_b", "question": "沙门氏菌病是如何传播的？"},
            {"question_id": "qc", "doc_id": "news/zh_b", "question": "口蹄疫是如何传播的？"},
        ],
    ).write()
    _, records = run(project)
    for record in records:
        assert {e["doc_id"] for e in record["evidence"]} == {record["doc_id"]}
    by_id = {r["question_id"]: r for r in records}
    assert by_id["qa"]["evidence"][0]["page_idx"] == 0 and by_id["qc"]["evidence"][0]["page_idx"] == 0
    assert "口蹄疫" in by_id["qc"]["answer"] and "口蹄疫" not in by_id["qa"]["answer"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

_RANKING_SCRIPT = """
import json, sys
sys.path.insert(0, {src!r})
from faar.chunking import build_page_chunks
from faar.retrieval import HybridRetriever
from faar.settings import RetrievalSettings
from faar.text_units import MULTILINGUAL_TEXT_POLICY as P
pages, queries = json.loads(sys.stdin.read())
settings = RetrievalSettings(embedding_backend="local-hash-v1", top_k=5)
chunks = [c for i, t in enumerate(pages) for c in build_page_chunks(example_id="d", doc_name="d", page_id=i, page_text=t, settings=settings, chunk_policy=P.chunk_policy)]
retriever = HybridRetriever(chunks, settings, tokenizer=P.tokenizer)
print(json.dumps([[(h.chunk.chunk_id, h.bm25_score, h.dense_score, h.fused_score) for h in retriever.retrieve(q)] for q in queries]))
"""


def test_ranking_is_identical_across_processes_and_hash_seeds() -> None:
    """R4."""
    pages = ZH_PAGES + MIXED_PAGES
    queries = [text for _, text, _ in ZH_QUESTIONS + MIXED_QUESTIONS] + ["", "？！"]
    inproc = retriever_for(pages, MULTILINGUAL_TEXT_POLICY)
    expected = json.loads(json.dumps([[(h.chunk.chunk_id.replace("doc", "d"), h.bm25_score, h.dense_score, h.fused_score) for h in inproc.retrieve(q)] for q in queries]))
    outputs = []
    for seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", _RANKING_SCRIPT.format(src=str(SRC))],
            input=json.dumps([pages, queries]),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs[0] == outputs[1] == outputs[2] == expected


def test_two_fresh_runs_write_identical_predictions(tmp_path: Path) -> None:
    """R4."""
    project = make_project(tmp_path)
    first, _ = run(project, "run-a")
    second, _ = run(project, "run-b")
    read = lambda name: (project.root / "results" / "engineering" / name / "predictions.jsonl").read_bytes()  # noqa: E731
    assert read("run-a") == read("run-b")
    assert first.summary["predictions_sha256"] == second.summary["predictions_sha256"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def test_generation_needs_no_evaluation_input(tmp_path: Path) -> None:
    """N3: predictions are the same with the evaluation manifest present and with it deleted."""
    with_manifest = make_project(tmp_path, "with")
    without = make_project(tmp_path, "without")
    without.evaluation_manifest.unlink()
    (without.pilot_dir / "selection_record.json").unlink()
    _, records_with = run(with_manifest)
    _, records_without = run(without)
    assert records_with == records_without


def test_chinese_text_is_content_and_tokens_not_a_tokenizer_limit(tmp_path: Path) -> None:
    """N1: a Han-only document is searched. Symbol-only OCR keeps its own reason. The reasons stay apart."""
    project = Project(
        tmp_path / "reasons",
        {"zh/only": [ZH_PAGES[3]], "zh/symbols": ["# \n\n#"], "zh/blank": ["", None]},
        [
            {"question_id": "q-han", "doc_id": "zh/only", "question": "在审题过程中应该遵循哪些步骤？"},
            {"question_id": "q-sym", "doc_id": "zh/symbols", "question": "在审题过程中应该遵循哪些步骤？"},
            {"question_id": "q-blank", "doc_id": "zh/blank", "question": "在审题过程中应该遵循哪些步骤？"},
        ],
    ).write()
    result, records = run(project)
    by_id = {r["question_id"]: r for r in records}
    assert (by_id["q-han"]["status"], by_id["q-han"]["no_evidence_reason"]) == ("answered", None)
    assert by_id["q-sym"]["no_evidence_reason"] == "no_text_content"
    assert by_id["q-blank"]["no_evidence_reason"] == "no_text_chunks"
    assert result.summary["no_evidence_by_reason"]["no_retrieval_tokens"] == 0


def test_run_config_and_fingerprint_record_the_policy_ids(tmp_path: Path) -> None:
    """N2."""
    project = make_project(tmp_path)
    result, _ = run(project, "run-new")
    legacy, _ = run(project, "run-old", text_policy=LEGACY_TEXT_POLICY)
    config = json.loads((project.root / "results/engineering/run-new/run_config.json").read_text(encoding="utf-8"))
    retrieval = config["retrieval"]
    assert retrieval["tokenisation"]["tokenizer"] == "multilingual-v1"
    assert retrieval["chunking"]["policy"] == "cjk-weighted-words-v1"
    assert retrieval["chunking"]["cjk_chars_per_word"] == 2
    assert "gets no lexical" not in json.dumps(retrieval), "the old claim that Han text has no signal must be gone"
    assert result.summary["fingerprint"] != legacy.summary["fingerprint"]
    old_config = json.loads((project.root / "results/engineering/run-old/run_config.json").read_text(encoding="utf-8"))
    assert old_config["retrieval"]["tokenisation"]["tokenizer"] == "ascii-alnum-v1"
    assert old_config["retrieval"]["chunking"]["policy"] == "whitespace-words-v1"
    assert old_config["retrieval"]["tokenisation"]["retrieval_tokeniser"] == "[a-z0-9%$]+"


def test_the_policy_is_part_of_the_generation_fingerprint(tmp_path: Path) -> None:
    """N2: rerunning into the same directory under another policy is refused."""
    project = make_project(tmp_path)
    run(project, "run-x")
    with pytest.raises(pr.RunnerRefusal, match="fingerprint"):
        run(project, "run-x", text_policy=LEGACY_TEXT_POLICY)
    same, _ = run(project, "run-x")
    assert not same.wrote and "verified identical" in same.message


def test_the_english_only_project_gives_the_same_records_under_both_policies(tmp_path: Path) -> None:
    """K3, C3: English documents answer the same under the old and the new policy, apart from the config."""
    docs = {DOC_A: list(EN_PAGES)}
    questions = [{"question_id": qid, "doc_id": DOC_A, "question": text} for qid, text, _ in EN_QUESTIONS]
    project = Project(tmp_path / "en", {DOC_A: docs[DOC_A]}, questions).write()
    _, new = run(project, "run-new")
    _, old = run(project, "run-old", text_policy=LEGACY_TEXT_POLICY)
    assert new == old
