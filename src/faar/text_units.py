"""Tokenizers and chunk boundaries for text that may hold Chinese.

The original retrieval code cuts text into ``[a-z0-9%$]+`` tokens and splits
pages on whitespace. Both assume that words are separated by spaces and written
in ASCII. Chinese has neither property: a Chinese question has no token, and an
unspaced Chinese page is one "word" and therefore one unbounded chunk.

This module holds two versioned policies that fix that for the offline
engineering path. The original behaviour stays available under its own id, and
it remains the default of every shared entry point (``HybridRetriever``,
``LocalHashEmbedder``, ``build_page_chunks``), so ``faar.graph`` and
``faar.benchmarks`` are unchanged.

Tokenizers (``tokenize``):

``ascii-alnum-v1``
    ``re.findall("[a-z0-9%$]+", text.lower())``. The original behaviour.

``multilingual-v1``
    1. Normalise with Unicode NFKC, then ``casefold``. Full-width digits, ``％``
       and Latin letters fold to their ASCII forms.
    2. Delete whitespace that sits between two CJK characters. OCR line breaks
       and letter-spaced headings put spaces inside Chinese words (``考 察``).
    3. Cut the text into word tokens and CJK runs. A word token is a run of
       Unicode letters or digits, ``%`` and ``$``, so ``12%`` and ``$5`` stay
       whole. ``_`` and every other symbol separate tokens.
    4. A CJK run (Han, Hiragana, Katakana) becomes overlapping character
       bigrams. A run of one character gives that character.

    For text that is pure ASCII the result equals ``ascii-alnum-v1``. For other
    text it differs on purpose: ``café`` is one token and Chinese text has
    tokens. Bigrams need no dictionary, no model and no state, and they give a
    substring the same tokens in a query and in a document. A dictionary
    segmenter does not: ``jieba`` cuts ``沙门氏菌病是…`` into ``沙门氏菌`` and
    ``病是`` but ``沙门氏菌病急性`` into ``沙门氏菌`` and ``病``, so the same
    phrase gets different tokens depending on the text around it. ``jieba``
    also writes a cache file on first use and prints to stderr. Bigrams cost
    more tokens per passage and count common function words as tokens.

Chunk policies (``faar.chunking.build_page_chunks``):

``whitespace-words-v1``
    Windows of ``chunk_size_words`` whitespace-delimited words. The original
    behaviour.

``cjk-weighted-words-v1``
    Cut the page into units. A CJK character is one unit and any other run of
    non-space characters is one unit. A CJK unit weighs 1 and any other unit
    weighs ``CJK_CHARS_PER_WORD`` (2), so a window of ``chunk_size_words`` words
    holds at most ``2 * chunk_size_words`` CJK characters. Two CJK characters
    stand in for one word because a Chinese word has about that many
    characters. The window advances by ``chunk_size_words - chunk_overlap_words``
    words on the same scale. Text without CJK characters gets the boundaries and
    the text of the original policy.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

LEGACY_TOKENIZER = "ascii-alnum-v1"
MULTILINGUAL_TOKENIZER = "multilingual-v1"
TOKENIZERS = (LEGACY_TOKENIZER, MULTILINGUAL_TOKENIZER)

LEGACY_CHUNK_POLICY = "whitespace-words-v1"
MULTILINGUAL_CHUNK_POLICY = "cjk-weighted-words-v1"
CHUNK_POLICIES = (LEGACY_CHUNK_POLICY, MULTILINGUAL_CHUNK_POLICY)

# Han (Unified, Extension A, Compatibility, Extensions B to H), the iteration marks
# U+3005 to U+3007, Hiragana and Katakana. The punctuation U+30A0 and U+30FB are left out.
CJK_CHARACTERS = "々-〇ぁ-ゟァ-ヺー-ヿ㐀-䶿一-鿿豈-﫿\U00020000-\U000323af"

# Two CJK characters weigh as much as one whitespace-delimited word.
CJK_CHARS_PER_WORD = 2

_LEGACY_TOKEN = re.compile(r"[a-z0-9%$]+")
_TOKEN = re.compile(rf"[{CJK_CHARACTERS}]+|(?:(?![{CJK_CHARACTERS}])[^\W_]|[%$])+")
_CJK_GAP = re.compile(rf"(?<=[{CJK_CHARACTERS}])\s+(?=[{CJK_CHARACTERS}])")
_CJK_CHAR = re.compile(f"[{CJK_CHARACTERS}]")
_UNIT = re.compile(rf"[{CJK_CHARACTERS}]|[^\s{CJK_CHARACTERS}]+")

TOKENIZER_DESCRIPTIONS = {
    LEGACY_TOKENIZER: "runs of [a-z0-9%$] in lower-cased text",
    MULTILINGUAL_TOKENIZER: (
        "NFKC, casefold, delete whitespace between CJK characters, then runs of Unicode letters or digits plus % and $ "
        "as word tokens and overlapping character bigrams for each CJK run"
    ),
}


def contains_cjk(text: str) -> bool:
    return _CJK_CHAR.search(text) is not None


def check_tokenizer(tokenizer: str) -> str:
    """Return ``tokenizer`` if it names a known tokenizer, and raise ``ValueError`` otherwise."""
    if tokenizer not in TOKENIZERS:
        raise ValueError(f"unknown tokenizer {tokenizer!r}; expected one of {', '.join(TOKENIZERS)}")
    return tokenizer


def tokenize(text: str, tokenizer: str = LEGACY_TOKENIZER) -> list[str]:
    """Return the retrieval tokens of ``text`` under the named tokenizer."""
    if check_tokenizer(tokenizer) == LEGACY_TOKENIZER:
        return _LEGACY_TOKEN.findall(text.lower())
    return _tokenize_multilingual(text)


def _tokenize_multilingual(text: str) -> list[str]:
    folded = _CJK_GAP.sub("", unicodedata.normalize("NFKC", text).casefold())
    tokens: list[str] = []
    for match in _TOKEN.finditer(folded):
        piece = match.group()
        if _CJK_CHAR.match(piece):
            tokens.extend([piece] if len(piece) == 1 else [piece[i : i + 2] for i in range(len(piece) - 1)])
        else:
            tokens.append(piece)
    return tokens


def chunk_spans(text: str, chunk_size_words: int, chunk_overlap_words: int) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` character spans of the ``cjk-weighted-words-v1`` chunks of ``text``.

    Whitespace inside a span is not yet normalised. ``faar.chunking`` collapses
    each whitespace run to one space, as the original policy did.
    """
    units = [(m.start(), m.end(), 1 if _CJK_CHAR.match(m.group()) else CJK_CHARS_PER_WORD) for m in _UNIT.finditer(text)]
    size = chunk_size_words * CJK_CHARS_PER_WORD
    step = max(chunk_size_words - chunk_overlap_words, 1) * CJK_CHARS_PER_WORD
    prefix = [0]
    for _, _, weight in units:
        prefix.append(prefix[-1] + weight)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(units):
        # Last unit boundary that keeps the window within `size`. `size` is at least the weight
        # of one unit, so the window always holds a unit and the loop always advances.
        end = max(bisect_right(prefix, prefix[start] + size, lo=start + 1) - 1, start + 1)
        spans.append((units[start][0], units[end - 1][1]))
        if end >= len(units):
            break
        # The next window starts one step on. It never starts past `end`, so no unit is skipped.
        start = min(max(bisect_left(prefix, prefix[start] + step, lo=start + 1), start + 1), end)
    return spans


@dataclass(frozen=True)
class TextPolicy:
    """The tokenizer and chunk policy that one retrieval setup uses together."""

    tokenizer: str
    chunk_policy: str

    def __post_init__(self) -> None:
        check_tokenizer(self.tokenizer)
        if self.chunk_policy not in CHUNK_POLICIES:
            raise ValueError(f"unknown chunk policy {self.chunk_policy!r}; expected one of {', '.join(CHUNK_POLICIES)}")


LEGACY_TEXT_POLICY = TextPolicy(LEGACY_TOKENIZER, LEGACY_CHUNK_POLICY)
MULTILINGUAL_TEXT_POLICY = TextPolicy(MULTILINGUAL_TOKENIZER, MULTILINGUAL_CHUNK_POLICY)
