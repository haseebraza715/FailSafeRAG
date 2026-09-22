from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=4)
def _load_symspell(source_str: str):
    from symspellpy import SymSpell

    symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    if not symspell.load_dictionary(source_str, term_index=0, count_index=1):
        raise RuntimeError(f"Could not load SymSpell frequency dictionary: {source_str}")
    return symspell


def correct_text(text: str, dictionary_path: Path | None = None) -> str:
    """Correct alphabetic OCR tokens with the local, caller-supplied SymSpell dictionary."""
    source = dictionary_path or _dictionary_path_from_env()
    if source is None:
        raise RuntimeError("SYMSPELL_DICTIONARY must point to a local frequency dictionary for the symspell ablation.")
    try:
        from symspellpy import Verbosity
    except ImportError as exc:  # pragma: no cover - dependency/runtime specific
        raise RuntimeError("symspellpy is required for --wordlevel_fallback symspell.") from exc

    symspell = _load_symspell(str(source))

    def correct_token(match: re.Match[str]) -> str:
        token = match.group(0)
        suggestions = symspell.lookup(token.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if not suggestions:
            return token
        candidate = suggestions[0].term
        return candidate.capitalize() if token[0].isupper() else candidate

    return re.sub(r"[A-Za-z]{3,}", correct_token, text)


def _dictionary_path_from_env() -> Path | None:
    value = os.getenv("SYMSPELL_DICTIONARY")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(f"SYMSPELL_DICTIONARY does not exist: {path}")
    return path
