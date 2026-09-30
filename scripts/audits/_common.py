"""Shared helpers for the audit scripts in this directory.

Every audit script resolves its default paths against the repository root, fails with a
one-line message and a nonzero exit code when a required local asset is missing, and writes
only into the directory named by ``--out``. None of them calls a model, opens a network
connection or reads a credential.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXIT_FAILED_CHECK = 1
EXIT_MISSING_INPUT = 2

# Han (unified ideographs, extension A, compatibility ideographs). Kana are not included.
HAN = re.compile("[㐀-䶿一-鿿豈-﫿\U00020000-\U0002fa1f]")


def fail(message: str, code: int = EXIT_MISSING_INPUT) -> NoReturn:
    """Print a one-line error to stderr and exit."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def resolve(path: str | Path, base: Path = REPO_ROOT) -> Path:
    """Return ``path`` as absolute. A relative path is taken from ``base``, the repository root by default."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else base / candidate


def require_file(path: Path, what: str, hint: str | None = None) -> Path:
    if not path.is_file():
        fail(f"required {what} not found: {path}" + (f" ({hint})" if hint else ""))
    return path


def require_dir(path: Path, what: str, hint: str | None = None) -> Path:
    if not path.is_dir():
        fail(f"required {what} not found: {path}" + (f" ({hint})" if hint else ""))
    return path


def read_json(path: Path, what: str) -> Any:
    require_file(path, what)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {what} {path}: {exc}")


def read_jsonl(path: Path, what: str) -> list[dict[str, Any]]:
    require_file(path, what)
    rows: list[dict[str, Any]] = []
    number = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    rows.append(json.loads(line))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {what} {path} (line {number}): {exc}")
    return rows


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def ids_digest(ids: Sequence[str]) -> str:
    """sha256 of the ids in the given order, joined by a newline."""
    return sha256_hex("\n".join(ids).encode("utf-8"))


def norm_ws(text: str) -> str:
    """Collapse every whitespace run to one space, as the chunker does before it cuts chunks."""
    return " ".join(text.split())


def refuse_protected_out(out: Path) -> Path:
    """Refuse an output directory under ``results/pilots/``, which holds frozen pilot records."""
    parts = tuple(part.casefold() for part in out.resolve().parts)
    if any(parts[i : i + 2] == ("results", "pilots") for i in range(len(parts) - 1)):
        fail(f"--out {out} is under results/pilots/, which holds frozen pilot records; choose another directory")
    return out


def prepare_out(out: Path) -> Path:
    """Check ``--out`` is allowed and create it."""
    refuse_protected_out(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
