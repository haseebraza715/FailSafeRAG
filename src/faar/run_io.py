from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

FINGERPRINT_SCHEMA_VERSION = 3

_CODE_IDENTITY_CACHE: dict[str, str] = {}


def _measurement_code_digest() -> str:
    package_dir = Path(__file__).resolve().parent
    cached = _CODE_IDENTITY_CACHE.get(str(package_dir))
    if cached is not None:
        return cached
    hasher = hashlib.sha256()
    for source in sorted(package_dir.glob("*.py")):
        hasher.update(source.name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(source.read_bytes())
    digest = hasher.hexdigest()
    _CODE_IDENTITY_CACHE[str(package_dir)] = digest
    return digest

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2))


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def safe_checkpoint_stem(example_id: str) -> str:
    if not isinstance(example_id, str) or not example_id:
        raise ValueError(f"example_id must be a non-empty string; received {example_id!r}.")
    if "\x00" in example_id:
        raise ValueError("example_id must not contain a null byte.")
    encoded = quote(example_id, safe="._-")
    if encoded in {".", ".."}:
        encoded = f"id-{hashlib.sha256(example_id.encode('utf-8')).hexdigest()}"
    if len(encoded.encode("utf-8")) > 200:
        encoded = f"id-{hashlib.sha256(example_id.encode('utf-8')).hexdigest()}"
    return encoded


def canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return str(value)


def run_fingerprint(
    settings: Any,
    *,
    profile: str,
    dataset: str | None,
    split: str | None,
    manifest_sha256: str | None = None,
) -> str:
    return canonical_digest(
        {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "code_identity": _measurement_code_digest(),
            "profile": profile,
            "dataset": dataset or "phase0",
            "split": split or "development",
            "manifest_sha256": manifest_sha256,
            "retrieval": settings.retrieval.model_dump(),
            "gate": settings.gate.model_dump(),
            "recovery": settings.recovery.model_dump(),
            "experiment": settings.experiment.model_dump(),
        }
    )


def select_shard(example_ids: list[str], shard_index: int, num_shards: int) -> list[str]:
    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise ValueError(f"num_shards must be an integer; received {num_shards!r}.")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1; received {num_shards}.")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValueError(f"shard_index must be an integer; received {shard_index!r}.")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} out of range for num_shards {num_shards}.")
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("example_ids must not contain duplicates.")
    ordered = sorted(example_ids)
    chunk_size = math.ceil(len(ordered) / num_shards)
    start = shard_index * chunk_size
    return ordered[start : start + chunk_size]


def shard_label(shard_index: int, num_shards: int) -> str:
    return f"shard{shard_index + 1}of{num_shards}"
