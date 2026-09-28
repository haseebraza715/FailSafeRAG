"""Read-only FAAR asset audit: inventory, provenance, OHR coverage and split integrity.

The audit never modifies source assets. It reads project-local asset roots and
explicitly configured locations (CLI flags, exported FAAR_* / HF_HOME values, or
the path keys of the project `.env`), then writes:

  <out-dir>/asset_manifest.json   machine-readable manifest (deterministic order)
  <out-dir>/asset_audit_summary.md generated summary of the manifest figures

It imports only the standard library plus the stdlib-only helper
`faar.ohr_inventory` (existing OHR alias rules). No models, services or GPU work
are touched. `pypdfium2` is used for PDF page counts only when importable.

Exit codes describe audit execution, not experimental readiness:
  0  audit completed; no integrity failure. Missing, unreadable or ambiguous
     assets are findings, and the manifest's `experimental_readiness` block
     says whether any of them blocks preparation.
  1  audit could not run (invalid project root, unwritable output directory)
  2  audit completed and found an integrity failure: a lock hash mismatch
     (QA, split, ArXivQA source or OHR PDF archive), unreadable or malformed
     QA/split source, duplicate question IDs, or question-ID overlap between splits

Usage:
  python scripts/data/audit_assets.py --project-root . --out-dir results/data_audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import zipfile
import zlib
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
VOLATILE_FIELDS = ["generated_at_utc", "repository.dirty", "repository.dirty_paths", "runtime_sec"]

QA_REL = "OHR-Bench/data/qas_v2.json"
QA_V1_REL = "OHR-Bench/data/qas.json"
SUBQA_REL = "OHR-Bench/data/subqas.json"
SPLIT_REL = "config/datasets/ohr_split.json"
SPLIT_LOCK_REL = "config/split_checksums.json"
MODEL_LOCK_REL = "config/model_revisions.json"
ARXIVQA_LOCK_REL = "config/arxivqa_source_lock.json"
PDF_LOCK_REL = "config/ohr_pdf_source_lock.json"
RETRIEVAL_BASE_REL = "OHR-Bench/data/retrieval_base"
GT_DIRNAME = "gt"
DEFAULT_PDF_ZIP_REL = "data/ohr_bench_raw/pdfs.zip"
PREP_ROOT_REL = "data/benchmark_prep"
PHASE0_IMAGES_REL = "artifacts/phase0/page_images"

# Project-local roots scanned for the inventory. Missing roots are findings.
PROJECT_ROOTS = [
    "OHR-Bench/data",
    "data",
    "artifacts",
    "logs",
    "results",
    "examples/demo_corpus",
    "config",
]
# Only these .env keys are read; values of all other keys are never loaded.
PATH_ENV_KEYS = [
    "FAAR_PDF_ROOT",
    "FAAR_PDF_ZIP",
    "FAAR_DOCUMENT_INVENTORY",
    "FAAR_OUT_ROOT",
    "FAAR_SCRATCH",
    "HF_HOME",
]
EXCLUDED_NAMES = {".DS_Store", "__pycache__", ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
PAGE_SUFFIX_RE = re.compile(r"^(?P<source>.+\.pdf)_(?P<page>\d+)$")
# Script presence, not language identification. Ranges are inclusive code points.
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "han": [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F)],
    "kana": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)],
    "hangul": [(0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF)],
}
INTEGRITY_CODES = {
    "lock_mismatch",
    "qa_unreadable",
    "qa_malformed_rows",
    "qa_duplicate_ids",
    "split_unreadable",
    "split_id_overlap",
    "split_unknown_ids",
    "split_duplicate_ids",
    "split_count_mismatch",
}


# ---------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def cached_sha256(path: Path) -> str:
    """sha256_file memoised by (path, size, mtime) so large archives are hashed once per run."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha256_file(path)
    return _SHA_CACHE[key]


def git_blob_sha1(path: Path) -> str:
    """Git blob id of a file's bytes, for comparison with upstream git metadata."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def git_flat_tree_sha1(directory: Path, rename: dict[str, str] | None = None) -> str:
    """Git tree id of a directory holding only regular files (git mode 100755 when owner-executable, else 100644).

    `rename` maps on-disk names to the names git upstream records, for entries
    whose names differ only in Unicode normalisation.
    """
    rename = rename or {}
    entries = []
    for p in directory.iterdir():
        if p.is_file() and not p.is_symlink() and p.name not in EXCLUDED_NAMES:
            name = rename.get(_nfc(p.name), p.name).encode("utf-8")
            mode = b"100755" if p.stat().st_mode & 0o100 else b"100644"
            entries.append((name, mode, bytes.fromhex(git_blob_sha1(p))))
    body = b"".join(mode + b" " + name + b"\0" + sha for name, mode, sha in sorted(entries))
    return hashlib.sha1(b"tree %d\0" % len(body) + body).hexdigest()


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def display_path(path: Path, project_root: Path) -> str:
    """Project-relative POSIX path, or a home-scrubbed absolute path outside it."""
    resolved = path.expanduser().absolute()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        home = str(Path.home())
        text = resolved.as_posix()
        return "~" + text[len(home):] if text.startswith(home) else text


def read_json(path: Path) -> tuple[Any, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"unreadable: {type(exc).__name__}"
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON at line {exc.lineno}"


def normalise_text(text: str) -> str:
    """Case-folded alphanumerics only; used for evidence-context alignment."""
    return re.sub(r"[\W_]+", "", text.casefold())


def load_ohr_inventory_helpers(project_root: Path):
    """Import the repo's stdlib-only alias rules; fall back to exact matching."""
    # Prefer the audited checkout's rules; fall back to this script's own checkout.
    for src in (project_root / "src", Path(__file__).resolve().parents[2] / "src"):
        if (src / "faar" / "ohr_inventory.py").is_file():
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
            break
    try:
        from faar import ohr_inventory  # noqa: PLC0415

        return (
            ohr_inventory.resolve_ohr_inventory_path,
            ohr_inventory._candidate_inventory_names,
            "faar.ohr_inventory.resolve_ohr_inventory_path",
        )
    except ImportError:

        def exact_only(inventory_dir: Path, doc_name: str):
            path = inventory_dir / f"{doc_name}.json"
            return (path, doc_name, "exact") if path.is_file() else (None, None, "missing")

        return exact_only, lambda doc: [doc], "exact-filename-only (faar.ohr_inventory not importable)"


class Issues:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, code: str, severity: str, message: str, *, count: int | None = None, ref: str | None = None) -> None:
        item: dict[str, Any] = {"code": code, "severity": severity, "message": message}
        if count is not None:
            item["count"] = count
        if ref is not None:
            item["details_ref"] = ref
        self.items.append(item)

    def sorted(self) -> list[dict[str, Any]]:
        order = {"integrity": 0, "blocking": 1, "warning": 2, "info": 3}
        return sorted(self.items, key=lambda i: (order.get(i["severity"], 9), i["code"], i["message"]))


# ---------------------------------------------------------------- configuration


def read_env_paths(project_root: Path, cli: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    """Resolve configured path keys: CLI > exported env > project .env (path keys only)."""
    dotenv: dict[str, str] = {}
    env_file = project_root / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            key, sep, value = line.strip().partition("=")
            key = key.removeprefix("export ").strip()
            if sep and key in PATH_ENV_KEYS and value.strip():
                dotenv[key] = value.strip().strip("'\"")
    resolved: dict[str, dict[str, Any]] = {}
    for key in PATH_ENV_KEYS:
        source, raw = None, None
        if cli.get(key):
            source, raw = "cli", cli[key]
        elif os.environ.get(key, "").strip():
            source, raw = "environment", os.environ[key].strip()
        elif key in dotenv:
            source, raw = ".env", dotenv[key]
        if raw is None:
            resolved[key] = {"configured": False}
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = project_root / path
        resolved[key] = {
            "configured": True,
            "source": source,
            "path": display_path(path, project_root),
            "exists": path.exists(),
            "_abs": path,
        }
    return resolved


# ---------------------------------------------------------------- inventory


def classify(rel: str, root_label: str) -> tuple[str, str]:
    """Return (asset_class, form) for a file path relative to the project."""
    name = rel.rsplit("/", 1)[-1]
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if root_label == "model_cache":
        return "model_cache", "downloaded"
    if suffix in ARCHIVE_SUFFIXES:
        return "source_archive", "compressed_input"
    if "/.cache/huggingface/" in f"/{rel}":
        return "download_cache_metadata", "generated"
    if rel.startswith("config/"):
        return "lock_or_config", "source"
    if rel.startswith("OHR-Bench/data/retrieval_base/"):
        parts = rel.split("/")
        return ("ground_truth_text" if parts[3] == GT_DIRNAME else "ocr_text"), "extracted_source"
    if rel.startswith("OHR-Bench/data/"):
        return "qa_metadata", "extracted_source"
    if rel.startswith("data/external/"):
        if suffix in {".jsonl", ".parquet", ".json", ".csv"}:
            return "qa_metadata", "downloaded"
        return "dataset_card", "downloaded"
    if rel.startswith("results/environment/"):
        return "environment_record", "generated"
    if rel.startswith(("artifacts/", "logs/", "examples/", "data/phase0/", "results/smoke/")):
        return "prototype_artifact", "generated"
    if suffix == ".pdf":
        source = root_label.startswith("configured:") or rel.startswith("data/ohr_bench_raw/")
        return "pdf", "extracted_source" if source else "prepared_copy"
    if suffix in IMAGE_SUFFIXES:
        return ("figure_image" if "/figures/" in rel else "page_image"), "generated"
    if "/ocr/" in rel and suffix == ".txt":
        return "ocr_text", "generated"
    if "/docling/" in rel:
        return "docling_output", "generated"
    if suffix in {".json", ".jsonl", ".csv", ".tsv"}:
        return "prepared_manifest", "generated"
    return "other", "generated"


def scan_root(
    root: Path,
    label: str,
    project_root: Path,
    excluded_dirs: list[Path],
    seen_inodes: dict[tuple[int, int], str],
) -> dict[str, Any]:
    entry: dict[str, Any] = {"label": label, "path": display_path(root, project_root)}
    if root.is_symlink():
        entry["status"] = "symlink_not_followed"
        return entry
    if not root.exists():
        entry["status"] = "missing"
        return entry
    if not os.access(root, os.R_OK):
        entry["status"] = "inaccessible"
        return entry
    entry["status"] = "scanned"
    files = 0
    logical = 0
    by_class: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    by_ext: Counter[str] = Counter()
    by_child: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    symlinks: list[str] = []
    hardlink_dupes: list[dict[str, str]] = []
    errors: list[str] = []
    excluded = Counter()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e.filename))):
        current = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            child = current / name
            if name in EXCLUDED_NAMES or any(child == ex or ex in child.parents for ex in excluded_dirs):
                excluded[name if name in EXCLUDED_NAMES else "audit_output"] += 1
                continue
            if child.is_symlink():
                symlinks.append(display_path(child, project_root))
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            path = current / name
            if name in EXCLUDED_NAMES or path.suffix in EXCLUDED_SUFFIXES:
                excluded[name if name in EXCLUDED_NAMES else path.suffix] += 1
                continue
            rel = display_path(path, project_root)
            try:
                st = path.lstat()
            except OSError as exc:
                errors.append(f"{rel}: {type(exc).__name__}")
                continue
            if path.is_symlink():
                symlinks.append(rel)
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen_inodes:
                hardlink_dupes.append({"path": rel, "first_seen": seen_inodes[key]})
                continue
            seen_inodes[key] = rel
            files += 1
            logical += st.st_size
            child_key = path.relative_to(root).parts[0] if len(path.relative_to(root).parts) > 1 else "<files at root>"
            by_child[child_key][0] += 1
            by_child[child_key][1] += st.st_size
            cls = classify(rel, label)
            by_class[cls][0] += 1
            by_class[cls][1] += st.st_size
            by_ext[path.suffix.lower() or "<none>"] += 1
    entry.update(
        {
            "files": files,
            "logical_bytes": logical,
            "by_asset_class": [
                {"asset_class": c, "form": f, "files": v[0], "logical_bytes": v[1]}
                for (c, f), v in sorted(by_class.items())
            ],
            "by_extension": dict(sorted(by_ext.items())),
            "by_top_level_child": {k: {"files": v[0], "logical_bytes": v[1]} for k, v in sorted(by_child.items())},
            "symlinks_not_followed": sorted(symlinks),
            "hardlink_duplicates_not_counted": sorted(hardlink_dupes, key=lambda d: d["path"]),
            "excluded": dict(sorted(excluded.items())),
            "errors": sorted(errors),
        }
    )
    return entry


def build_inventory(project_root: Path, out_dir: Path, env_paths: dict[str, dict[str, Any]], issues: Issues) -> dict[str, Any]:
    seen: dict[tuple[int, int], str] = {}
    roots: list[tuple[Path, str]] = [(project_root / rel, rel) for rel in PROJECT_ROOTS]
    for key in PATH_ENV_KEYS:
        info = env_paths[key]
        if not info.get("configured"):
            continue
        path: Path = info["_abs"]
        label = "model_cache" if key == "HF_HOME" else f"configured:{key}"
        if path.is_file():
            path = path.parent if key != "FAAR_PDF_ZIP" else path
        roots.append((path, label))
    scanned = []
    for root, label in roots:
        if root.is_file():
            st = root.stat()
            key = (st.st_dev, st.st_ino)
            rel = display_path(root, project_root)
            dup = key in seen
            seen.setdefault(key, rel)
            cls, form = classify(rel, label)
            scanned.append(
                {
                    "label": label,
                    "path": rel,
                    "status": "scanned_file",
                    "files": 0 if dup else 1,
                    "logical_bytes": 0 if dup else st.st_size,
                    "by_asset_class": [] if dup else [{"asset_class": cls, "form": form, "files": 1, "logical_bytes": st.st_size}],
                }
            )
            continue
        entry = scan_root(root, label, project_root, [out_dir], seen)
        scanned.append(entry)
        if entry["status"] != "scanned":
            sev = "info" if label in {"results", "examples/demo_corpus"} else "warning"
            issues.add("root_" + entry["status"], sev, f"Asset root {entry['path']} is {entry['status']}.")
    totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for entry in scanned:
        for row in entry.get("by_asset_class", []):
            totals[(row["asset_class"], row["form"])][0] += row["files"]
            totals[(row["asset_class"], row["form"])][1] += row["logical_bytes"]
    return {
        "policy": {
            "logical_bytes": "sum of st_size for regular files; not allocated disk usage (du)",
            "symlinks": "never followed; listed per root and not counted",
            "hard_links": "counted once by (device, inode) across all roots; later paths listed as duplicates",
            "excluded": sorted(EXCLUDED_NAMES) + sorted(EXCLUDED_SUFFIXES) + ["the audit output directory"],
            "scope": "project-local asset roots plus explicitly configured FAAR_* / HF_HOME locations only",
        },
        "roots": scanned,
        "totals_by_asset_class": [
            {"asset_class": c, "form": f, "files": v[0], "logical_bytes": v[1]} for (c, f), v in sorted(totals.items())
        ],
        "total_files": sum(v[0] for v in totals.values()),
        "total_logical_bytes": sum(v[1] for v in totals.values()),
    }


# ---------------------------------------------------------------- git and hashes


def git_info(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", "-C", str(project_root), *args], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout if done.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    if head is None:
        return {"available": False}
    status = run("status", "--porcelain=v1", "--untracked-files=normal") or ""
    dirty = sorted(line[3:] for line in status.splitlines() if line.strip())
    return {"available": True, "commit": head.strip(), "branch": (run("rev-parse", "--abbrev-ref", "HEAD") or "").strip(), "dirty": bool(dirty), "dirty_paths": dirty}


def git_blob_state(project_root: Path, rel: str) -> str:
    """'tracked-clean', 'tracked-modified', 'untracked', or 'git-unavailable'."""
    try:
        listed = subprocess.run(["git", "-C", str(project_root), "ls-files", "--error-unmatch", "--", rel], capture_output=True, text=True, timeout=60)
        if listed.returncode != 0:
            return "untracked"
        diff = subprocess.run(["git", "-C", str(project_root), "diff", "--quiet", "HEAD", "--", rel], capture_output=True, timeout=60)
        return "tracked-clean" if diff.returncode == 0 else "tracked-modified"
    except (OSError, subprocess.TimeoutExpired):
        return "git-unavailable"


def tree_digest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.name not in EXCLUDED_NAMES):
        rows.append(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}")
    return {"files": len(rows), "sha256_of_sorted_path_hash_lines": hashlib.sha256("\n".join(rows).encode()).hexdigest()}


def build_hashes(project_root: Path, issues: Issues, env_paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for rel in [QA_REL, QA_V1_REL, SUBQA_REL, SPLIT_REL, SPLIT_LOCK_REL, MODEL_LOCK_REL, ARXIVQA_LOCK_REL, PDF_LOCK_REL,
                "data/external/arxivqa/raw/arxivqa.jsonl",
                "data/external/arxivqa/jina/data/test-00000-of-00001.parquet",
                "data/external/arxivqa/vidore/test-00000-of-00001.parquet"]:
        path = project_root / rel
        if path.is_file():
            files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "git_state": git_blob_state(project_root, rel)}
        else:
            files[rel] = {"status": "missing"}

    comparisons = []
    lock, err = read_json(project_root / SPLIT_LOCK_REL)
    expected = {}
    if isinstance(lock, dict):
        expected = {SPLIT_REL: lock.get("split_sha256"), QA_REL: lock.get("qas_v2_sha256")}
    else:
        issues.add("lock_missing", "warning", f"{SPLIT_LOCK_REL} is {err}; split/QA hashes cannot be verified.")
    arx, _ = read_json(project_root / ARXIVQA_LOCK_REL)
    if isinstance(arx, dict):
        expected["data/external/arxivqa/raw/arxivqa.jsonl"] = arx.get("official_source", {}).get("sha256")
        expected["data/external/arxivqa/vidore/test-00000-of-00001.parquet"] = arx.get("vidore_source", {}).get("sha256")
    for rel, want in sorted(expected.items()):
        got = files.get(rel, {}).get("sha256")
        if want is None:
            state = "no_lock"
        elif got is None:
            state = "file_missing"
        else:
            state = "match" if got == want else "mismatch"
        comparisons.append({"path": rel, "lock_sha256": want, "actual_sha256": got, "result": state})
        if state == "mismatch":
            issues.add("lock_mismatch", "integrity", f"{rel} SHA-256 differs from its recorded lock.")
        elif state == "file_missing":
            sev = "blocking" if rel in (SPLIT_REL, QA_REL) else "info"
            issues.add("locked_file_missing", sev, f"Locked file {rel} is not present locally.")

    pdf_lock, pdf_err = read_json(project_root / PDF_LOCK_REL)
    upstream: dict[str, Any] = {"lock": PDF_LOCK_REL, "status": pdf_err or "read"}
    if isinstance(pdf_lock, dict):
        archive_lock = pdf_lock.get("pdf_archive", {})
        zip_info = env_paths["FAAR_PDF_ZIP"]
        zip_path = zip_info["_abs"] if zip_info.get("configured") else project_root / archive_lock.get("local_path", DEFAULT_PDF_ZIP_REL)
        want = archive_lock.get("sha256")
        got = cached_sha256(zip_path) if zip_path.is_file() else None
        state = "file_missing" if got is None else ("match" if got == want else "mismatch")
        comparisons.append({"path": display_path(zip_path, project_root), "lock_sha256": want, "actual_sha256": got, "result": state, "lock": PDF_LOCK_REL})
        if state == "mismatch":
            issues.add("lock_mismatch", "integrity", f"PDF archive {display_path(zip_path, project_root)} differs from {PDF_LOCK_REL}.")
        upstream.update(upstream_text_identity(project_root, pdf_lock.get("qa_and_text_reference", {}), issues))

    trees = {}
    base = project_root / RETRIEVAL_BASE_REL
    if base.is_dir():
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            trees[f"{RETRIEVAL_BASE_REL}/{child.name}"] = tree_digest(child)
    return {"files": dict(sorted(files.items())), "lock_comparisons": comparisons, "tree_digests": trees, "upstream_identity": upstream}


def upstream_text_identity(project_root: Path, ref: dict[str, Any], issues: Issues) -> dict[str, Any]:
    """Compare local QA and retrieval-base bytes with upstream git ids recorded in the lock (offline)."""
    out: dict[str, Any] = {"repository": ref.get("repository"), "commit_at_lookup": ref.get("commit_at_lookup")}
    qa_path = project_root / QA_REL
    want = ref.get("qas_v2_json", {}).get("git_blob_sha1")
    if want and qa_path.is_file():
        got = git_blob_sha1(qa_path)
        out["qas_v2_git_blob"] = {"upstream": want, "local": got, "result": "match" if got == want else "mismatch"}
    rename = {_nfc(name): unicodedata.normalize(form, name) for name, form in ref.get("upstream_name_forms", {}).items()}
    trees = {}
    for rel, want_tree in sorted(ref.get("retrieval_base_category_tree_sha1", {}).items()):
        directory = project_root / RETRIEVAL_BASE_REL / rel
        if not directory.is_dir():
            trees[rel] = {"result": "directory_missing"}
            continue
        got = git_flat_tree_sha1(directory)
        if got == want_tree:
            result = "match"
        elif git_flat_tree_sha1(directory, rename) == want_tree:
            result = "match_after_recorded_name_normalisation"
        else:
            result = "mismatch"
        trees[rel] = {"upstream": want_tree, "local": got, "result": result}
    out["retrieval_base_trees"] = trees
    counts = Counter(t["result"] for t in trees.values())
    out["retrieval_base_tree_results"] = dict(sorted(counts.items()))
    if out.get("qas_v2_git_blob", {}).get("result") == "mismatch" or counts.get("mismatch") or counts.get("directory_missing"):
        issues.add("upstream_identity_mismatch", "warning", "Local QA or retrieval-base content differs from the upstream git ids recorded in the PDF source lock.", ref="hashes.upstream_identity")
    return out


# ---------------------------------------------------------------- OHR QA and split


def scripts_present(text: str) -> list[str]:
    found = set()
    for ch in text:
        cp = ord(ch)
        if cp < 0x1100:
            continue
        for name, ranges in SCRIPT_RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                found.add(name)
    return sorted(found)


def script_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    combos = Counter("+".join(scripts_present(str(r["questions"]))) or "none" for r in rows)
    flags = {name: sum(1 for r in rows if name in scripts_present(str(r["questions"]))) for name in SCRIPT_RANGES}
    return {
        "field_checked": "questions",
        "ranges_checked": {n: [f"U+{lo:04X}-U+{hi:04X}" for lo, hi in rs] for n, rs in SCRIPT_RANGES.items()},
        "questions_containing": flags,
        "questions_containing_any": sum(n for k, n in combos.items() if k != "none"),
        "script_combinations": dict(sorted(combos.items())),
        "interpretation": "character presence only; not a language label (Han occurs in Chinese and Japanese, and short fragments such as names or units occur in otherwise non-CJK questions). No question is excluded on these flags.",
    }


def load_qa(project_root: Path, issues: Issues) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload, err = read_json(project_root / QA_REL)
    summary: dict[str, Any] = {"path": QA_REL}
    if err or not isinstance(payload, list):
        summary["status"] = err or "not a JSON list"
        issues.add("qa_unreadable", "integrity", f"{QA_REL}: {summary['status']}.")
        return [], summary
    required = {"ID": str, "doc_name": str, "questions": str, "answers": (str, int, float, list), "evidence_page_no": (int, list)}
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        problems = []
        if not isinstance(row, dict):
            malformed.append({"index": index, "problems": ["row is not an object"]})
            continue
        for key, kind in required.items():
            if key not in row:
                problems.append(f"missing {key}")
            elif not isinstance(row[key], kind) or isinstance(row[key], bool):
                problems.append(f"{key} has type {type(row[key]).__name__}")
        pages = row.get("evidence_page_no")
        page_list = pages if isinstance(pages, list) else [pages]
        if "evidence_page_no" in row and (not page_list or not all(isinstance(p, int) and not isinstance(p, bool) for p in page_list)):
            problems.append("evidence_page_no is not an integer or non-empty integer list")
        if problems:
            malformed.append({"index": index, "id": row.get("ID"), "problems": problems})
            continue
        rows.append(row)
    ids = Counter(str(r["ID"]) for r in rows)
    dup_ids = sorted(i for i, n in ids.items() if n > 1)
    docs = Counter(str(r["doc_name"]) for r in rows)
    summary.update(
        {
            "status": "read",
            "records": len(payload),
            "valid_records": len(rows),
            "malformed_records": malformed,
            "unique_ids": len(ids),
            "duplicate_ids": dup_ids,
            "raw_documents": len(docs),
            "doc_types": dict(sorted(Counter(str(r.get("doc_type", "")) for r in rows).items())),
            "evidence_sources": dict(sorted(Counter(str(r.get("evidence_source", "")) for r in rows).items())),
            "answer_forms": dict(sorted(Counter(str(r.get("answer_form", "")) for r in rows).items())),
            "multi_page_evidence_records": sum(isinstance(r["evidence_page_no"], list) for r in rows),
            "question_scripts": script_summary(rows),
            "page_excerpt_documents": sum(bool(PAGE_SUFFIX_RE.match(d.rsplit("/", 1)[-1])) for d in docs),
        }
    )
    if malformed:
        issues.add("qa_malformed_rows", "integrity", f"{len(malformed)} QA rows are malformed.", count=len(malformed), ref="ohr.qa.malformed_records")
    if dup_ids:
        issues.add("qa_duplicate_ids", "integrity", f"{len(dup_ids)} question IDs occur more than once.", count=len(dup_ids), ref="ohr.qa.duplicate_ids")

    v1, _ = read_json(project_root / QA_V1_REL)
    if isinstance(v1, list):
        v1_ids = {str(r.get("ID")) for r in v1 if isinstance(r, dict)}
        by_id = {str(r["ID"]): r for r in rows}
        v1_by_id = {str(r.get("ID")): r for r in v1 if isinstance(r, dict)}
        core = ("doc_name", "questions", "answers", "evidence_page_no")
        changed = sorted(i for i in v1_ids & by_id.keys() if any(v1_by_id[i].get(k) != by_id[i].get(k) for k in core))
        summary["qas_v1_comparison"] = {
            "qas_v1_records": len(v1),
            "v1_ids_in_v2": len(v1_ids & by_id.keys()),
            "v1_ids_not_in_v2": len(v1_ids - by_id.keys()),
            "shared_ids_with_changed_core_fields": len(changed),
        }
    return rows, summary


def load_split(project_root: Path, qa_ids: set[str], issues: Issues) -> dict[str, Any]:
    payload, err = read_json(project_root / SPLIT_REL)
    if err or not isinstance(payload, dict) or not isinstance(payload.get("splits"), dict):
        issues.add("split_unreadable", "integrity", f"{SPLIT_REL}: {err or 'missing splits object'}.")
        return {"status": err or "malformed"}
    splits = {name: [str(i) for i in ids] for name, ids in sorted(payload["splits"].items()) if isinstance(ids, list)}
    result: dict[str, Any] = {
        "status": "read",
        "recorded_source": payload.get("source"),
        "recorded_seed": payload.get("seed"),
        "recorded_policy": payload.get("split_policy"),
        "declared_counts": payload.get("counts"),
        "counts": {n: len(v) for n, v in splits.items()},
    }
    dup = {n: sorted(i for i, c in Counter(v).items() if c > 1) for n, v in splits.items()}
    unknown = {n: sorted(set(v) - qa_ids) for n, v in splits.items()}
    assigned = set().union(*[set(v) for v in splits.values()]) if splits else set()
    names = sorted(splits)
    overlaps = {f"{a}&{b}": sorted(set(splits[a]) & set(splits[b])) for i, a in enumerate(names) for b in names[i + 1:]}
    result.update(
        {
            "duplicate_ids_within_split": dup,
            "ids_not_in_qa": unknown,
            "qa_ids_not_assigned": sorted(qa_ids - assigned),
            "question_id_overlap": {k: len(v) for k, v in overlaps.items()},
            "question_id_overlap_ids": overlaps,
        }
    )
    if any(dup.values()):
        issues.add("split_duplicate_ids", "integrity", "A split lists the same question ID more than once.", ref="ohr.split.duplicate_ids_within_split")
    if any(unknown.values()):
        issues.add("split_unknown_ids", "integrity", "Split IDs are absent from the QA source.", count=sum(map(len, unknown.values())), ref="ohr.split.ids_not_in_qa")
    if any(overlaps.values()):
        issues.add("split_id_overlap", "integrity", "Question IDs occur in more than one split.", count=sum(map(len, overlaps.values())), ref="ohr.split.question_id_overlap_ids")
    if isinstance(payload.get("counts"), dict) and payload["counts"] != result["counts"]:
        issues.add("split_count_mismatch", "integrity", "Declared split counts differ from the listed IDs.")
    if result["qa_ids_not_assigned"]:
        issues.add("qa_ids_unassigned", "warning", "QA IDs are not assigned to any split.", count=len(result["qa_ids_not_assigned"]))
    result["_splits"] = splits
    return result


# ---------------------------------------------------------------- page inventories


def load_page_inventory(path: Path) -> dict[str, Any]:
    """Validate one OHR page-inventory JSON without defaulting missing page indices."""
    payload, err = read_json(path)
    info: dict[str, Any] = {"problems": []}
    if err:
        info["problems"].append(err)
        info["pages"] = {}
        return info
    if not isinstance(payload, list):
        info["problems"].append("payload is not a list")
        info["pages"] = {}
        return info
    pages: dict[int, str] = {}
    duplicates, missing_idx, bad_idx, empty = [], 0, [], []
    for row in payload:
        if not isinstance(row, dict):
            info["problems"].append("non-object entry")
            continue
        if "page_idx" not in row:
            missing_idx += 1
            continue
        idx = row["page_idx"]
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
            bad_idx.append(repr(idx))
            continue
        text = row.get("text")
        text = text if isinstance(text, str) else ""
        if idx in pages:
            duplicates.append(idx)
            pages[idx] += "\n" + text
        else:
            pages[idx] = text
    for idx, text in pages.items():
        if not text.strip():
            empty.append(idx)
    if missing_idx:
        info["problems"].append(f"{missing_idx} entries missing page_idx")
    if bad_idx:
        info["problems"].append(f"invalid page_idx values: {sorted(set(bad_idx))[:5]}")
    if duplicates:
        info["problems"].append(f"duplicate page_idx: {sorted(set(duplicates))[:10]}")
    ids = sorted(pages)
    info["pages"] = pages
    info["page_ids"] = ids
    info["empty_pages"] = sorted(empty)
    info["contiguous_from_zero"] = ids == list(range(len(ids)))
    return info


def list_inventory_keys(inv_dir: Path) -> set[str]:
    return {p.relative_to(inv_dir).with_suffix("").as_posix() for p in inv_dir.rglob("*.json") if p.is_file()}


# ---------------------------------------------------------------- PDFs and images


def pdf_parser_available() -> bool:
    try:
        import pypdfium2  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def pdf_page_count(data: bytes) -> int | None:
    """Page count via pypdfium2; None when the parser is missing, -1 when the PDF cannot be parsed."""
    try:
        import pypdfium2  # noqa: PLC0415
    except ImportError:
        return None
    try:
        document = pypdfium2.PdfDocument(data)
        try:
            return len(document)
        finally:
            document.close()
    except Exception:  # noqa: BLE001 - any parser failure (damaged or password-protected) is "unreadable"
        return -1


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def pdf_sources(project_root: Path, env_paths: dict[str, dict[str, Any]], issues: Issues) -> dict[str, Any]:
    """List candidate source PDFs by NFC document key without reading their contents.

    Candidates come from a configured PDF root and from the configured or
    default archive. Archive listing failures are findings; the archive is not
    opened again unless a selected document needs one of its members.
    Prepared copies are listed separately and never count as the source.
    """
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    locations: list[dict[str, Any]] = []
    root_info = env_paths["FAAR_PDF_ROOT"]
    if root_info.get("configured"):
        root = root_info["_abs"]
        loc: dict[str, Any] = {"kind": "pdf_root", "path": root_info["path"], "exists": root.is_dir()}
        if root.is_dir():
            count = 0
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() == ".pdf" and p.is_file() and not p.is_symlink():
                    key = p.relative_to(root).with_suffix("").as_posix()
                    candidates[_nfc(key)].append({"kind": "pdf_root", "name": key, "path": display_path(p, project_root), "_abs": p, "bytes": p.stat().st_size})
                    count += 1
            loc["pdf_files"] = count
        else:
            issues.add("pdf_root_missing", "warning", f"Configured PDF root {loc['path']} is not a directory.")
        locations.append(loc)
    zip_info = env_paths["FAAR_PDF_ZIP"]
    zip_path = zip_info["_abs"] if zip_info.get("configured") else project_root / DEFAULT_PDF_ZIP_REL
    loc = {"kind": "pdf_zip", "path": display_path(zip_path, project_root), "configured": bool(zip_info.get("configured")), "exists": zip_path.is_file()}
    if zip_path.is_file():
        loc["archive_bytes"] = zip_path.stat().st_size
        loc["sha256"] = cached_sha256(zip_path)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                infos = [i for i in archive.infolist() if not i.is_dir()]
        except (zipfile.BadZipFile, OSError) as exc:
            loc["error"] = f"unreadable archive: {type(exc).__name__}"
            issues.add("pdf_archive_unreadable", "warning", f"PDF archive {loc['path']} cannot be listed; its members are unavailable.")
        else:
            pdf_infos = [i for i in infos if i.filename.lower().endswith(".pdf")]
            prefixes = Counter(i.filename.split("/", 1)[0] for i in pdf_infos if "/" in i.filename)
            loc.update(
                {
                    "members": len(infos),
                    "pdf_members": len(pdf_infos),
                    "declared_uncompressed_bytes": sum(i.file_size for i in infos),
                    "declared_compressed_member_bytes": sum(i.compress_size for i in infos),
                    "encrypted_members": sum(bool(i.flag_bits & 0x1) for i in infos),
                    "non_pdf_members": sorted(i.filename for i in infos if not i.filename.lower().endswith(".pdf"))[:20],
                    "top_level_prefixes": dict(sorted(prefixes.items())),
                }
            )
            # Strip one wrapping directory (e.g. "pdfs/") when every member shares it
            # and it is not itself a document category.
            wrapper = next(iter(prefixes)) if len(prefixes) == 1 and pdf_infos and all("/" in i.filename for i in pdf_infos) else None
            if wrapper and all(i.filename.count("/") >= 2 for i in pdf_infos):
                loc["stripped_wrapper_directory"] = wrapper
            else:
                wrapper = None
            for i in pdf_infos:
                name = i.filename[:-4]
                if wrapper:
                    name = name[len(wrapper) + 1:]
                candidates[_nfc(name)].append({"kind": "pdf_zip", "name": name, "member": i.filename, "bytes": i.file_size, "encrypted": bool(i.flag_bits & 0x1), "_zip": zip_path})
            nfc_groups = Counter(_nfc(i.filename) for i in pdf_infos)
            loc["canonically_equivalent_member_groups"] = sum(1 for n in nfc_groups.values() if n > 1)
    locations.append(loc)
    prepared: list[dict[str, Any]] = []
    prep = project_root / PREP_ROOT_REL
    if prep.is_dir():
        for p in sorted(prep.glob("*/pdfs/**/*.pdf")):
            if p.is_file() and not p.is_symlink():
                # data/benchmark_prep/<run>/pdfs/<category>/<name>.pdf -> <category>/<name>
                key = Path(*p.relative_to(prep).parts[2:]).with_suffix("").as_posix()
                prepared.append({"doc_key": key, "path": display_path(p, project_root), "bytes": p.stat().st_size, "sha256": cached_sha256(p)})
    return {"candidates": candidates, "locations": locations, "prepared_copies": prepared}


class PdfReader:
    """Reads candidate PDF bytes, opening an archive only on first use."""

    def __init__(self) -> None:
        self._archives: dict[Path, zipfile.ZipFile | str] = {}

    def read(self, entry: dict[str, Any]) -> tuple[bytes | None, str | None]:
        if entry["kind"] == "pdf_root":
            try:
                return entry["_abs"].read_bytes(), None
            except FileNotFoundError:
                return None, "source_file_missing"
            except OSError as exc:
                return None, f"source_file_unreadable: {type(exc).__name__}"
        if entry.get("encrypted"):
            return None, "encrypted_archive_member_unsupported"
        zip_path = entry["_zip"]
        archive = self._archives.get(zip_path)
        if archive is None:
            try:
                archive = zipfile.ZipFile(zip_path)
            except (zipfile.BadZipFile, OSError) as exc:
                archive = f"archive_unreadable: {type(exc).__name__}"
            self._archives[zip_path] = archive
        if isinstance(archive, str):
            return None, archive
        try:
            return archive.read(entry["member"]), None
        except KeyError:
            return None, "archive_member_missing"
        except RuntimeError as exc:  # zipfile raises RuntimeError for password-protected members
            return None, "encrypted_archive_member_unsupported" if "encrypt" in str(exc).lower() else "archive_member_unreadable: RuntimeError"
        except NotImplementedError:
            return None, "archive_member_compression_unsupported"
        except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as exc:
            return None, f"archive_member_corrupt: {type(exc).__name__}"

    def close(self) -> None:
        for archive in self._archives.values():
            if isinstance(archive, zipfile.ZipFile):
                archive.close()


def check_document_pdfs(
    docs: list[str],
    candidate_names: dict[str, list[str]],
    pdfs: dict[str, Any],
    inventories: dict[str, dict[str, dict[str, Any]]],
    resolution: dict[str, dict[str, dict[str, Any]]],
    issues: Issues,
) -> dict[str, Any]:
    """Select one source PDF per document and compare its pages with text inventories.

    Status per document: missing_source | ambiguous_source | unreadable_source |
    parser_unavailable | unreadable_pdf | page_set_mismatch | verified_complete.
    Distinct candidate files are ambiguous unless their bytes are identical; no
    candidate is preferred merely because it reads successfully.
    """
    parser = pdf_parser_available()
    reader = PdfReader()
    per_doc: dict[str, dict[str, Any]] = {}
    used_keys: set[str] = set()
    try:
        for doc in docs:
            entries: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for name in candidate_names[doc]:
                if _nfc(name) in pdfs["candidates"]:
                    used_keys.add(_nfc(name))
                for entry in pdfs["candidates"].get(_nfc(name), []):
                    ident = (entry["kind"], entry.get("member") or entry.get("path"))
                    if ident not in seen:
                        seen.add(ident)
                        entries.append(entry)
            info: dict[str, Any] = {"candidates": [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries]}
            per_doc[doc] = info
            if not entries:
                info["status"] = "missing_source"
                continue
            payloads = [reader.read(e) for e in entries]
            errors = [err for _, err in payloads if err]
            hashes = {hashlib.sha256(data).hexdigest() for data, _ in payloads if data is not None}
            if len(entries) > 1 and (errors or len(hashes) > 1):
                info["status"] = "ambiguous_source"
                info["reason"] = "several candidate PDFs with different or unverifiable bytes; no recorded provenance selects one"
                continue
            data, err = payloads[0]
            if err:
                info["status"] = "unreadable_source"
                info["reason"] = err
                continue
            info["selected"] = info["candidates"][0]
            info["pdf_sha256"] = next(iter(hashes))
            if len(entries) > 1:
                info["identical_duplicates"] = len(entries) - 1
            if not parser:
                info["status"] = "parser_unavailable"
                continue
            n = pdf_page_count(data)
            if n is None or n < 0:
                info["status"] = "unreadable_pdf"
                continue
            info["pdf_pages"] = n
            expected = set(range(n))
            for source, inv in inventories.items():
                key = resolution[source][doc]["resolved_key"]
                if not key:
                    continue  # an absent source file is reported as a missing document, not as missing pages
                ids = set(inv.get(key, {}).get("page_ids", []))
                info[f"{source}_pages_missing_vs_pdf"] = sorted(expected - ids)
                info[f"{source}_pages_beyond_pdf"] = sorted(ids - expected)
            gt_ok = not info.get("gt_pages_missing_vs_pdf") and not info.get("gt_pages_beyond_pdf")
            info["status"] = "verified_complete" if gt_ok else "page_set_mismatch"
    finally:
        reader.close()

    status_counts = Counter(v["status"] for v in per_doc.values())
    by_status = {s: sorted(d for d, v in per_doc.items() if v["status"] == s) for s in sorted(status_counts)}
    excerpt = {d for d in docs if PAGE_SUFFIX_RE.match(d.rsplit("/", 1)[-1])}
    summary = {
        "definitions": {
            "verified_complete": "exactly one source PDF (or byte-identical duplicates) opens with the parser and its pages 0..n-1 equal the gt page_idx set",
            "page_set_mismatch": "PDF opens but its page set differs from the gt page_idx set",
            "missing_source": "no source PDF candidate for any resolved document name",
            "ambiguous_source": "several different or unverifiable candidate PDFs; none selected",
            "unreadable_source": "the file or archive member could not be read (missing, corrupt, encrypted or unsupported)",
            "parser_unavailable": "bytes read but pypdfium2 is not importable, so page coverage is unknown",
            "unreadable_pdf": "bytes read but the parser cannot open the PDF (damaged or password-protected)",
            "excerpt_document": "benchmark document whose name ends in '.pdf_<page>': a page excerpt of a larger publication; its own page set is the benchmark unit",
        },
        "parser": "pypdfium2" if parser else "unavailable",
        "denominator_documents": len(docs),
        "status_counts": dict(sorted(status_counts.items())),
        "documents_by_status": {s: v for s, v in by_status.items() if s != "verified_complete"},
        "excerpt_documents": {"total": len(excerpt), **{s: sum(1 for d in excerpt if per_doc[d]["status"] == s) for s in sorted(status_counts)}},
        "mineru_pages_missing_vs_pdf": {d: v["MinerU_pages_missing_vs_pdf"] for d, v in sorted(per_doc.items()) if v.get("MinerU_pages_missing_vs_pdf")},
        "page_set_mismatches": [
            {"doc_name": d, "pdf_pages": v["pdf_pages"], "gt_pages_missing_vs_pdf": v["gt_pages_missing_vs_pdf"][:50], "gt_pages_beyond_pdf": v["gt_pages_beyond_pdf"][:50]}
            for d, v in sorted(per_doc.items()) if v["status"] == "page_set_mismatch"
        ],
        "source_pdfs_not_mapped_to_qa": {
            "count": len(set(pdfs["candidates"]) - used_keys),
            "by_category": dict(sorted(Counter(k.split("/", 1)[0] for k in set(pdfs["candidates"]) - used_keys).items())),
        },
        "pdf_page_total_verified_documents": sum(v.get("pdf_pages", 0) for v in per_doc.values() if v["status"] == "verified_complete"),
    }
    labels = {
        "missing_source": ("pdf_document_missing", "QA documents have no source PDF candidate."),
        "ambiguous_source": ("pdf_source_ambiguous", "QA documents have several conflicting source PDF candidates."),
        "unreadable_source": ("pdf_source_unreadable", "Source PDF files or archive members could not be read."),
        "parser_unavailable": ("pdf_parser_unavailable", "pypdfium2 is not importable; PDF page coverage is unknown."),
        "unreadable_pdf": ("pdf_unreadable", "Source PDFs could not be parsed."),
        "page_set_mismatch": ("pdf_page_set_mismatch", "Source PDF page sets differ from gt page inventories."),
    }
    any_source = any(v["status"] != "missing_source" for v in per_doc.values())
    for status, (code, message) in labels.items():
        n = status_counts.get(status, 0)
        if not n or (status == "missing_source" and not any_source):
            continue
        severity = "blocking" if status == "parser_unavailable" else "warning"
        issues.add(code, severity, f"{n} {message}", count=n, ref=f"ohr.coverage.pdf_coverage.documents_by_status.{status}")
    if not any_source:
        issues.add("pdf_source_missing", "blocking", "No OHR source PDF archive or PDF root is available; page images cannot be rendered and complete-document page sets cannot be verified.")
    return {"summary": summary, "per_document": {d: v for d, v in sorted(per_doc.items())}}


def page_image_index(project_root: Path, env_paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Map document -> page ids with prepared images, and example_id -> phase0 images."""
    by_doc: dict[str, set[int]] = defaultdict(set)
    by_example: dict[str, set[int]] = defaultdict(set)
    roots = [project_root / PREP_ROOT_REL]
    if env_paths["FAAR_OUT_ROOT"].get("configured"):
        roots.append(env_paths["FAAR_OUT_ROOT"]["_abs"])
    pattern = re.compile(r"^(?P<stem>.+)_page_(?P<page>\d+)$")
    for root in roots:
        if not root.is_dir():
            continue
        for images_dir in sorted(root.glob("*/images")):
            for p in images_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file():
                    m = pattern.match(p.relative_to(images_dir).with_suffix("").as_posix())
                    if m:
                        by_doc[m["stem"]].add(int(m["page"]))
    phase0 = project_root / PHASE0_IMAGES_REL
    if phase0.is_dir():
        for p in phase0.glob("*.png"):
            m = re.match(r"^(?P<id>[0-9a-f-]{36})_p(?P<page>\d+)$", p.stem)
            if m:
                by_example[m["id"]].add(int(m["page"]))
    return {"by_doc": by_doc, "by_example": by_example}


# ---------------------------------------------------------------- OHR coverage


def ohr_coverage(
    project_root: Path,
    rows: list[dict[str, Any]],
    split: dict[str, Any],
    env_paths: dict[str, dict[str, Any]],
    issues: Issues,
) -> dict[str, Any]:
    resolve, alias_candidates, resolver_name = load_ohr_inventory_helpers(project_root)
    base = project_root / RETRIEVAL_BASE_REL
    gt_dir = env_paths["FAAR_DOCUMENT_INVENTORY"]["_abs"] if env_paths["FAAR_DOCUMENT_INVENTORY"].get("configured") else base / GT_DIRNAME
    text_dirs = {"gt": gt_dir}
    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and child.name != GT_DIRNAME:
                text_dirs[child.name] = child
    docs = sorted({str(r["doc_name"]) for r in rows})
    result: dict[str, Any] = {"alias_resolver": resolver_name, "text_sources": {}}

    # Resolve each raw document name against every text source.
    resolution: dict[str, dict[str, dict[str, Any]]] = {}
    inventories: dict[str, dict[str, dict[str, Any]]] = {}
    for source, inv_dir in text_dirs.items():
        present = inv_dir.is_dir()
        keys = list_inventory_keys(inv_dir) if present else set()
        nfc_groups: dict[str, list[str]] = defaultdict(list)
        for k in sorted(keys):
            nfc_groups[_nfc(k)].append(k)
        nfc_keys = {n: ks[0] for n, ks in nfc_groups.items()}
        equivalent_files = {n: ks for n, ks in sorted(nfc_groups.items()) if len(ks) > 1}
        ambiguous: dict[str, list[str]] = {}
        per_doc: dict[str, dict[str, Any]] = {}
        loaded: dict[str, dict[str, Any]] = {}
        for doc in docs:
            if not present:
                per_doc[doc] = {"diagnosis": "missing", "resolved_key": None}
                continue
            path, key, kind = resolve(inv_dir, doc)
            if key is not None:
                # Report the name as stored on disk; APFS resolves canonically equivalent
                # names that a byte-exact filesystem would not.
                key = key if key in keys else nfc_keys.get(_nfc(key), key)
                if key != doc and _nfc(key) == _nfc(doc):
                    kind = "unicode_normalisation_alias"
            on_disk = sorted({nfc_keys[_nfc(c)] for c in [*alias_candidates(doc), doc] if _nfc(c) in nfc_keys})
            if len(on_disk) > 1:
                ambiguous[doc] = on_disk
            per_doc[doc] = {"diagnosis": kind, "resolved_key": key}
            if path is not None and key not in loaded:
                loaded[key] = load_page_inventory(Path(path))
        # Alias collisions: distinct raw names that resolve to the same file.
        targets: dict[str, list[str]] = defaultdict(list)
        for doc, info in per_doc.items():
            if info["resolved_key"]:
                targets[info["resolved_key"]].append(doc)
        collisions = {k: sorted(v) for k, v in sorted(targets.items()) if len(v) > 1}
        referenced = set(targets)
        malformed = {k: v["problems"] for k, v in sorted(loaded.items()) if v["problems"]}
        empty = {k: v["empty_pages"] for k, v in sorted(loaded.items()) if v.get("empty_pages")}
        noncontig = sorted(k for k, v in loaded.items() if v.get("page_ids") and not v["contiguous_from_zero"])
        result["text_sources"][source] = {
            "path": display_path(inv_dir, project_root),
            "present": present,
            "inventory_files": len(keys),
            "files_referenced_by_qa": len(referenced),
            "files_not_referenced_by_qa": len(keys - referenced),
            "unreferenced_by_category": dict(sorted(Counter(k.split("/", 1)[0] for k in keys - referenced).items())),
            "documents_resolved": {kind: sum(1 for i in per_doc.values() if i["diagnosis"] == kind) for kind in ("exact", "alias", "unicode_alias", "unicode_normalisation_alias", "missing")},
            "alias_resolutions": [{"qa_doc_name": d, "resolved_key": i["resolved_key"], "diagnosis": i["diagnosis"]} for d, i in sorted(per_doc.items()) if i["diagnosis"] != "exact"],
            "unresolved_documents": sorted(d for d, i in per_doc.items() if i["diagnosis"] == "missing"),
            "alias_collisions": collisions,
            "ambiguous_mappings": ambiguous,
            "canonically_equivalent_files": equivalent_files,
            "pages_in_referenced_files": sum(len(loaded[k].get("page_ids", [])) for k in referenced),
            "malformed_files": malformed,
            "files_with_empty_pages": len(empty),
            "empty_pages_total": sum(len(v) for v in empty.values()),
            "empty_pages_by_file": empty,
            "files_not_contiguous_from_zero": noncontig,
        }
        if present and malformed:
            issues.add(f"{source}_malformed_inventory", "warning", f"{len(malformed)} {source} inventory files are malformed.", count=len(malformed), ref=f"ohr.coverage.text_sources.{source}.malformed_files")
        norm_aliases = [d for d, i in per_doc.items() if i["diagnosis"] == "unicode_normalisation_alias"]
        if norm_aliases:
            issues.add("unicode_normalisation_alias", "info", f"{len(norm_aliases)} QA document names differ from their {source} file name only in Unicode normalisation (canonically equivalent); compare names after NFC.", count=len(norm_aliases), ref=f"ohr.coverage.text_sources.{source}.alias_resolutions")
        if ambiguous:
            issues.add("ambiguous_mapping", "warning", f"{len(ambiguous)} QA document names match more than one {source} file under the alias rules.", count=len(ambiguous), ref=f"ohr.coverage.text_sources.{source}.ambiguous_mappings")
        if equivalent_files:
            issues.add("canonically_equivalent_files", "warning", f"{len(equivalent_files)} groups of {source} files have canonically equivalent names.", count=len(equivalent_files), ref=f"ohr.coverage.text_sources.{source}.canonically_equivalent_files")
        if collisions:
            issues.add("alias_collision", "warning", f"{len(collisions)} {source} files are the target of more than one raw document name.", count=len(collisions), ref=f"ohr.coverage.text_sources.{source}.alias_collisions")
        if not present:
            issues.add(f"{source}_missing", "blocking" if source == "gt" else "warning", f"Text source {source} is missing at {display_path(inv_dir, project_root)}.")
        resolution[source] = per_doc
        inventories[source] = loaded

    # MinerU-vs-gt page-set agreement per referenced document.
    if "MinerU" in inventories and "gt" in inventories:
        diff_docs = []
        for doc in docs:
            g, m = resolution["gt"][doc]["resolved_key"], resolution["MinerU"][doc]["resolved_key"]
            if g and m:
                gp = set(inventories["gt"][g].get("page_ids", []))
                mp = set(inventories["MinerU"][m].get("page_ids", []))
                if gp != mp:
                    diff_docs.append({"doc_name": doc, "gt_only_pages": sorted(gp - mp), "mineru_only_pages": sorted(mp - gp)})
        result["mineru_vs_gt_page_sets"] = {"documents_compared": sum(1 for d in docs if resolution["gt"][d]["resolved_key"] and resolution["MinerU"][d]["resolved_key"]), "documents_with_different_page_sets": len(diff_docs), "differences": diff_docs}

    pdfs = pdf_sources(project_root, env_paths, issues)
    candidate_names = {
        d: list(dict.fromkeys([*alias_candidates(d), *(resolution[s][d]["resolved_key"] for s in inventories if resolution[s][d]["resolved_key"])]))
        for d in docs
    }
    pdf_check = check_document_pdfs(docs, candidate_names, pdfs, inventories, resolution, issues)
    pdf_status = {d: v["status"] for d, v in pdf_check["per_document"].items()}
    prepared_keys = {_nfc(c["doc_key"]) for c in pdfs["prepared_copies"]}
    images = page_image_index(project_root, env_paths)

    # Per-question evidence coverage and alignment.
    norm_cache: dict[tuple[str, str, int], str] = {}

    def norm_page(source: str, key: str, page: int) -> str:
        ck = (source, key, page)
        if ck not in norm_cache:
            norm_cache[ck] = normalise_text(inventories[source][key]["pages"].get(page, ""))
        return norm_cache[ck]

    per_question: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: str(r["ID"])):
        doc = str(row["doc_name"])
        pages = row["evidence_page_no"] if isinstance(row["evidence_page_no"], list) else [row["evidence_page_no"]]
        q: dict[str, Any] = {"id": str(row["ID"]), "doc_name": doc, "evidence_pages": sorted(set(pages)), "evidence_source": row.get("evidence_source", "")}
        found_scripts = scripts_present(str(row["questions"]))
        if found_scripts:
            q["question_scripts"] = found_scripts
        for source in inventories:
            key = resolution[source][doc]["resolved_key"]
            if key is None:
                q[f"{source}_status"] = "doc_missing"
                continue
            inv = inventories[source][key]
            ids = set(inv.get("page_ids", []))
            missing = [p for p in pages if p not in ids]
            empty = [p for p in pages if p in ids and not inv["pages"][p].strip()]
            q[f"{source}_status"] = "page_missing" if missing else ("page_empty" if empty else "ok")
            if missing:
                q[f"{source}_missing_pages"] = sorted(set(missing))
        # Evidence-context alignment against the gt page set (page-number convention check).
        ctx = normalise_text(str(row.get("evidence_context", "")))
        key = resolution.get("gt", {}).get(doc, {}).get("resolved_key")
        if key is None or len(ctx) < 8:
            q["alignment"] = "not_checked"
        else:
            ids = inventories["gt"][key].get("page_ids", [])
            if any(p in ids and ctx in norm_page("gt", key, p) for p in pages):
                q["alignment"] = "on_evidence_page"
            elif any(p - 1 in ids and ctx in norm_page("gt", key, p - 1) for p in pages):
                q["alignment"] = "on_page_minus_1"
            elif any(p + 1 in ids and ctx in norm_page("gt", key, p + 1) for p in pages):
                q["alignment"] = "on_page_plus_1"
            elif any(ctx in norm_page("gt", key, p) for p in ids):
                q["alignment"] = "elsewhere_in_document"
            else:
                q["alignment"] = "not_found_verbatim"
            mkey = resolution.get("MinerU", {}).get(doc, {}).get("resolved_key")
            if mkey is not None and q["alignment"] == "on_evidence_page":
                q["mineru_evidence_string_match"] = any(ctx in norm_page("MinerU", mkey, p) for p in pages if p in inventories["MinerU"][mkey]["pages"])
        # PDF and page images.
        cands = candidate_names[doc]
        q["pdf_status"] = pdf_status[doc]
        if pdf_status[doc] == "missing_source" and any(_nfc(c) in prepared_keys for c in cands):
            q["pdf_status"] = "prepared_copy_only"
        img_pages = set()
        for c in dict.fromkeys(cands):
            img_pages |= images["by_doc"].get(c, set())
        img_pages |= images["by_example"].get(q["id"], set())
        q["evidence_images_present"] = all(p in img_pages for p in pages)
        per_question.append(q)

    def count(pred) -> int:
        return sum(1 for q in per_question if pred(q))

    total = len(per_question)
    aligned = Counter(q["alignment"] for q in per_question)
    result["page_convention"] = {
        "assumption_in_code": "evidence_page_no is used directly as the 0-based page_idx (src/faar/benchmarks.py)",
        "method": "evidence_context, case-folded and reduced to alphanumerics, searched as a substring in gt page text",
        "alignment_counts": dict(sorted(aligned.items())),
        "checked": total - aligned.get("not_checked", 0),
        "evidence_page_out_of_gt_range": count(lambda q: q.get("gt_status") == "page_missing"),
    }
    mineru_checked = [q for q in per_question if "mineru_evidence_string_match" in q]
    result["mineru_evidence_string_match"] = {
        "definition": "denominator: questions whose normalised evidence_context occurs on the gt evidence page; numerator: of those, questions where it also occurs on the MinerU evidence page",
        "interpretation_limits": [
            "exact-substring diagnostic after case folding and removal of non-alphanumerics; not a measure of OCR accuracy and not a failure label",
            "formatting and representation changes (table markup, LaTeX, reading order, hyphenation) cause mismatches when content is intact",
            "charts, formulas and multi-page evidence often have no continuous-text form and are under-represented in the denominator",
            "neither a match nor a mismatch establishes whether the question is answerable from the MinerU text; classifying real failures needs visual inspection of the page",
        ],
        "denominator": len(mineru_checked),
        "numerator": sum(q["mineru_evidence_string_match"] for q in mineru_checked),
        "by_evidence_source": {
            src: {"denominator": len(g), "numerator": sum(q["mineru_evidence_string_match"] for q in g)}
            for src, g in sorted(_group(mineru_checked, "evidence_source").items())
        },
    }
    result["question_coverage"] = {
        "denominator_questions": total,
        "definitions": {
            "<source>_evidence_ok": "document resolves, every evidence page index exists and has non-empty text",
            "pdf_status": "status of the document's source PDF (see pdf_coverage.definitions); prepared smoke copies are reported as prepared_copy_only and never count as the source",
            "evidence_images_present": "a rendered image exists for every evidence page",
        },
        **{f"{s}_evidence_ok": count(lambda q, s=s: q.get(f"{s}_status") == "ok") for s in inventories},
        **{f"{s}_status_counts": dict(sorted(Counter(q.get(f"{s}_status") for q in per_question).items())) for s in inventories},
        "pdf_status_counts": dict(sorted(Counter(q["pdf_status"] for q in per_question).items())),
        "pdf_verified_complete": count(lambda q: q["pdf_status"] == "verified_complete"),
        "evidence_images_present": count(lambda q: q["evidence_images_present"]),
    }
    result["document_coverage"] = {
        "denominator_documents": len(docs),
        **{f"{s}_resolved": sum(1 for d in docs if resolution[s][d]["resolved_key"]) for s in inventories},
        "pdf_status_counts": pdf_check["summary"]["status_counts"],
        "complete_document_coverage": {
            "definition": pdf_check["summary"]["definitions"]["verified_complete"],
            "status": "unknown" if not pdf_check["summary"]["status_counts"].get("verified_complete") and set(pdf_check["summary"]["status_counts"]) <= {"missing_source", "parser_unavailable"} else "checked",
            "numerator_verified_complete": pdf_check["summary"]["status_counts"].get("verified_complete", 0),
            "denominator_documents": len(docs),
            "gt_recorded_pages_for_qa_documents": result["text_sources"].get("gt", {}).get("pages_in_referenced_files"),
        },
    }
    result["pdf_coverage"] = pdf_check["summary"]
    result["pdf_per_document"] = pdf_check["per_document"]
    result["pdf_sources"] = {"locations": pdfs["locations"], "prepared_copies": pdfs["prepared_copies"]}

    result["ocr_conditions"] = ocr_conditions(docs, per_question, resolution, inventories, pdf_check["per_document"])

    # Split-level document overlap (raw and alias-normalised).
    splits = split.get("_splits", {})
    by_id = {str(r["ID"]): str(r["doc_name"]) for r in rows}
    gt_key = {d: (resolution.get("gt", {}).get(d, {}).get("resolved_key") or d) for d in docs}
    per_split = {}
    for name, ids in splits.items():
        raw = {by_id[i] for i in ids if i in by_id}
        per_split[name] = {
            "raw": raw,
            "normalised": {gt_key[d] for d in raw},
            "excerpt_source": {_excerpt_source(gt_key[d]) for d in raw},
        }
    names = sorted(per_split)
    overlap: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f"{a}&{b}"] = {
                "raw_document_names": len(per_split[a]["raw"] & per_split[b]["raw"]),
                "alias_normalised_documents": len(per_split[a]["normalised"] & per_split[b]["normalised"]),
                "excerpt_source_groups": len(per_split[a]["excerpt_source"] & per_split[b]["excerpt_source"]),
            }
    eval_docs_seen = {}
    train_docs = per_split.get("train", {}).get("normalised", set())
    for name in ("val", "test"):
        if name in splits:
            ids = [i for i in splits[name] if i in by_id]
            seen = sum(1 for i in ids if gt_key[by_id[i]] in train_docs)
            eval_docs_seen[name] = {"questions": len(ids), "questions_whose_document_is_in_train": seen}
    exclusive = {}
    for name in names:
        others = set().union(*[per_split[o]["normalised"] for o in names if o != name]) if len(names) > 1 else set()
        only = per_split[name]["normalised"] - others
        ids = [i for i in splits[name] if i in by_id and gt_key[by_id[i]] in only]
        exclusive[name] = {"documents": len(only), "questions": len(ids)}
    for name, s in per_split.items():
        pages = 0
        known = 0
        for d in s["raw"]:
            key = resolution.get("gt", {}).get(d, {}).get("resolved_key")
            if key:
                known += 1
                pages += len(inventories["gt"][key].get("page_ids", []))
        s["summary"] = {"raw_documents": len(s["raw"]), "alias_normalised_documents": len(s["normalised"]), "documents_with_gt_inventory": known, "gt_recorded_pages": pages}
    result["split_documents"] = {
        "normalisation_rules": [
            "existing faar.ohr_inventory alias rules: textbook_needrop_en_ -> jiaocai_needrop_en_, trailing .pdf strip, NFKD ascii fold, unique alphanumeric-skeleton match within a category",
            "a document normalises to the gt inventory key it resolves to; unresolved names keep their raw name",
        ],
        "informational_grouping": "excerpt_source_groups strips a '.pdf_<page>' suffix to group page excerpts cut from one source PDF; it is a heuristic, not an alias rule, and is not applied to other counts",
        "per_split": {n: s["summary"] for n, s in sorted(per_split.items())},
        "pairwise_overlap": overlap,
        "evaluation_questions_on_train_documents": eval_docs_seen,
        "documents_exclusive_to_split": exclusive,
        "interpretation": "the split is question-disjoint, not document-disjoint. Shared documents are leakage only if something was trained or tuned on the training questions or their documents; the split cannot support an unseen-document claim.",
    }
    result["per_question"] = per_question
    return result


def ocr_conditions(docs, per_question, resolution, inventories, pdf_docs) -> dict[str, Any]:
    """Document-level record of missing, empty or incomplete noisy text. Nothing is excluded."""
    out: dict[str, Any] = {
        "policy": "conditions are recorded for every OCR source except gt; no document or question is excluded and no text is substituted",
    }
    for source in sorted(s for s in inventories if s != "gt"):
        missing_docs, missing_pages, empty_pages = [], {}, {}
        for d in docs:
            key = resolution[source][d]["resolved_key"]
            if key is None:
                missing_docs.append(d)
                continue
            inv = inventories[source][key]
            gt_key = resolution.get("gt", {}).get(d, {}).get("resolved_key")
            reference = set(inventories.get("gt", {}).get(gt_key, {}).get("page_ids", [])) if gt_key else set()
            pdf_pages = pdf_docs.get(d, {}).get("pdf_pages")
            if pdf_pages is not None:
                reference |= set(range(pdf_pages))
            absent = sorted(reference - set(inv.get("page_ids", [])))
            if absent:
                missing_pages[d] = absent
            if inv.get("empty_pages"):
                gt_pages = inventories.get("gt", {}).get(gt_key, {}).get("pages", {}) if gt_key else {}
                empty_pages[d] = {
                    "pages": inv["empty_pages"],
                    "gt_nonempty": [pg for pg in inv["empty_pages"] if gt_pages.get(pg, "").strip()],
                }
        out[source] = {
            "reference_page_set": "gt page_idx set, extended by verified PDF pages when available",
            "documents_missing": missing_docs,
            "documents_with_missing_pages": missing_pages,
            "documents_with_empty_pages": empty_pages,
            "document_counts": {"missing": len(missing_docs), "with_missing_pages": len(missing_pages), "with_empty_pages": len(empty_pages), "with_any_condition": len(set(missing_docs) | set(missing_pages) | set(empty_pages))},
            "page_counts": {
                "missing": sum(len(v) for v in missing_pages.values()),
                "empty": sum(len(v["pages"]) for v in empty_pages.values()),
                "empty_where_gt_has_text": sum(len(v["gt_nonempty"]) for v in empty_pages.values()),
            },
            "question_status_counts": dict(sorted(Counter(q.get(f"{source}_status") for q in per_question).items())),
        }
    return out


def _group(items: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get(key, ""))].append(item)
    return groups


def _excerpt_source(key: str) -> str:
    head, _, name = key.rpartition("/")
    m = PAGE_SUFFIX_RE.match(name)
    return f"{head}/{m['source']}" if m else key


# ---------------------------------------------------------------- manifests and provenance


def prepared_manifest_checks(project_root: Path, issues: Issues) -> list[dict[str, Any]]:
    """Check that project-relative paths named inside prepared manifests exist."""
    checks = []
    prep = project_root / PREP_ROOT_REL
    if not prep.is_dir():
        return checks
    path_re = re.compile(r"^(data|artifacts|results)/[^\s]+\.(pdf|png|txt|json|md)$")
    for manifest in sorted(prep.rglob("*.json")):
        payload, err = read_json(manifest)
        rel = display_path(manifest, project_root)
        if err:
            checks.append({"manifest": rel, "status": err})
            continue
        referenced: set[str] = set()
        absolute_paths = 0
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, str):
                if path_re.match(node):
                    referenced.add(node)
                elif node.startswith(str(Path.home())):
                    absolute_paths += 1
        missing = sorted(p for p in referenced if not (project_root / p).exists())
        by_ext = Counter(p.rsplit(".", 1)[-1] for p in missing)
        checks.append(
            {
                "manifest": rel,
                "status": "read",
                "referenced_paths": len(referenced),
                "missing_paths": len(missing),
                "missing_by_extension": dict(sorted(by_ext.items())),
                "missing_examples": missing[:5],
                "absolute_home_paths": absolute_paths,
            }
        )
        if missing:
            issues.add("manifest_paths_missing", "warning", f"{rel} names {len(missing)} project paths that do not exist.", count=len(missing))
    return checks


def provenance(project_root: Path, hashes: dict[str, Any], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    cmp = {c["path"]: c for c in hashes["lock_comparisons"]}
    qa_git = hashes["files"].get(QA_REL, {}).get("git_state")
    up = hashes.get("upstream_identity", {})
    blob = up.get("qas_v2_git_blob", {}).get("result")
    trees = up.get("retrieval_base_trees", {})
    tree_ok = lambda prefix: bool(trees) and all(v["result"].startswith("match") for k, v in trees.items() if k.startswith(prefix))  # noqa: E731
    pdf_cmp = next((c for c in hashes["lock_comparisons"] if c.get("lock") == PDF_LOCK_REL), None)
    models, _ = read_json(project_root / MODEL_LOCK_REL)
    got_rev = (models or {}).get("models", {}).get("got_ocr", {}).get("revision") if isinstance(models, dict) else None
    lock_note = f"upstream git ids recorded in {PDF_LOCK_REL}"
    records = [
        {
            "asset": "OHR-Bench qas_v2.json",
            "status": "verified" if cmp.get(QA_REL, {}).get("result") == "match" else "unknown",
            "evidence": f"SHA-256 lock comparison = {cmp.get(QA_REL, {}).get('result')}; git state = {qa_git}; git blob vs {lock_note} = {blob}",
            "limitation": "Upstream identity is with a GitHub blob at a recorded commit; the release is not versioned upstream.",
        },
        {
            "asset": "OHR split (config/datasets/ohr_split.json)",
            "status": "verified" if cmp.get(SPLIT_REL, {}).get("result") == "match" else "unknown",
            "evidence": f"SHA-256 lock comparison = {cmp.get(SPLIT_REL, {}).get('result')}",
            "limitation": "Question-disjoint split (seed and policy recorded in file); not document-disjoint.",
        },
        {
            "asset": "retrieval_base/gt text",
            "status": "verified" if tree_ok("gt/") else "recorded-but-unverified",
            "evidence": f"category tree ids vs {lock_note}: " + ", ".join(f"{k}={v['result']}" for k, v in trees.items() if k.startswith("gt/")),
            "limitation": "Identity with upstream bytes only; upstream describes gt as human-verified, which this audit cannot check.",
        },
        {
            "asset": "retrieval_base/MinerU text",
            "status": "unknown",
            "evidence": f"bytes match upstream: {tree_ok('MinerU/')}. Upstream README calls gt and MinerU 'illustration' data; MinerU-0.9.3 appears only as a results-table row label.",
            "limitation": "The engine version, configuration and run date that produced these files are not recorded. Byte identity with upstream does not establish them.",
        },
        {
            "asset": "OHR PDF archive",
            "status": "verified" if pdf_cmp and pdf_cmp["result"] == "match" else "unknown",
            "evidence": f"SHA-256 vs {PDF_LOCK_REL} = {pdf_cmp['result'] if pdf_cmp else 'no lock'}; the lock records the Hugging Face LFS checksum and revision",
            "limitation": "The host checksum identifies served bytes, not authorship; no upstream manifest ties the archive to a qas_v2.json version, so compatibility rests on the name and page-count checks.",
        },
        {
            "asset": "ArXivQA arxivqa.jsonl",
            "status": "verified" if cmp.get("data/external/arxivqa/raw/arxivqa.jsonl", {}).get("result") == "match" else "unknown",
            "evidence": f"SHA-256 comparison with config/arxivqa_source_lock.json = {cmp.get('data/external/arxivqa/raw/arxivqa.jsonl', {}).get('result')}",
            "limitation": "QA source only; full-paper PDFs, page images and OCR referenced by the paper inventory are checked separately.",
        },
        {
            "asset": "ArXivQA ViDoRe parquet",
            "status": "verified" if cmp.get("data/external/arxivqa/vidore/test-00000-of-00001.parquet", {}).get("result") == "match" else "unknown",
            "evidence": f"lock comparison = {cmp.get('data/external/arxivqa/vidore/test-00000-of-00001.parquet', {}).get('result')}",
            "limitation": "The locked ViDoRe parquet is absent from the audited checkout; a different, unlocked Jina parquet is present.",
        },
    ]
    per_doc = coverage.get("pdf_per_document", {})
    smoke = project_root / PREP_ROOT_REL / "smoke/provenance"
    for p in sorted(smoke.rglob("*.json")) if smoke.is_dir() else []:
        payload, _ = read_json(p)
        if isinstance(payload, dict):
            pdf = payload.get("pdf_path")
            actual = sha256_file(project_root / pdf) if pdf and (project_root / pdf).is_file() else None
            rev = payload.get("got_ocr_revision")
            source_sha = per_doc.get(str(payload.get("doc_name")), {}).get("pdf_sha256")
            records.append(
                {
                    "asset": f"smoke GOT-OCR output for {payload.get('doc_name')}",
                    "status": "recorded-but-unverified",
                    "evidence": f"provenance records GOT-OCR revision {rev} (matches model lock: {rev == got_rev}); prepared PDF hash matches record: {actual == payload.get('pdf_sha256')}; prepared copy matches source archive member: {None if source_sha is None else source_sha == actual}",
                    "limitation": "Model snapshot files were not hashed by this audit. The prepared copy is kept distinct from the source archive.",
                }
            )
    return records


def model_cache_summary(project_root: Path, env_paths: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare locked model revisions with snapshots in an explicitly configured HF cache."""
    info = env_paths["HF_HOME"]
    if not info.get("configured"):
        return {"status": "not_configured"}
    hub = info["_abs"] / "hub" if (info["_abs"] / "hub").is_dir() else info["_abs"]
    lock, _ = read_json(project_root / MODEL_LOCK_REL)
    models = lock.get("models", {}) if isinstance(lock, dict) else {}
    rows = []
    for role, spec in sorted(models.items()):
        repo_dir = hub / ("models--" + str(spec.get("repository", "")).replace("/", "--"))
        snapshots = sorted(p.name for p in (repo_dir / "snapshots").iterdir()) if (repo_dir / "snapshots").is_dir() else []
        rows.append({"role": role, "repository": spec.get("repository"), "locked_revision": spec.get("revision"), "snapshots_present": snapshots, "locked_snapshot_present": spec.get("revision") in snapshots})
    return {"status": "scanned", "path": info["path"], "models": rows, "note": "snapshot directory presence only; snapshot files are not hashed"}


# ---------------------------------------------------------------- outputs


def strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_private(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [strip_private(v) for v in value]
    return value


def content_digest(manifest: dict[str, Any]) -> str:
    stable = json.loads(json.dumps(manifest))
    for dotted in VOLATILE_FIELDS + ["content_sha256"]:
        node = stable
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(parts[-1], None)
    return canonical_sha256(stable)


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{n} B"


def write_summary(manifest: dict[str, Any], path: Path) -> None:
    inv = manifest["inventory"]
    ohr = manifest["ohr"]
    cov = ohr.get("coverage", {})
    qc = cov.get("question_coverage", {})
    execution = manifest["audit_execution"]
    readiness = manifest["experimental_readiness"]
    lines = [
        "# Asset audit summary (generated)",
        "",
        f"Generated by `scripts/data/audit_assets.py` from `asset_manifest.json` (schema {manifest['schema_version']}, content sha256 `{manifest['content_sha256']}`).",
        f"Commit `{manifest['repository'].get('commit')}`, dirty = {manifest['repository'].get('dirty')}.",
        "",
        "## Audit execution and experimental readiness",
        "",
        f"- Audit execution: completed = {execution['completed']}, exit code {execution['exit_code']}, integrity failures {execution['integrity_failures'] or 'none'}.",
        f"- Experimental readiness: **{readiness['status']}**; blocking issue codes {readiness['blocking_issue_codes'] or 'none'}.",
        f"- {readiness['meaning']}",
        "",
        "## Inventory by root",
        "",
        "| Root | Status | Files | Logical bytes |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in inv["roots"]:
        lines.append(f"| `{r['path']}` | {r['status']} | {r.get('files', '')} | {r.get('logical_bytes', '')} |")
    lines += ["", "## Inventory by asset class", "", "| Class | Form | Files | Logical bytes |", "| --- | --- | ---: | ---: |"]
    for r in inv["totals_by_asset_class"]:
        lines.append(f"| {r['asset_class']} | {r['form']} | {r['files']} | {r['logical_bytes']} ({human_bytes(r['logical_bytes'])}) |")
    lines += ["", "## Source locks", "", "| Path | Result |", "| --- | --- |"]
    for c in manifest["hashes"]["lock_comparisons"]:
        lines.append(f"| `{c['path']}` | {c['result']} |")
    up = manifest["hashes"].get("upstream_identity", {})
    if up.get("qas_v2_git_blob"):
        lines.append(f"\nUpstream identity: qas_v2 git blob {up['qas_v2_git_blob']['result']}; retrieval-base category trees {up.get('retrieval_base_tree_results')}.")
    qa = ohr.get("qa", {})
    scripts = qa.get("question_scripts", {})
    lines += [
        "",
        "## OHR-Bench",
        "",
        f"- QA records {qa.get('records')}, unique IDs {qa.get('unique_ids')}, duplicate IDs {len(qa.get('duplicate_ids', []))}, malformed {len(qa.get('malformed_records', []))}, raw documents {qa.get('raw_documents')}.",
        f"- Split counts {ohr.get('split', {}).get('counts')}; question-ID overlap {ohr.get('split', {}).get('question_id_overlap')}.",
        f"- Question script presence (characters, not language): {scripts.get('questions_containing')}; any of these {scripts.get('questions_containing_any')}; combinations {scripts.get('script_combinations')}.",
    ]
    for key, value in sorted(qc.items()):
        if key not in ("definitions",):
            lines.append(f"- `{key}` = {value} (of {qc.get('denominator_questions')} questions)" if isinstance(value, int) and key != "denominator_questions" else f"- `{key}` = {value}")
    lines.append(f"- Page convention alignment: {cov.get('page_convention', {}).get('alignment_counts')}")
    sm = cov.get("mineru_evidence_string_match", {})
    lines.append(f"- MinerU evidence string-match diagnostic (not an OCR-accuracy measure or failure label): {sm.get('numerator')}/{sm.get('denominator')}")
    pc = cov.get("pdf_coverage", {})
    if pc:
        lines += [
            "",
            "## Source PDFs",
            "",
            f"- Parser: {pc.get('parser')}; documents {pc.get('denominator_documents')}; status counts {pc.get('status_counts')}.",
            f"- Excerpt documents ('.pdf_<page>' names): {pc.get('excerpt_documents')}.",
            f"- Pages in verified-complete documents: {pc.get('pdf_page_total_verified_documents')}.",
        ]
        for loc in cov.get("pdf_sources", {}).get("locations", []):
            if loc.get("exists"):
                lines.append(f"- `{loc['path']}`: {loc.get('archive_bytes', '')} bytes, {loc.get('pdf_members', loc.get('pdf_files'))} PDFs, declared uncompressed {loc.get('declared_uncompressed_bytes', 'n/a')} bytes.")
    oc = cov.get("ocr_conditions", {})
    for source, v in sorted(oc.items()):
        if isinstance(v, dict):
            lines.append(f"- {source} OCR conditions (recorded, not excluded): documents {v['document_counts']}; questions {v['question_status_counts']}.")
    for pair, v in sorted(cov.get("split_documents", {}).get("pairwise_overlap", {}).items()):
        lines.append(f"- Document overlap {pair}: {v}")
    lines += ["", "## Issues", "", "| Severity | Code | Message |", "| --- | --- | --- |"]
    for issue in manifest["issues"]:
        lines.append(f"| {issue['severity']} | {issue['code']} | {issue['message']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(project_root: Path, out_dir: Path, cli_paths: dict[str, str | None]) -> tuple[dict[str, Any], int]:
    started = datetime.now(UTC)
    issues = Issues()
    env_paths = read_env_paths(project_root, cli_paths)
    for key, info in env_paths.items():
        if info.get("configured") and not info["exists"]:
            issues.add("configured_path_missing", "warning", f"{key} is configured but {info['path']} does not exist.")
    inventory = build_inventory(project_root, out_dir, env_paths, issues)
    hashes = build_hashes(project_root, issues, env_paths)
    rows, qa_summary = load_qa(project_root, issues)
    split = load_split(project_root, {str(r["ID"]) for r in rows}, issues) if rows else {"status": "skipped: QA unreadable"}
    coverage = ohr_coverage(project_root, rows, split, env_paths, issues) if rows else {}
    manifests = prepared_manifest_checks(project_root, issues)
    prov = provenance(project_root, hashes, coverage)
    if not env_paths["HF_HOME"].get("configured"):
        issues.add("model_cache_not_configured", "info", "HF_HOME is not configured; the default user cache was not scanned. Pass --hf-cache to include one.")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": started.isoformat(timespec="seconds"),
        "volatile_fields": VOLATILE_FIELDS,
        "audit_command": "scripts/data/audit_assets.py",
        "repository": git_info(project_root),
        "scan_scope": {
            "project_root": ".",
            "project_roots": PROJECT_ROOTS,
            "configured_locations": {k: strip_private(v) for k, v in env_paths.items()},
            "not_scanned": "personal directories outside the project and unconfigured caches",
            "output_dir_excluded": display_path(out_dir, project_root),
        },
        "inventory": inventory,
        "hashes": hashes,
        "provenance": prov,
        "ohr": {"qa": qa_summary, "split": strip_private(split), "coverage": strip_private(coverage)},
        "prepared_manifests": manifests,
        "model_cache": model_cache_summary(project_root, env_paths),
        "issues": issues.sorted(),
    }
    exit_code = 2 if any(i["code"] in INTEGRITY_CODES for i in manifest["issues"]) else 0
    manifest["exit_code"] = exit_code
    manifest["audit_execution"] = {
        "completed": True,
        "exit_code": exit_code,
        "integrity_failures": sorted({i["code"] for i in manifest["issues"] if i["code"] in INTEGRITY_CODES}),
        "meaning": "the audit ran to completion; this says nothing about experimental readiness",
    }
    blocking = sorted({i["code"] for i in manifest["issues"] if i["severity"] in ("integrity", "blocking")})
    manifest["experimental_readiness"] = {
        "status": "blocked" if blocking else "no_recorded_blockers",
        "blocking_issue_codes": blocking,
        "meaning": "blocked: an integrity or blocking issue prevents preparation or a pilot. no_recorded_blockers: the audited assets are usable in principle; warnings and the lead's scope decisions still apply before any experiment.",
    }
    manifest["runtime_sec"] = round((datetime.now(UTC) - started).total_seconds(), 1)
    manifest["content_sha256"] = content_digest(manifest)
    return manifest, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only FAAR asset audit (no CUDA, no network, no credentials).")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory (excluded from the inventory).")
    parser.add_argument("--pdf-root", help="Explicit OHR PDF directory (overrides FAAR_PDF_ROOT).")
    parser.add_argument("--pdf-zip", help="Explicit OHR PDF archive (overrides FAAR_PDF_ZIP).")
    parser.add_argument("--inventory", help="Explicit gt page-inventory directory (overrides FAAR_DOCUMENT_INVENTORY).")
    parser.add_argument("--hf-cache", help="Explicit Hugging Face cache to inventory (overrides HF_HOME).")
    args = parser.parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        print(f"error: project root is not a directory: {project_root}", file=sys.stderr)
        return 1
    out_dir = args.out_dir.expanduser()
    out_dir = (out_dir if out_dir.is_absolute() else Path.cwd() / out_dir).resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create output directory {out_dir}: {exc}", file=sys.stderr)
        return 1
    cli = {"FAAR_PDF_ROOT": args.pdf_root, "FAAR_PDF_ZIP": args.pdf_zip, "FAAR_DOCUMENT_INVENTORY": args.inventory, "HF_HOME": args.hf_cache}
    manifest, code = run_audit(project_root, out_dir, cli)
    (out_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(manifest, out_dir / "asset_audit_summary.md")
    counts = Counter(i["severity"] for i in manifest["issues"])
    print(f"audit complete: exit={code} issues={dict(sorted(counts.items()))} content_sha256={manifest['content_sha256']}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
