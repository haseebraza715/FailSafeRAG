from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar.annotation import label_studio_config, write_label_studio_tasks
from faar.ocr import GOT_OCR_MODEL, extract_got_ocr
from faar.settings import AppSettings

IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40}")
MUTABLE_REVISION_NAMES = {"main", "master", "latest", "head"}
MODEL_REVISIONS_PATH = ROOT / "config" / "model_revisions.json"


def _committed_got_ocr_lock() -> dict[str, str]:
    """Read GOT-OCR lock from the committed config file (never env-overridable settings)."""
    payload = json.loads(MODEL_REVISIONS_PATH.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    locked = models.get("got_ocr") if isinstance(models, dict) else None
    if not isinstance(locked, dict):
        raise ValueError("config/model_revisions.json is missing models.got_ocr.")
    repository = str(locked.get("repository") or "").strip()
    revision = str(locked.get("revision") or "").strip()
    if not repository or not revision:
        raise ValueError("config/model_revisions.json got_ocr lock is incomplete.")
    return {"repository": repository, "revision": revision}


def validate_locked_got_ocr(
    model_name: str,
    revision: str | None,
    settings: AppSettings | None = None,
) -> str:
    _ = settings  # Kept for call-site compatibility; lock comes from committed JSON only.
    candidate = (revision or "").strip()
    if not candidate:
        raise ValueError("GOT-OCR revision must not be empty.")
    if candidate.lower() in MUTABLE_REVISION_NAMES:
        raise ValueError(f"GOT-OCR revision must be an immutable commit SHA, not branch {candidate!r}.")
    if not IMMUTABLE_REVISION_RE.fullmatch(candidate):
        raise ValueError("GOT-OCR revision must be a lowercase 40-character hexadecimal commit SHA.")
    locked = _committed_got_ocr_lock()
    if model_name != locked["repository"]:
        raise ValueError(
            f"GOT-OCR model {model_name!r} does not match locked repository {locked['repository']!r}."
        )
    if candidate != locked["revision"]:
        raise ValueError("GOT-OCR revision does not match config/model_revisions.json.")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract annotation-study OCR text with GOT-OCR 2.0.")
    parser.add_argument("--engine", required=True, choices=["got-ocr-2"])
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    settings = AppSettings(project_root=Path.cwd())
    model_name = (
        args.model
        if args.model is not None
        else os.getenv("GOT_OCR_MODEL", settings.recovery.got_ocr_model or GOT_OCR_MODEL)
    )
    revision = (
        args.revision
        if args.revision is not None
        else os.getenv("GOT_OCR_MODEL_REVISION", settings.recovery.got_ocr_revision)
    )
    revision = validate_locked_got_ocr(model_name, revision, settings)

    payload = json.loads(args.samples.read_text())
    samples = payload.get("samples", payload)
    if not isinstance(samples, list):
        raise ValueError("samples.json must contain a list under 'samples'.")
    args.out.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        image_paths = [Path(path) for path in sample.get("image_paths", [])]
        if not image_paths:
            raise FileNotFoundError(f"No page image recorded for annotation example {sample.get('example_id')}.")
        pages = []
        for index, image_path in enumerate(image_paths):
            text = extract_got_ocr(image_path, model_name=model_name, revision=revision)
            pages.append(f"===== PAGE {index} =====\n{text}")
        (args.out / f"{sample['example_id']}.txt").write_text("\n\n".join(pages) + "\n")

    annotation_root = args.out.parent
    label_studio_dir = annotation_root / "label_studio"
    label_studio_dir.mkdir(parents=True, exist_ok=True)
    (label_studio_dir / "config.xml").write_text(label_studio_config())
    write_label_studio_tasks(samples, args.out, label_studio_dir / "tasks.json")
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "engine": args.engine,
        "model": model_name,
        "revision": revision,
        "count": len(samples),
        "label_set": ["semantic", "word_level", "structural", "other"],
    }
    (label_studio_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"ocr_dir": str(args.out), "label_studio_tasks": str(label_studio_dir / 'tasks.json')}, indent=2))


if __name__ == "__main__":
    main()
