from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_INVENTORY = "OHR-Bench/data/retrieval_base/gt"
DEFAULT_PDF_ZIP = "data/ohr_bench_raw/pdfs.zip"
SPLIT_RELATIVE_PATH = Path("config/datasets/ohr_split.json")

INVENTORY_LABEL = "document inventory"
PDF_ROOT_LABEL = "FAAR_PDF_ROOT / --pdf-root"
PDF_ZIP_LABEL = "FAAR_PDF_ZIP / --pdf-zip"


class DatasetPathError(ValueError):
    """Raised when a configured dataset path is missing or the wrong kind."""


def env_raw(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip()


def load_project_dotenv(project_root: Path) -> bool:
    """Load the ignored project `.env` without overriding exported values.

    Cluster entry points and preparation CLIs read FAAR_* paths from the
    process environment. Supervisors are told to put those values in `.env`.
    """
    env_file = resolve_against_project_root(".env", project_root)
    if not env_file.is_file():
        return False
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_file, override=False)
    return True


def resolve_against_project_root(raw: str | Path, project_root: Path) -> Path:
    """Resolve a configured path against the explicit project root, never cwd."""
    root = project_root.expanduser().resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise DatasetPathError(f"{label} is not a directory: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise DatasetPathError(f"{label} is not a file: {path}")
    return path


@dataclass(frozen=True)
class DatasetPaths:
    inventory_dir: Path
    pdf_root: Path | None
    pdf_zip: Path | None


def resolve_dataset_paths(
    *,
    project_root: Path,
    inventory: str | Path | None = None,
    pdf_root: str | Path | None = None,
    pdf_zip: str | Path | None = None,
    require_pdf_source: bool = False,
) -> DatasetPaths:
    """Resolve inventory and PDF sources against project_root and check kinds.

    Every explicitly configured path is validated even when another PDF source
    is also set. Relative values are resolved against project_root, not cwd.
    The default inventory directory is always required. The default PDF zip is
    required only when require_pdf_source is true and no pdf_root is configured.
    """
    root = project_root.expanduser().resolve()
    inventory_value = DEFAULT_INVENTORY if inventory is None else inventory
    inventory_dir = require_directory(
        resolve_against_project_root(inventory_value, root),
        INVENTORY_LABEL,
    )

    resolved_pdf_root = None
    if pdf_root is not None:
        resolved_pdf_root = require_directory(
            resolve_against_project_root(pdf_root, root),
            PDF_ROOT_LABEL,
        )

    resolved_pdf_zip = None
    if pdf_zip is not None:
        resolved_pdf_zip = require_file(
            resolve_against_project_root(pdf_zip, root),
            PDF_ZIP_LABEL,
        )
    elif require_pdf_source and resolved_pdf_root is None:
        resolved_pdf_zip = require_file(
            resolve_against_project_root(DEFAULT_PDF_ZIP, root),
            PDF_ZIP_LABEL,
        )

    return DatasetPaths(
        inventory_dir=inventory_dir,
        pdf_root=resolved_pdf_root,
        pdf_zip=resolved_pdf_zip,
    )
