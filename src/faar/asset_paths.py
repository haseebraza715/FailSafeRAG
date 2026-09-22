from __future__ import annotations

from pathlib import Path


class AssetPathError(ValueError):
    """Raised when a benchmark asset path is not project-portable or escapes the root."""


def ensure_within_project(path: Path, project_root: Path) -> Path:
    root = project_root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AssetPathError(
            f"Asset path escapes project root: {resolved} (project_root={root})"
        ) from exc
    return resolved


def to_relative_project_path(path: Path | str, project_root: Path) -> str:
    """Store manifests with project-relative POSIX paths (never absolute home paths)."""
    resolved = ensure_within_project(Path(path), project_root)
    return resolved.relative_to(project_root.expanduser().resolve()).as_posix()


def resolve_project_asset(path_value: str | Path, project_root: Path) -> Path:
    """Resolve a manifest path against project_root and reject traversal escapes."""
    root = project_root.expanduser().resolve()
    raw = Path(path_value).expanduser()
    candidate = raw if raw.is_absolute() else (root / raw)
    # Reject explicit parent traversal before resolve collapses it for relative inputs.
    if not raw.is_absolute() and ".." in raw.parts:
        raise AssetPathError(f"Asset path must not contain '..': {path_value}")
    return ensure_within_project(candidate, root)
