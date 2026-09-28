from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


class GitScanError(RuntimeError):
    pass


def _git_list_files(root: Path, *flags: str) -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", *flags],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GitScanError(
            f"'git ls-files' exited {completed.returncode}: {stderr or 'unknown error'}"
        )
    return [
        entry.decode("utf-8", errors="surrogateescape")
        for entry in completed.stdout.split(b"\x00")
        if entry
    ]


def scan_git_matches(root: Path, term: str) -> list[str]:
    tracked = _git_list_files(root)
    untracked = _git_list_files(root, "--others", "--exclude-standard")
    needle = term.encode("utf-8")
    matches: list[str] = []
    for relative in tracked + untracked:
        path = root / relative
        try:
            content = (
                os.readlink(path).encode("utf-8", errors="surrogateescape")
                if path.is_symlink()
                else path.read_bytes()
            )
        except OSError as exc:
            raise GitScanError(f"could not scan {relative!r}: {exc}") from exc
        if needle in content:
            matches.append("./" + relative)
    return matches


def scan_git_pattern_matches(root: Path, pattern: str) -> list[str]:
    regex = re.compile(pattern.encode("utf-8"))
    tracked = _git_list_files(root)
    untracked = _git_list_files(root, "--others", "--exclude-standard")
    matches: list[str] = []
    for relative in tracked + untracked:
        path = root / relative
        try:
            content = (
                os.readlink(path).encode("utf-8", errors="surrogateescape")
                if path.is_symlink()
                else path.read_bytes()
            )
        except OSError as exc:
            raise GitScanError(f"could not scan {relative!r}: {exc}") from exc
        if regex.search(content):
            matches.append("./" + relative)
    return matches


def pdf_author(pdf_path: Path) -> str | None:
    if not pdf_path.exists():
        return None
    if shutil.which("pdfinfo") is None:
        return None
    completed = subprocess.run(["pdfinfo", str(pdf_path)], text=True, capture_output=True, check=False)
    for line in completed.stdout.splitlines():
        if line.startswith("Author:"):
            return line.split(":", 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FAAR plan's before-submitting anonymization checks.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--paper", type=Path, default=Path("paper.pdf"))
    parser.add_argument("--out", type=Path, default=Path("paper_artifacts/pre_submission_check.json"))
    args = parser.parse_args()

    root = args.root.resolve()
    paper = args.paper if args.paper.is_absolute() else root / args.paper
    allowed_checklist_references = {
        "./docs/research/aaai-plan.md",
        "./docs/history/aaai-execution-status.md",
        "./scripts/release/pre_submission_check.py",
    }
    scanner_self_reference = "./scripts/release/pre_submission_check.py"
    try:
        raw_handle_matches = scan_git_matches(root, "haseebraza")
        raw_name_matches = scan_git_matches(root, "your-real-name")
        raw_home_matches = scan_git_pattern_matches(root, r"/Users/|/home/")
    except GitScanError as exc:
        raise SystemExit(
            f"Pre-submission check could not enumerate repository files: {exc}"
        ) from exc
    report = {
        "haseebraza_matches": raw_handle_matches,
        "your_real_name_matches": raw_name_matches,
        "home_path_matches": [path for path in raw_home_matches if path != scanner_self_reference],
        "unexpected_haseebraza_matches": [path for path in raw_handle_matches if path not in allowed_checklist_references],
        "unexpected_your_real_name_matches": [path for path in raw_name_matches if path not in allowed_checklist_references],
        "pdf_path": str(paper),
        "pdf_present": paper.exists(),
        "pdf_author": pdf_author(paper),
    }
    report["passed"] = (
        not report["unexpected_haseebraza_matches"]
        and not report["unexpected_your_real_name_matches"]
        and not report["home_path_matches"]
        and report["pdf_present"]
        and report["pdf_author"] == ""
    )
    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Pre-submission anonymization check failed.")


if __name__ == "__main__":
    main()
