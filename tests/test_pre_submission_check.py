from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CHECK_PATH = Path(__file__).resolve().parents[1] / "scripts/release/pre_submission_check.py"
CHECK_SPEC = spec_from_file_location("pre_submission_check", CHECK_PATH)
assert CHECK_SPEC and CHECK_SPEC.loader
pre_submission_check = module_from_spec(CHECK_SPEC)
sys.modules["pre_submission_check"] = pre_submission_check
CHECK_SPEC.loader.exec_module(pre_submission_check)

GitScanError = pre_submission_check.GitScanError
scan_git_matches = pre_submission_check.scan_git_matches
HANDLE = "hasee" + "braza"
PLACEHOLDER = "your-real" + "-name"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def test_finds_tracked_and_non_ignored_untracked_matches(git_repo: Path) -> None:
    tracked = git_repo / "tracked.md"
    tracked.write_text(f"author {HANDLE}\n", encoding="utf-8")
    _git(git_repo, "add", "tracked.md")
    untracked = git_repo / "untracked.md"
    untracked.write_text(f"author {PLACEHOLDER}\n", encoding="utf-8")

    assert scan_git_matches(git_repo, HANDLE) == ["./tracked.md"]
    assert scan_git_matches(git_repo, PLACEHOLDER) == ["./untracked.md"]


def test_ignored_secret_env_and_venv_files_not_read_or_reported(
    git_repo: Path,
) -> None:
    (git_repo / ".gitignore").write_text(".env\n.venv/\nbuild/\n", encoding="utf-8")
    (git_repo / ".env").write_text(f"API_KEY={HANDLE}\n", encoding="utf-8")
    (git_repo / ".venv").mkdir()
    (git_repo / ".venv" / "site.txt").write_text(
        f"owner {PLACEHOLDER}\n", encoding="utf-8"
    )
    (git_repo / "build").mkdir()
    (git_repo / "build" / "cache.txt").write_text(
        f"author {HANDLE}\n", encoding="utf-8"
    )
    _git(git_repo, "add", ".gitignore")

    assert scan_git_matches(git_repo, HANDLE) == []
    assert scan_git_matches(git_repo, PLACEHOLDER) == []


def test_binary_files_do_not_crash(git_repo: Path) -> None:
    binary_hit = git_repo / "figure.bin"
    binary_hit.write_bytes(b"\x00\x01\xff" + HANDLE.encode() + b"\x00\x02binary")
    binary_miss = git_repo / "figure_miss.bin"
    binary_miss.write_bytes(b"\x00\x01\xff\xfe\x00\x02data")
    _git(git_repo, "add", "figure.bin", "figure_miss.bin")

    assert scan_git_matches(git_repo, HANDLE) == ["./figure.bin"]


def test_git_enumeration_failure_is_fail_closed(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_DIR", str(git_repo / "does-not-exist"))
    with pytest.raises(GitScanError):
        scan_git_matches(git_repo, HANDLE)


def test_tracked_symlink_is_scanned_without_following_target(
    git_repo: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text(HANDLE, encoding="utf-8")
    link = git_repo / "outside-link"
    link.symlink_to(outside)
    _git(git_repo, "add", "outside-link")

    assert scan_git_matches(git_repo, HANDLE) == []


def test_unreadable_tracked_file_fails_closed(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracked = git_repo / "tracked.md"
    tracked.write_text("safe", encoding="utf-8")
    _git(git_repo, "add", "tracked.md")
    original = Path.read_bytes

    def fail_selected(path: Path) -> bytes:
        if path == tracked:
            raise OSError("permission denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_selected)
    with pytest.raises(GitScanError, match="could not scan"):
        scan_git_matches(git_repo, HANDLE)
