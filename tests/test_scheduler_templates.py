from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "cluster/templates"

EXPECTED_ENTRYPOINTS = {
    "slurm_preflight.sbatch": [
        ("cluster/preflight.py", {"--check", "--project-root", "--out"}),
    ],
    "slurm_calibration_108.sbatch": [
        ("cluster/launcher.py", {"--project-root", "--entrypoint", "--preflight-out"}),
        (
            "scripts/data/prepare_benchmark_assets.py",
            {"--dataset", "--out-root", "--smoke-doc", "--require-calibration-108", "--execute"},
        ),
    ],
    "slurm_prepare_val_shard.sbatch": [
        ("cluster/launcher.py", {"--project-root", "--entrypoint", "--preflight-out"}),
        ("cluster/prepare_assets.py", {"--split", "--shard-index", "--num-shards", "--out-root", "--resume"}),
    ],
    "slurm_pilot_baseline.sbatch": [
        ("cluster/run_baselines.py", {"--project-root", "--dataset", "--split", "--max-examples", "--resume"}),
    ],
}

LAUNCHER_FLAGS = {"--project-root", "--entrypoint", "--preflight-out", "--cpu-only", "--faar-python"}


def _template_invocations(template: Path) -> list[tuple[str, set[str]]]:
    lines = template.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if '"${FAAR_PYTHON}"' in line]
    blocks: list[list[str]] = []
    for start in starts:
        block = [lines[start]]
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "export", "mkdir", "if", "done", "cd", "set")):
                break
            block.append(line)
        blocks.append(block)

    invocations: list[tuple[str, set[str]]] = []
    for block in blocks:
        tokens: list[str] = []
        for line in block:
            tokens.extend(shlex.split(line.replace('"${FAAR_PYTHON}"', "python").replace("\\", " ")))
        if len(tokens) < 2:
            continue
        script = tokens[1]
        flags = {t for t in tokens[2:] if t.startswith("--")}
        if script == "cluster/launcher.py":
            at = tokens.index("--entrypoint") if "--entrypoint" in tokens else None
            child_flags = {
                t
                for t in tokens[at + 2 :]
                if t.startswith("--") and t not in LAUNCHER_FLAGS
            } if at is not None else set()
            invocations.append((script, {t for t in flags if t in LAUNCHER_FLAGS}))
            if at is not None and tokens[at + 1]:
                invocations.append((tokens[at + 1], child_flags))
        else:
            invocations.append((script, flags))
    return invocations


@pytest.mark.parametrize("name", sorted(EXPECTED_ENTRYPOINTS))
def test_template_invokes_real_supported_arguments(name: str) -> None:
    expected = EXPECTED_ENTRYPOINTS[name]
    invocations = _template_invocations(TEMPLATES / name)
    assert invocations, f"{name} contains no python invocation"
    for script, flags in invocations:
        completed = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert completed.returncode == 0, f"{script} --help failed: {completed.stderr}"
        help_text = completed.stdout + completed.stderr
        for flag in flags:
            assert flag in help_text, f"{script} --help does not document {flag}"
    found = [(script, flags) for script, flags in invocations if any(flags)]
    for script, expected_flags in expected:
        actual = next((flags for s, flags in found if s == script), set())
        assert expected_flags.issubset(actual), f"{name} is missing {expected_flags - actual} for {script}"


def test_template_dry_run_flags_are_safe() -> None:
    for name, flags in _template_invocations(TEMPLATES / "slurm_preflight.sbatch"):
        assert "--dry-run" not in flags  # preflight --check must not be muted


def test_templates_never_hardcode_private_values() -> None:
    sensitive = re.compile(
        r"(haseeb|raza|\/Users\/|\/home\/|OPENAI_API_KEY=sk|sk-[A-Za-z0-9]{10,}|AKIA|\b[\w.+-]+@[\w-]+\.[\w.]+\b)",
        re.IGNORECASE,
    )
    for name in (
        "slurm_preflight.sbatch",
        "slurm_calibration_108.sbatch",
        "slurm_prepare_val_shard.sbatch",
        "slurm_pilot_baseline.sbatch",
    ):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            assert not sensitive.search(line), f"{name}:{line_no} looks hard-coded: {line.strip()}"
            if "PARTITION" in line and "EDIT" not in line:
                assert False, f"{name}:{line_no} has a non-placeholder partition"


def test_templates_source_project_dotenv() -> None:
    for name in (
        "slurm_preflight.sbatch",
        "slurm_calibration_108.sbatch",
        "slurm_prepare_val_shard.sbatch",
        "slurm_pilot_baseline.sbatch",
        "slurm_one_gpu.sbatch",
        "pbs_one_gpu.pbs",
    ):
        text = (TEMPLATES / name).read_text(encoding="utf-8")
        assert "source .env" in text, f"{name} does not source the project .env"
        source_at = text.index("source .env")
        scratch_at = text.find("FAAR_SCRATCH=")
        if scratch_at != -1:
            assert source_at < scratch_at, f"{name} uses FAAR_SCRATCH before sourcing .env"


def test_all_templates_pass_bash_syntax() -> None:
    for path in sorted(TEMPLATES.glob("*")):
        if path.suffix not in {".sbatch", ".pbs"}:
            continue
        completed = subprocess.run(["bash", "-n", str(path)], check=False, capture_output=True, text=True)
        assert completed.returncode == 0, f"{path.name}: {completed.stderr}"


def test_one_gpu_template_has_no_hardcoded_scheduler_defaults() -> None:
    text = (TEMPLATES / "slurm_one_gpu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=<EDIT_PARTITION>" in text
    assert "#SBATCH --qos=<EDIT_QOS>" in text
    assert "#SBATCH --account=<EDIT_ACCOUNT>" in text
    assert not re.search(r"^#SBATCH --partition=gpu\s*$", text, re.MULTILINE)
    assert not re.search(r"^#SBATCH --qos=normal\s*$", text, re.MULTILINE)
    preflight = (TEMPLATES / "slurm_preflight.sbatch").read_text(encoding="utf-8")
    calibration = (TEMPLATES / "slurm_calibration_108.sbatch").read_text(encoding="utf-8")
    assert "--require-calibration-108" in calibration
    assert "cluster/preflight.py" in preflight
    assert "#SBATCH --partition=<EDIT: partition that offers the intended GPU, e.g. gpu>" in preflight
    assert "#SBATCH --partition=<EDIT: partition that offers the intended GPU, e.g. gpu>" in calibration


def test_one_gpu_template_fails_closed_before_python(tmp_path: Path) -> None:
    script = tmp_path / "slurm_one_gpu.sbatch"
    script.write_text((TEMPLATES / "slurm_one_gpu.sbatch").read_text(encoding="utf-8"), encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\necho PYTHON_STARTED > python-started\nexit 0\n")
    fake_python.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "FAAR_PYTHON": str(fake_python)},
    )
    assert completed.returncode == 2
    assert "<EDIT_PARTITION>" in completed.stderr
    assert "not the bounded 108-page calibration" in completed.stderr
    assert not (tmp_path / "python-started").exists()


def test_templates_include_signal_and_resume_behavior() -> None:
    calibration = (TEMPLATES / "slurm_calibration_108.sbatch").read_text()
    sharded = (TEMPLATES / "slurm_prepare_val_shard.sbatch").read_text()
    pilot = (TEMPLATES / "slurm_pilot_baseline.sbatch").read_text()
    one_gpu = (TEMPLATES / "slurm_one_gpu.sbatch").read_text()
    assert "--signal=TERM@120" in calibration
    assert "--signal=TERM@120" in sharded
    assert "--signal=TERM@120" in pilot
    assert "--resume" in sharded
    assert "--resume" in pilot
    assert "This is NOT the bounded 108-page calibration" in one_gpu
    assert "Do not submit this file for supervisor preflight or calibration" in one_gpu
