from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

CLUSTER = Path(__file__).resolve().parents[1] / "cluster"
PREFLIGHT_SPEC = spec_from_file_location("preflight", CLUSTER / "preflight.py")
assert PREFLIGHT_SPEC and PREFLIGHT_SPEC.loader
preflight = module_from_spec(PREFLIGHT_SPEC)
sys.modules["preflight"] = preflight
PREFLIGHT_SPEC.loader.exec_module(preflight)
LAUNCHER_SPEC = spec_from_file_location("launcher", CLUSTER / "launcher.py")
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = module_from_spec(LAUNCHER_SPEC)
sys.modules["launcher"] = launcher
LAUNCHER_SPEC.loader.exec_module(launcher)

GIB = 1024**3


@pytest.fixture(autouse=True)
def isolate_launcher_env(monkeypatch) -> None:
    for name in (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "VLM_BACKEND",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "FAAR_PREFLIGHT_JSON",
        "FAAR_PYTHON",
        "CUDA_VISIBLE_DEVICES",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SCRATCH",
        "SCRATCH_DIR",
        "LOCAL_SCRATCH",
    ):
        monkeypatch.delenv(name, raising=False)


def _gpu_payload(*, devices: list[dict], cpu_count: int = 8, physical_bytes: int = 64 * GIB) -> dict:
    return {
        "schema_version": 2,
        "cpu_count": cpu_count,
        "memory": {"physical_bytes": physical_bytes, "cgroup_limit_bytes": None},
        "scheduler": {},
        "torch": {"available": True, "cuda_available": bool(devices), "devices": devices},
        "scratch": [],
        "cache": [],
    }


def test_required_key_for_backend() -> None:
    assert launcher.required_key_for_backend("openai") == "OPENAI_API_KEY"
    assert launcher.required_key_for_backend("oai") is None
    assert launcher.required_key_for_backend("claude-sonnet-4-5") == "ANTHROPIC_API_KEY"
    assert launcher.required_key_for_backend("anthropic") == "ANTHROPIC_API_KEY"
    assert launcher.required_key_for_backend("mock") is None


@pytest.mark.parametrize(
    "run_args",
    [
        ["--mode", "faar", "--vlm", "claude-sonnet-4-5"],
        ["--mode=colpali", "--vlm=anthropic"],
        ["--ablate", "no_gate", "--vlm", "claude"],
        ["--recovery", "always_vlm", "--vlm", "claude-sonnet-4-5"],
    ],
)
def test_explicit_vlm_arg_resolves_before_env_and_requires_anthropic_key(
    monkeypatch, tmp_path: Path, run_args: list[str]
) -> None:
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    assert launcher.run_requires_vlm(run_args) is True
    with pytest.raises(launcher.LaunchError, match="ANTHROPIC_API_KEY"):
        launcher.validate_required_keys(tmp_path, run_args)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    launcher.validate_required_keys(tmp_path, run_args)


def test_explicit_vlm_openai_overrides_claude_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLM_BACKEND", "claude-sonnet-4-5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    launcher.validate_required_keys(tmp_path, ["--mode", "faar", "--vlm", "openai"])
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, ["--mode", "faar", "--vlm", "openai"])


def test_explicit_vlm_claude_overrides_openai_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLM_BACKEND", "openai")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    with pytest.raises(launcher.LaunchError, match="ANTHROPIC_API_KEY"):
        launcher.validate_required_keys(tmp_path, ["--mode", "faar", "--vlm", "claude-sonnet-4-5"])


def test_explicit_vlm_on_b0_does_not_require_a_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    args = ["--gate", "off", "--recovery", "off", "--out", "results/b0.json", "--vlm", "claude-sonnet-4-5"]
    assert launcher.run_requires_vlm(args) is False
    launcher.validate_required_keys(tmp_path, args)


def test_build_child_command_plain_argv_no_shell(tmp_path: Path) -> None:
    (tmp_path / "scripts/experiments").mkdir(parents=True)
    (tmp_path / "scripts/experiments/run.py").write_text("", encoding="utf-8")
    argv = launcher.build_child_command(tmp_path, ["--gate", "on", "--out", "results/x.json"], "/usr/bin/python3")
    assert argv == ["/usr/bin/python3", str(tmp_path / "scripts/experiments/run.py"), "--gate", "on", "--out", "results/x.json"]
    assert all(isinstance(item, str) for item in argv)


def test_build_child_command_missing_runpy_fails(tmp_path: Path) -> None:
    with pytest.raises(launcher.LaunchError, match="entrypoint is not a file"):
        launcher.build_child_command(tmp_path, [], "/usr/bin/python3")


def test_validate_calibration_launch_requires_lock_for_execute() -> None:
    with pytest.raises(launcher.LaunchError, match="--require-calibration-108"):
        launcher.validate_calibration_launch(
            "scripts/data/prepare_benchmark_assets.py",
            ["--dataset", "ohrbench", "--smoke-doc", launcher.CALIBRATION_108_DOC, "--execute"],
        )


def test_validate_calibration_launch_rejects_wrong_smoke_doc() -> None:
    with pytest.raises(launcher.LaunchError, match="manual/User_Manual_1500S_Classic_EN"):
        launcher.validate_calibration_launch(
            "scripts/data/prepare_benchmark_assets.py",
            [
                "--dataset",
                "ohrbench",
                "--smoke-doc",
                "academic/demo",
                "--require-calibration-108",
                "--execute",
            ],
        )


def test_validate_calibration_launch_accepts_locked_execute() -> None:
    launcher.validate_calibration_launch(
        "scripts/data/prepare_benchmark_assets.py",
        [
            "--dataset",
            "ohrbench",
            "--smoke-doc",
            launcher.CALIBRATION_108_DOC,
            "--require-calibration-108",
            "--execute",
        ],
    )


def test_validate_calibration_launch_ignores_other_entrypoints() -> None:
    launcher.validate_calibration_launch("scripts/experiments/run.py", ["--execute"])


def test_launcher_main_refuses_unlocked_calibration_execute_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entrypoint = tmp_path / "scripts/data/prepare_benchmark_assets.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("raise SystemExit('child must not start')\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "load_preflight", lambda root: (_ for _ in ()).throw(AssertionError("too late")))
    code = launcher.main(
        [
            "--project-root",
            str(tmp_path),
            "--entrypoint",
            "scripts/data/prepare_benchmark_assets.py",
            "--dataset",
            "ohrbench",
            "--smoke-doc",
            launcher.CALIBRATION_108_DOC,
            "--execute",
        ]
    )
    assert code == 2


def test_build_child_command_accepts_repository_local_entrypoint(tmp_path: Path) -> None:
    entrypoint = tmp_path / "scripts/data/prepare_benchmark_assets.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    argv = launcher.build_child_command(
        tmp_path,
        ["--dataset", "ohrbench", "--execute"],
        "/usr/bin/python3",
        "scripts/data/prepare_benchmark_assets.py",
    )
    assert argv == [
        "/usr/bin/python3",
        str(entrypoint),
        "--dataset",
        "ohrbench",
        "--execute",
    ]


def test_build_child_command_rejects_entrypoint_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("", encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="must stay inside"):
        launcher.build_child_command(root, [], "/usr/bin/python3", outside)
    with pytest.raises(launcher.LaunchError, match="must stay inside"):
        launcher.build_child_command(root, [], "/usr/bin/python3", "../outside.py")


def test_default_preflight_path_is_job_namespaced(tmp_path: Path) -> None:
    payload = _gpu_payload(devices=[])
    payload["scheduler"] = {"pbs_job_id": "123.cluster"}
    assert launcher.default_preflight_path(tmp_path, payload) == (
        tmp_path / "results/environment/preflight_123.cluster.json"
    )


def test_check_one_gpu_rejects_zero_and_multiple() -> None:
    with pytest.raises(launcher.LaunchError, match="No CUDA GPU"):
        launcher.check_one_gpu(_gpu_payload(devices=[]), cpu_only=False)
    many = [{"index": 0}, {"index": 1}]
    with pytest.raises(launcher.LaunchError, match="exactly one logical GPU"):
        launcher.check_one_gpu(_gpu_payload(devices=many), cpu_only=False)
    launcher.check_one_gpu(_gpu_payload(devices=[{"index": 0}]), cpu_only=False)


def test_check_one_gpu_cpu_only_skips_zero_gpus() -> None:
    launcher.check_one_gpu(_gpu_payload(devices=[]), cpu_only=True)


def test_check_one_gpu_rejects_unavailable_torch() -> None:
    payload = _gpu_payload(devices=[])
    payload["torch"] = {"available": False, "cuda_available": False, "devices": []}
    with pytest.raises(launcher.LaunchError, match="torch unavailable"):
        launcher.check_one_gpu(payload, cpu_only=False)


def test_derive_limits_from_preflight() -> None:
    payload = _gpu_payload(
        devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}],
        cpu_count=8,
        physical_bytes=64 * GIB,
    )
    assert launcher.derive_max_rss_gb(payload) == 32.0
    assert launcher.derive_min_gpu_free_gb(payload) == 3.2
    assert launcher.derive_gpu_memory_fraction(payload, 8.0) == 0.5
    assert launcher.derive_threads(payload) == 4


def test_derive_limits_respect_scheduler_budget() -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    payload["memory"]["cgroup_limit_bytes"] = 20 * GIB
    assert launcher.derive_max_rss_gb(payload) == 18.0
    payload["scheduler"] = {"slurm_mem_per_node_bytes": 40 * GIB}
    payload["memory"]["cgroup_limit_bytes"] = None
    assert launcher.derive_max_rss_gb(payload) == 36.0


def test_derive_gpu_fraction_is_budget_over_total_and_rejects_oversize(monkeypatch) -> None:
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 4 * GIB}])
    assert launcher.derive_gpu_memory_fraction(payload, 8.0) == 0.5
    assert launcher.derive_gpu_memory_fraction(payload, 1.6) == 0.1
    with pytest.raises(launcher.LaunchError, match="exceeds the visible GPU"):
        launcher.derive_gpu_memory_fraction(payload, 20.0)
    with pytest.raises(launcher.LaunchError, match="invalid GPU budget"):
        launcher.derive_gpu_memory_fraction(payload, float("inf"))


def test_check_gpu_free_vram_requires_budget_plus_reserve() -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 4 * GIB}])
    with pytest.raises(launcher.LaunchError, match="co-tenant reserve"):
        launcher.check_gpu_free_vram(payload, 8.0, 3.2)
    launcher.check_gpu_free_vram(payload, 0.8, 3.2)
    launcher.check_gpu_free_vram(payload, 0.0, 3.2)
    launcher.check_gpu_free_vram(_gpu_payload(devices=[]), 8.0, 3.2)


def test_check_gpu_free_vram_rejects_missing_total() -> None:
    payload = _gpu_payload(devices=[{"index": 0}])
    with pytest.raises(launcher.LaunchError, match="total VRAM"):
        launcher.check_gpu_free_vram(payload, 8.0, 3.2)


def test_parse_gpu_budget_gb_rejects_invalid_values() -> None:
    assert launcher.parse_gpu_budget_gb("8") == 8.0
    assert launcher.parse_gpu_budget_gb("8.5") == 8.5
    for bad in ("", "abc", "-1", "0", "nan", "inf", "  "):
        with pytest.raises(launcher.LaunchError, match="FAAR_GPU_BUDGET_GB"):
            launcher.parse_gpu_budget_gb(bad)


def test_parse_gpu_memory_fraction_rejects_invalid_values() -> None:
    assert launcher.parse_gpu_memory_fraction("0.5") == 0.5
    assert launcher.parse_gpu_memory_fraction("1") == 1.0
    for bad in ("", "abc", "0", "-0.5", "1.5", "nan", "inf"):
        with pytest.raises(launcher.LaunchError, match="FAAR_MAX_GPU_MEMORY_FRACTION"):
            launcher.parse_gpu_memory_fraction(bad)


def test_resolve_gpu_budget_fails_closed_without_budget_or_fraction(monkeypatch) -> None:
    monkeypatch.delenv("FAAR_GPU_BUDGET_GB", raising=False)
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    with pytest.raises(launcher.LaunchError, match="explicit FAAR_GPU_BUDGET_GB"):
        launcher.resolve_gpu_budget(payload)
    assert launcher.resolve_gpu_budget(payload, cpu_only=True) is None


def test_resolve_gpu_budget_rejects_ambiguous_dual_limits(monkeypatch) -> None:
    monkeypatch.setenv("FAAR_GPU_BUDGET_GB", "8")
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.25")
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    with pytest.raises(launcher.LaunchError, match="Set only one"):
        launcher.resolve_gpu_budget(payload)
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION")
    assert launcher.resolve_gpu_budget(payload) == 8.0
    with pytest.raises(launcher.LaunchError, match="exceeds the visible GPU"):
        monkeypatch.setenv("FAAR_GPU_BUDGET_GB", "20")
        launcher.resolve_gpu_budget(payload)


def test_resolve_gpu_budget_accepts_explicit_fraction_for_backward_compat(monkeypatch) -> None:
    monkeypatch.delenv("FAAR_GPU_BUDGET_GB", raising=False)
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.25")
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    assert launcher.resolve_gpu_budget(payload) == 4.0
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "1.5")
    with pytest.raises(launcher.LaunchError, match="FAAR_MAX_GPU_MEMORY_FRACTION"):
        launcher.resolve_gpu_budget(payload)


def test_resolve_min_gpu_free_gb_explicit_overrides_derived(monkeypatch) -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    monkeypatch.delenv("FAAR_MIN_GPU_FREE_GB", raising=False)
    assert launcher.resolve_min_gpu_free_gb(payload) == 3.2
    monkeypatch.setenv("FAAR_MIN_GPU_FREE_GB", "5")
    assert launcher.resolve_min_gpu_free_gb(payload) == 5.0
    monkeypatch.setenv("FAAR_MIN_GPU_FREE_GB", "0")
    with pytest.raises(launcher.LaunchError, match="FAAR_MIN_GPU_FREE_GB"):
        launcher.resolve_min_gpu_free_gb(payload)


def test_derive_threads_falls_back_to_cpu_count() -> None:
    payload = _gpu_payload(devices=[], cpu_count=12)
    assert launcher.derive_threads(payload) == 6
    payload["scheduler"] = {"slurm_cpus_per_task": 2}
    assert launcher.derive_threads(payload) == 1


def test_set_derived_env_only_when_absent(monkeypatch) -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    monkeypatch.delenv("FAAR_MAX_RSS_GB", raising=False)
    monkeypatch.delenv("FAAR_MIN_GPU_FREE_GB", raising=False)
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    derived = launcher.set_derived_env(payload, gpu_budget_gb=8.0)
    assert derived["FAAR_MAX_RSS_GB"] == "32.0"
    assert derived["FAAR_MIN_GPU_FREE_GB"] == "3.2"
    assert derived["FAAR_MAX_GPU_MEMORY_FRACTION"] == "0.5"
    assert derived["OMP_NUM_THREADS"] == "4"
    assert derived["MKL_NUM_THREADS"] == "4"

    monkeypatch.setenv("FAAR_MAX_RSS_GB", "99")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    derived2 = launcher.set_derived_env(payload, gpu_budget_gb=8.0)
    assert "FAAR_MAX_RSS_GB" not in derived2
    assert "OMP_NUM_THREADS" not in derived2
    assert os.environ["FAAR_MAX_RSS_GB"] == "99"
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_set_derived_env_fails_closed_without_budget(monkeypatch) -> None:
    payload = _gpu_payload(devices=[{"total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    with pytest.raises(launcher.LaunchError, match="explicit budget"):
        launcher.set_derived_env(payload)


def test_cpu_only_does_not_derive_gpu_guards(monkeypatch) -> None:
    payload = _gpu_payload(devices=[])
    derived = launcher.set_derived_env(payload, cpu_only=True)
    assert "FAAR_MIN_GPU_FREE_GB" not in derived
    assert "FAAR_MAX_GPU_MEMORY_FRACTION" not in derived
    assert "FAAR_MIN_GPU_FREE_GB" not in os.environ
    assert "FAAR_MAX_GPU_MEMORY_FRACTION" not in os.environ


def test_cpu_only_ignores_invalid_gpu_only_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FAAR_GPU_BUDGET_GB", "invalid")
    monkeypatch.setenv("FAAR_MIN_GPU_FREE_GB", "invalid")
    monkeypatch.setattr(launcher, "load_preflight", lambda root: _gpu_payload(devices=[]))
    monkeypatch.setattr(launcher, "write_preflight_report", lambda payload, path: None)
    monkeypatch.setattr(launcher, "validate_hf_cache", lambda: None)
    monkeypatch.setattr(launcher, "validate_required_keys", lambda root, run_args: None)
    monkeypatch.setattr(
        launcher, "build_child_command", lambda root, run_args, executable, entrypoint: ["child"]
    )
    monkeypatch.setattr(launcher, "run_child", lambda argv, cwd: 0)

    assert launcher.main(["--project-root", str(tmp_path), "--cpu-only"]) == 0


def test_gpu_main_enforces_budget_before_child(monkeypatch, tmp_path: Path) -> None:
    payload = _gpu_payload(
        devices=[
            {
                "index": 0,
                "total_memory_bytes": 16 * GIB,
                "free_memory_bytes": 16 * GIB,
            }
        ]
    )
    monkeypatch.setenv("FAAR_GPU_BUDGET_GB", "8.1")
    monkeypatch.setattr(launcher, "load_preflight", lambda root: payload)
    monkeypatch.setattr(launcher, "write_preflight_report", lambda payload, path: None)
    monkeypatch.setattr(launcher, "validate_hf_cache", lambda: None)
    monkeypatch.setattr(launcher, "validate_required_keys", lambda root, run_args: None)
    monkeypatch.setattr(
        launcher, "build_child_command", lambda root, run_args, executable, entrypoint: ["child"]
    )
    child_env: dict[str, str] = {}

    def capture_child(argv, cwd) -> int:
        child_env.update(os.environ)
        return 0

    monkeypatch.setattr(launcher, "run_child", capture_child)

    assert launcher.main(["--project-root", str(tmp_path)]) == 0
    assert float(child_env["FAAR_MAX_GPU_MEMORY_FRACTION"]) == 8.1 / 16
    assert child_env["FAAR_MIN_GPU_FREE_GB"] == "3.2"


def test_validate_hf_cache_default_created(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache_dir = tmp_path / ".cache" / "huggingface"
    launcher.validate_hf_cache()
    assert cache_dir.is_dir()


def test_validate_hf_cache_explicit_missing_fails(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-hf"
    monkeypatch.setenv("HF_HOME", str(missing))
    with pytest.raises(launcher.LaunchError, match="does not exist"):
        launcher.validate_hf_cache()


def test_validate_hf_cache_unwritable_fails(monkeypatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "hf"
    cache_dir.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache_dir))
    monkeypatch.setattr(launcher.os, "access", lambda path, mode: False)
    with pytest.raises(launcher.LaunchError, match="not writable"):
        launcher.validate_hf_cache()


def test_validate_required_keys_env_and_dotenv(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    launcher.validate_required_keys(tmp_path)

    monkeypatch.delenv("OPENAI_API_KEY")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-from-dotenv\nANTHROPIC_API_KEY=sk-ant-from-dotenv\n", encoding="utf-8")
    launcher.validate_required_keys(tmp_path)

    monkeypatch.setenv("VLM_BACKEND", "claude-sonnet-4-5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").unlink()
    with pytest.raises(launcher.LaunchError, match="ANTHROPIC_API_KEY"):
        launcher.validate_required_keys(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    launcher.validate_required_keys(tmp_path)


def test_empty_key_values_are_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, ["--mode", "faar"])
    monkeypatch.delenv("OPENAI_API_KEY")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, ["--mode", "faar"])


def test_b0_does_not_require_a_vlm_key(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = ["--gate", "off", "--recovery", "off", "--out", "results/b0.json"]
    assert launcher.run_requires_vlm(args) is False
    launcher.validate_required_keys(tmp_path, args)


@pytest.mark.parametrize(
    "args",
    [
        ["--mode", "faar"],
        ["--mode=colpali"],
        ["--ablate", "no_gate"],
        ["--recovery", "always_vlm"],
        ["--recovery=random_type"],
    ],
)
def test_vlm_profiles_require_a_key(monkeypatch, tmp_path: Path, args: list[str]) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert launcher.run_requires_vlm(args) is True
    with pytest.raises(launcher.LaunchError, match="OPENAI_API_KEY"):
        launcher.validate_required_keys(tmp_path, args)


def test_run_child_propagates_exit_code(tmp_path: Path) -> None:
    code = launcher.run_child([sys.executable, "-c", "import sys; sys.exit(7)"], cwd=tmp_path)
    assert code == 7


def test_signal_with_no_child_yet_is_remembered(monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_CHILD", None)
    monkeypatch.setattr(launcher, "_pending_signal", None)
    monkeypatch.setattr(launcher, "_forwarded_signal", None)
    launcher._forward(signal.SIGINT, None)
    assert launcher._pending_signal == signal.SIGINT
    assert launcher._forwarded_signal == signal.SIGINT
    assert launcher._CHILD is None


def test_pending_signal_before_popen_prevents_spawn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_pending_signal", signal.SIGTERM)
    monkeypatch.setattr(launcher, "_forwarded_signal", None)
    spawns = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: spawns.append(args))
    code = launcher.run_child([sys.executable, "-c", "pass"], cwd=tmp_path)
    assert code == 128 + signal.SIGTERM
    assert spawns == []


def test_forwarded_signal_child_exits_zero_yields_128_plus_signal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_CHILD", None)
    monkeypatch.setattr(launcher, "_pending_signal", None)
    monkeypatch.setattr(launcher, "_forwarded_signal", None)
    forwarded = []

    class _FakeProc:
        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            forwarded.append(signum)

        def wait(self) -> int:
            return 0

    def _popen_with_delayed_signal(*args, **kwargs) -> _FakeProc:
        os.kill(os.getpid(), signal.SIGTERM)
        return _FakeProc()

    monkeypatch.setattr(launcher.subprocess, "Popen", _popen_with_delayed_signal)
    code = launcher.run_child([sys.executable, "-c", "pass"], cwd=tmp_path)
    assert code == 128 + signal.SIGTERM
    assert forwarded == [signal.SIGTERM]
    assert launcher._pending_signal == signal.SIGTERM
    assert launcher._forwarded_signal == signal.SIGTERM


def _write_fake_runpy(tmp_path: Path) -> None:
    (tmp_path / "scripts/experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts/experiments/run.py").write_text(
        "\n".join(
            [
                "import os, signal, sys, time",
                "def handle(signum, frame):",
                "    with open('sig_caught', 'w', encoding='utf-8') as fh:",
                "        fh.write(str(signum))",
                "    with open('env_dump.txt', 'w', encoding='utf-8') as fh:",
                "        for name in ('FAAR_MAX_RSS_GB','FAAR_MIN_GPU_FREE_GB','FAAR_MAX_GPU_MEMORY_FRACTION','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES'):",
                "            fh.write(f'{name}={os.environ.get(name, \"UNSET\")}\\n')",
                "    sys.exit(0)",
                "signal.signal(signal.SIGTERM, handle)",
                "print('READY', flush=True)",
                "while True:",
                "    time.sleep(0.1)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _wait_for_ready(out_file: Path, proc: subprocess.Popen, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if out_file.is_file() and "READY" in out_file.read_text(encoding="utf-8", errors="replace"):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.05)
    return False


def test_launcher_forwards_sigterm_and_propagates_exit(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(
            _gpu_payload(devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
        ),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["VLM_BACKEND"] = "openai"
    env["OPENAI_API_KEY"] = "sk-dummy-secret-never-printed"
    out = tmp_path / "out.log"
    err = tmp_path / "err.log"
    proc = subprocess.Popen(
        [sys.executable, str(CLUSTER / "launcher.py"), "--cpu-only", "--project-root", str(tmp_path), "arg-one"],
        cwd=str(tmp_path),
        env=env,
        stdout=open(out, "w", encoding="utf-8"),
        stderr=open(err, "w", encoding="utf-8"),
    )
    try:
        assert _wait_for_ready(out, proc), f"child never became ready; stderr={err.read_text()}"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    assert (tmp_path / "sig_caught").exists()
    assert (tmp_path / "sig_caught").read_text(encoding="utf-8") == str(signal.SIGTERM)

    env_dump = (tmp_path / "env_dump.txt").read_text(encoding="utf-8")
    assert "FAAR_MAX_RSS_GB=32.0" in env_dump
    assert "FAAR_MIN_GPU_FREE_GB=UNSET" in env_dump
    assert "FAAR_MAX_GPU_MEMORY_FRACTION=UNSET" in env_dump
    assert "OMP_NUM_THREADS=4" in env_dump
    assert "MKL_NUM_THREADS=4" in env_dump
    assert "CUDA_VISIBLE_DEVICES=UNSET" in env_dump

    logs = out.read_text(encoding="utf-8") + err.read_text(encoding="utf-8")
    assert "sk-dummy-secret-never-printed" not in logs
    assert "sk-dummy-secret-never-printed" not in env_dump


def test_launcher_rejects_multiple_gpus_end_to_end(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(
            _gpu_payload(
                devices=[
                    {"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB},
                    {"index": 1, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB},
                ]
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["OPENAI_API_KEY"] = "sk-dummy-secret"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [sys.executable, str(CLUSTER / "launcher.py"), "--project-root", str(tmp_path), "--mode", "faar"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "exactly one logical GPU" in result.stderr
    assert "arg-one" not in result.stderr


def test_launcher_reads_gpu_budget_from_dotenv(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    (tmp_path / "scripts/experiments/run.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(
            _gpu_payload(devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])
        ),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    (tmp_path / ".env").write_text("FAAR_GPU_BUDGET_GB=8\n", encoding="utf-8")
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env.pop("FAAR_GPU_BUDGET_GB", None)
    env.pop("FAAR_MAX_GPU_MEMORY_FRACTION", None)
    result = subprocess.run(
        [sys.executable, str(CLUSTER / "launcher.py"), "--project-root", str(tmp_path), "--gate", "off"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "FAAR_GPU_BUDGET_GB=8" in result.stdout


def test_launcher_missing_key_fails_before_child(tmp_path: Path) -> None:
    _write_fake_runpy(tmp_path)
    preflight_file = tmp_path / "preflight.json"
    preflight_file.write_text(
        __import__("json").dumps(_gpu_payload(devices=[{"index": 0, "total_memory_bytes": 16 * GIB, "free_memory_bytes": 16 * GIB}])),
        encoding="utf-8",
    )
    (tmp_path / "hf").mkdir()
    env = os.environ.copy()
    env["FAAR_PREFLIGHT_JSON"] = str(preflight_file)
    env["HF_HOME"] = str(tmp_path / "hf")
    env["FAAR_GPU_BUDGET_GB"] = "8"
    env.pop("OPENAI_API_KEY", None)
    env.pop("VLM_BACKEND", None)
    result = subprocess.run(
        [sys.executable, str(CLUSTER / "launcher.py"), "--project-root", str(tmp_path), "--mode", "faar"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "OPENAI_API_KEY" in result.stderr
    assert not (tmp_path / "sig_caught").exists()


def _write_fake_faar_demo(root: Path) -> Path:
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake = venv_bin / "faar-demo"
    fake.write_text("#!/bin/sh\nfor arg in \"$@\"; do printf '%s\\n' \"$arg\"; done\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_demo_script_path_resolution_portable(tmp_path: Path) -> None:
    _write_fake_faar_demo(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    demo = scripts_dir / "run_phase1_demo.sh"
    demo.write_text((Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_demo.sh").read_text(encoding="utf-8"), encoding="utf-8")
    demo.chmod(0o755)
    result = subprocess.run(
        ["sh", str(demo), "446d159e", "--seed", "7"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert "run-example" in lines
    assert "--project-root" in lines
    assert str(tmp_path) in lines
    assert "--example-id" in lines
    assert "446d159e" in lines
    assert "--seed" in lines
    assert "7" in lines


def test_demo_script_usage_without_args(tmp_path: Path) -> None:
    _write_fake_faar_demo(tmp_path)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    demo = scripts_dir / "run_phase1_demo.sh"
    demo.write_text((Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_demo.sh").read_text(encoding="utf-8"), encoding="utf-8")
    demo.chmod(0o755)
    result = subprocess.run(["sh", str(demo)], cwd=str(tmp_path), capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_launcher_syntax_compiles() -> None:
    import py_compile

    py_compile.compile(str(CLUSTER / "launcher.py"), doraise=True)
    py_compile.compile(str(CLUSTER / "preflight.py"), doraise=True)


@pytest.mark.parametrize(
    "script",
    [
        Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_demo.sh",
        Path(__file__).resolve().parents[1] / "cluster" / "templates" / "slurm_one_gpu.sbatch",
        Path(__file__).resolve().parents[1] / "cluster" / "templates" / "pbs_one_gpu.pbs",
    ],
)
def test_shell_template_syntax(script: Path) -> None:
    if "run_phase1_demo.sh" in script.name:
        checker = ["sh", "-n"]
    else:
        checker = ["bash", "-n"]
    result = subprocess.run(checker + [str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_templates_never_override_cuda_visible_devices() -> None:
    for template in (Path(__file__).resolve().parents[1] / "cluster" / "templates" / "slurm_one_gpu.sbatch",
                     Path(__file__).resolve().parents[1] / "cluster" / "templates" / "pbs_one_gpu.pbs"):
        text = template.read_text(encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES=" not in text
        assert "--gpus=1" in text or "ngpus=1" in text
