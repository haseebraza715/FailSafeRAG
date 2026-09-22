from __future__ import annotations

import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

PREFLIGHT_PATH = Path(__file__).resolve().parents[1] / "cluster/preflight.py"
PREFLIGHT_SPEC = spec_from_file_location("preflight", PREFLIGHT_PATH)
assert PREFLIGHT_SPEC and PREFLIGHT_SPEC.loader
preflight = module_from_spec(PREFLIGHT_SPEC)
sys.modules["preflight"] = preflight
PREFLIGHT_SPEC.loader.exec_module(preflight)


GIB = 1024**3


def test_parse_memory_bytes_variants() -> None:
    assert preflight._parse_memory_bytes("64G") == 64 * GIB
    assert preflight._parse_memory_bytes("12288M") == 12 * GIB
    assert preflight._parse_memory_bytes("20gb") == 20 * GIB
    assert preflight._parse_memory_bytes("4096") == 4096 * 1024**2
    assert preflight._parse_memory_bytes("512MiB") == 512 * 1024**2
    assert preflight._parse_memory_bytes("1T") == GIB * 1024
    for junk in (None, "", "max", "unlimited", "0", "abc", "12X"):
        assert preflight._parse_memory_bytes(junk) is None


def test_parse_walltime_seconds_variants() -> None:
    assert preflight._parse_walltime_seconds("2-01:30:00") == 2 * 86400 + 5400
    assert preflight._parse_walltime_seconds("08:00:00") == 8 * 3600
    assert preflight._parse_walltime_seconds("30:00") == 1800
    assert preflight._parse_walltime_seconds("12345") == 12345 * 60
    assert preflight._parse_walltime_seconds(None) is None
    assert preflight._parse_walltime_seconds("") is None
    assert preflight._parse_walltime_seconds("not-a-time") is None


def test_parse_gpu_count_comma_ids_and_typed_counts() -> None:
    assert preflight._parse_gpu_count("0,1") == 2
    assert preflight._parse_gpu_count("0,1,2") == 3
    assert preflight._parse_gpu_count("gpu:a100:2") == 2
    assert preflight._parse_gpu_count("gpu:2") == 2
    assert preflight._parse_gpu_count("2") == 2
    assert preflight._parse_gpu_count("1") == 1
    assert preflight._parse_gpu_count("gpu:a100:2,3") == 3
    assert preflight._parse_gpu_count("0", id_list=True) == 1
    assert preflight._parse_gpu_count(None) is None
    assert preflight._parse_gpu_count("") is None
    assert preflight._parse_gpu_count("0") is None


def test_scheduler_slurm_metadata_parsed(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    monkeypatch.setenv("SLURM_JOB_NAME", "faar-test")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    monkeypatch.setenv("SLURM_JOB_QOS", "normal")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "16")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "64G")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096M")
    monkeypatch.setenv("SLURM_GPUS", "1")
    monkeypatch.setenv("SLURM_TIME_LIMIT", "08:00:00")
    scheduler = preflight._scheduler()
    assert scheduler["scheduler_detected"] == "slurm"
    assert scheduler["slurm_job_id"] == "4242"
    assert scheduler["slurm_partition"] == "gpu"
    assert scheduler["slurm_cpus_on_node"] == 16
    assert scheduler["slurm_cpus_per_task"] == 8
    assert scheduler["slurm_mem_per_node_bytes"] == 64 * GIB
    assert scheduler["slurm_mem_per_cpu_bytes"] == 4096 * 1024**2
    assert scheduler["slurm_gpus_requested"] == 1
    assert scheduler["slurm_time_limit_seconds"] == 8 * 3600


def test_scheduler_pbs_metadata_parsed(monkeypatch) -> None:
    monkeypatch.setenv("PBS_JOBID", "9876.cluster")
    monkeypatch.setenv("PBS_JOBNAME", "faar-pbs")
    monkeypatch.setenv("PBS_O_QUEUE", "gpuq")
    monkeypatch.setenv("PBS_NUM_NODES", "1")
    monkeypatch.setenv("PBS_NUM_PPN", "8")
    monkeypatch.setenv("PBS_NCPUS", "8")
    monkeypatch.setenv("PBS_RESC_MEM", "32768mb")
    monkeypatch.setenv("PBS_GPUS", "1")
    monkeypatch.setenv("PBS_WALLTIME", "04:00:00")
    scheduler = preflight._scheduler()
    assert scheduler["scheduler_detected"] == "pbs"
    assert scheduler["pbs_job_id"] == "9876.cluster"
    assert scheduler["pbs_queue"] == "gpuq"
    assert scheduler["pbs_num_nodes"] == 1
    assert scheduler["pbs_num_ppn"] == 8
    assert scheduler["pbs_ncpus"] == 8
    assert scheduler["pbs_mem"] == "32768mb"
    assert scheduler["pbs_mem_bytes"] == 32768 * 1024**2
    assert scheduler["pbs_gpus_requested"] == 1
    assert scheduler["pbs_walltime_seconds"] == 4 * 3600


def test_scheduler_pbs_mem_alias_precedence(monkeypatch) -> None:
    monkeypatch.setenv("PBS_MEMORY", "16gb")
    assert preflight._pbs_mem_raw() == "16gb"
    monkeypatch.setenv("PBS_RESC_MEM", "32gb")
    monkeypatch.delenv("PBS_MEMORY")
    assert preflight._pbs_mem_raw() == "32gb"
    monkeypatch.setenv("PBS_MEM", "64gb")
    monkeypatch.delenv("PBS_RESC_MEM")
    assert preflight._pbs_mem_raw() == "64gb"


def test_redact_masks_secret_shaped_strings() -> None:
    payload = {
        "host": "worker-01",
        "scheduler": {"slurm_job_name": "job sk-abc12345secret"},
        "credential": "OPENAI_API_KEY=sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "aws": "AKIAIOSFODNN7EXAMPLE",
        "plain": "fine",
    }
    masked = preflight.redact(payload)
    assert masked["scheduler"]["slurm_job_name"] == "job <redacted>"
    assert masked["credential"] == "<redacted>"
    assert masked["aws"] == "<redacted>"
    assert masked["host"] == "worker-01"
    assert masked["plain"] == "fine"


def test_collect_render_never_leaks_secrets(monkeypatch, tmp_path: Path) -> None:
    secret = "sk-super-secret-test-value-abcdef1234567890"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-another-secret-value-12345")
    monkeypatch.setenv("SLURM_JOB_NAME", f"job-with-{secret}")
    payload = preflight.collect(tmp_path)
    rendered = preflight.render(payload)
    assert secret not in rendered
    assert "sk-ant-another-secret-value-12345" not in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "ANTHROPIC_API_KEY" not in rendered


def test_collect_signature_and_top_level_keys(tmp_path: Path) -> None:
    payload = preflight.collect(tmp_path)
    for key in (
        "schema_version",
        "collected_at_utc",
        "host",
        "python",
        "platform",
        "cpu_count",
        "memory",
        "disk",
        "scheduler",
        "cuda_visible_devices",
        "torch",
        "nvidia_smi",
        "ulimit",
        "scratch",
        "cache",
        "relevant_environment",
    ):
        assert key in payload
    assert "path" in payload["disk"]
    assert "cgroup_limit_bytes" in payload["memory"]


def test_cgroup_memory_structure() -> None:
    memory = preflight._memory_bytes()
    assert "physical_bytes" in memory
    for key in ("cgroup_limit_bytes", "cgroup_current_bytes", "cgroup_peak_bytes", "cgroup_controller"):
        assert key in memory


def test_unbounded_cgroup_limit_is_not_used_as_a_budget(monkeypatch) -> None:
    monkeypatch.setattr(preflight.os, "sysconf", lambda name: 1024 if name == "SC_PAGE_SIZE" else 1024)
    monkeypatch.setattr(
        preflight,
        "_cgroup_memory",
        lambda: {"limit_bytes": 2**63, "current_bytes": 1, "peak_bytes": 2, "controller": "v1"},
    )
    assert preflight._memory_bytes()["cgroup_limit_bytes"] is None


def test_ulimit_structure() -> None:
    limits = preflight._ulimit()
    assert "available" in limits
    assert "raw" not in limits


def test_ulimit_reports_unavailable_without_resource_module(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "resource", None)
    assert preflight._ulimit() == {"available": False}


def test_scratch_cache_writable_free_checks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    payload = preflight.collect(tmp_path)
    scratch = [entry for entry in payload["scratch"] if entry["env"] == "SCRATCH"]
    cache = [entry for entry in payload["cache"] if entry["env"] == "HF_HOME"]
    assert scratch and scratch[0]["exists"] is True
    assert scratch[0]["writable"] is True
    assert isinstance(scratch[0]["free_bytes"], int)
    assert cache and cache[0]["path"] == str(tmp_path)
    assert cache[0]["writable"] is True


def test_scratch_cache_missing_path_reported_not_writable(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("TMPDIR", str(missing))
    payload = preflight.collect(tmp_path)
    entry = next(e for e in payload["scratch"] if e["env"] == "TMPDIR")
    assert entry["exists"] is False
    assert entry["writable"] is False
    assert entry["free_bytes"] is None


def test_atomic_write_unique_and_durable(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "preflight.json"
    preflight._atomic_write_text(out, "{}\n")
    assert out.read_text(encoding="utf-8") == "{}\n"
    leftovers = list(tmp_path.rglob(".*.tmp"))
    assert leftovers == []


def test_main_writes_out_and_prints_payload(monkeypatch, tmp_path: Path) -> None:
    payload = {"schema_version": 2, "host": "fake", "ok": True}
    monkeypatch.setattr(preflight, "collect", lambda path, **kwargs: payload)
    out = tmp_path / "preflight.json"
    monkeypatch.setattr(sys, "argv", ["preflight.py", "--path", str(tmp_path), "--out", str(out)])
    assert preflight.main() == 0
    rendered = json_loads(out)
    assert rendered["host"] == "fake"
    assert "ok" in rendered


def json_loads(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def test_atomic_write_preserves_existing_on_failure(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "preflight.json"
    out.write_text("OLD\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        preflight._atomic_write_text(out, "NEW\n")
    assert out.read_text(encoding="utf-8") == "OLD\n"
    assert list(tmp_path.glob("*.tmp")) == []
