from __future__ import annotations

import hashlib
import json
import subprocess
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


def build_project(tmp_path: Path) -> Path:
    project = tmp_path
    (project / "OHR-Bench/data/retrieval_base/gt").mkdir(parents=True)
    (project / "OHR-Bench/data/qas_v2.json").write_text(json.dumps([{"ID": "e1", "doc_name": "a"}]))
    (project / "config/datasets").mkdir(parents=True)
    (project / "config/datasets/ohr_split.json").write_text(json.dumps({"splits": {"val": ["e1"]}}))
    (project / "data/ohr_bench_raw").mkdir(parents=True)
    (project / "data/ohr_bench_raw/pdfs.zip").write_bytes(b"PK")
    (project / "config").mkdir(exist_ok=True)
    (project / "config/split_checksums.json").write_text(
        json.dumps(
            {
                "split_sha256": hashlib.sha256((project / "config/datasets/ohr_split.json").read_bytes()).hexdigest(),
                "qas_v2_sha256": hashlib.sha256((project / "OHR-Bench/data/qas_v2.json").read_bytes()).hexdigest(),
            }
        )
    )
    (project / "config/model_revisions.json").write_text(
        json.dumps(
            {
                "models": {
                    "got_ocr": {"repository": "stepfun-ai/GOT-OCR-2.0-hf", "revision": "d" * 40},
                    "docling": {
                        "repository": "docling-project/docling-models",
                        "revision": "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4",
                    },
                }
            }
        )
    )
    return project


def gpu_payload(devices: list[dict] | None = None, scheduler: dict | None = None) -> dict:
    return {
        "torch": {
            "available": True,
            "cuda_available": True,
            "devices": (
                devices
                if devices is not None
                else [
                    {
                        "index": 0,
                        "name": "TestGPU",
                        "total_memory_bytes": 16 * GIB,
                        "free_memory_bytes": 16 * GIB,
                    }
                ]
            ),
        },
        "scheduler": scheduler or {"slurm_gpus_requested": 1},
        "memory": {"cgroup_limit_bytes": 32 * GIB},
    }


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FAAR_GPU_BUDGET_GB",
        "FAAR_MAX_GPU_MEMORY_FRACTION",
        "FAAR_MIN_GPU_FREE_GB",
        "FAAR_MAX_RSS_GB",
        "FAAR_OUT_ROOT",
        "FAAR_SCRATCH",
        "HF_HOME",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENAI_API_KEY",
        "VLM_BACKEND",
        "FAAR_DOCUMENT_INVENTORY",
        "FAAR_PDF_ROOT",
        "FAAR_PDF_ZIP",
    ):
        monkeypatch.delenv(name, raising=False)
    original_reachable = preflight._hf_repo_reachable
    monkeypatch.setattr(preflight, "_hf_repo_reachable", lambda repo, rev: (True, "HTTP 200"))
    monkeypatch.setattr(preflight, "_hf_repo_reachable_original", original_reachable, raising=False)


def test_run_checks_passes_with_no_warnings(
    tmp_path: Path, clean_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project(tmp_path)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    out_root = tmp_path / "out"
    out_root.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("FAAR_OUT_ROOT", str(out_root))
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("FAAR_GPU_BUDGET_GB", "8")
    monkeypatch.setenv("FAAR_MIN_GPU_FREE_GB", "3")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890abcdef")
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda path: type("usage", (), {"total": 100 * GIB, "used": 20 * GIB, "free": 80 * GIB})(),
    )
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=True)
    assert report["exit_code"] == 0, report
    assert report["summary"]["failed"] == 0
    assert report["summary"]["warnings"] == 0


def test_preflight_check_reads_pdf_zip_from_dotenv(
    tmp_path: Path, clean_env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = build_project(tmp_path)
    (project / "data/ohr_bench_raw/pdfs.zip").unlink()
    custom = project / "elsewhere" / "pdfs.zip"
    custom.parent.mkdir()
    custom.write_bytes(b"PK")
    (project / ".env").write_text(f"FAAR_PDF_ZIP={custom}\n", encoding="utf-8")
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/usr/bin/{name}")
    code = preflight.main(["--check", "--no-cuda", "--dry-run", "--project-root", str(project)])
    assert code != 1
    report = json.loads(capsys.readouterr().out)
    dataset = next(check for check in report["checks"] if check["name"] == "dataset_paths")
    assert dataset["status"] == "pass"
    assert report["environment"].get("FAAR_PDF_ZIP") == str(custom)


def test_run_checks_missing_cuda_is_blocking(tmp_path: Path, clean_env) -> None:
    project = build_project(tmp_path)
    report = preflight.run_checks(gpu_payload(devices=[]), project_root=project, path=project, cuda=True)
    assert report["exit_code"] == 1
    names = {check["name"] for check in report["checks"] if check["status"] == "fail"}
    assert "cuda_visibility" in names


def test_run_checks_no_cuda_skips_gpu_checks(tmp_path: Path, clean_env) -> None:
    project = build_project(tmp_path)
    report = preflight.run_checks(gpu_payload(devices=[]), project_root=project, path=project, cuda=False)
    assert report["exit_code"] < 2 or report["exit_code"] == 2  # warnings allowed, never blocking
    cuda_checks = [check for check in report["checks"] if check["name"] in {"cuda_visibility", "gpu_allocation_match"}]
    assert all(check["status"] == "skip" for check in cuda_checks)
    assert report["summary"]["failed"] == 0


def test_run_checks_allocation_mismatch_is_blocking(tmp_path: Path, clean_env) -> None:
    project = build_project(tmp_path)
    payload = gpu_payload(scheduler={"slurm_gpus_requested": 2})
    report = preflight.run_checks(payload, project_root=project, path=project, cuda=True)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "gpu_allocation_match"]
    assert matching[0]["status"] == "fail"


def test_run_checks_dual_gpu_limits_are_blocking(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_GPU_BUDGET_GB", "8")
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.5")
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=True)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "gpu_budget_consistency"]
    assert matching[0]["status"] == "fail"


def test_run_checks_tampered_split_is_blocking(tmp_path: Path, clean_env) -> None:
    project = build_project(tmp_path)
    with (project / "config/datasets/ohr_split.json").open("a") as handle:
        handle.write(" ")
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "locked_split_checksums"]
    assert matching[0]["status"] == "fail"


def test_run_checks_missing_dataset_is_blocking(tmp_path: Path, clean_env) -> None:
    project = build_project(tmp_path)
    (project / "config/datasets/ohr_split.json").unlink()
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "fail"


def test_run_checks_nonexistent_pdf_root_is_blocking(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_PDF_ROOT", str(tmp_path / "no-such-dir"))
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "fail"
    assert "FAAR_PDF_ROOT" in matching[0]["detail"]


def test_run_checks_missing_pdf_zip_env_is_blocking(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_PDF_ZIP", str(tmp_path / "no-such.zip"))
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "fail"


def test_run_checks_valid_pdf_root_and_zip_env_pass(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    pdf_root = tmp_path / "pdfs-here"
    pdf_root.mkdir()
    monkeypatch.setenv("FAAR_PDF_ROOT", str(pdf_root))
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "pass"
    monkeypatch.setenv("FAAR_PDF_ZIP", str(project / "data/ohr_bench_raw/pdfs.zip"))
    monkeypatch.delenv("FAAR_PDF_ROOT")
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "pass"


def test_hf_check_sends_bearer_token_without_leaking_it(
    tmp_path: Path, clean_env, monkeypatch
) -> None:
    project = build_project(tmp_path)
    secret = "hf-secret-token-value-123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    captured: list[dict] = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        captured.append({"headers": dict(request.headers), "url": request.full_url})
        return FakeResponse()

    monkeypatch.setattr(preflight.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(preflight, "_hf_repo_reachable", preflight._hf_repo_reachable_original)
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert captured
    assert captured[0]["headers"].get("Authorization") == f"Bearer {secret}"
    rendered = preflight.render(report)
    assert secret not in rendered


def test_run_checks_unwritable_cache_is_blocking(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    cache = tmp_path / "readonly-cache"
    cache.mkdir()
    cache.chmod(0o555)
    monkeypatch.setenv("HF_HOME", str(cache))
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "model_cache_paths"]
    assert matching[0]["status"] == "fail"


def test_run_checks_missing_ram_limit_warns(tmp_path: Path, clean_env) -> None:
    project = build_project(tmp_path)
    payload = gpu_payload()
    payload["memory"] = {"cgroup_limit_bytes": None}
    payload["scheduler"] = {}
    report = preflight.run_checks(payload, project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 2
    matching = [check for check in report["checks"] if check["name"] == "ram_limit"]
    assert matching[0]["status"] == "warn"


def test_report_never_contains_secret_values(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    secret = "sk-very-secret-value-abcdef0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    rendered = preflight.render(report)
    assert secret not in rendered
    for check in report["checks"]:
        assert "api_keys_by_stage" not in json.dumps(check.get("measurement") or {})


def test_hf_access_uses_cdn_head_without_download(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        preflight,
        "_hf_repo_reachable",
        lambda repo, rev: calls.append((repo, rev)) or (True, "HTTP 200"),
    )
    preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert calls == [
        ("stepfun-ai/GOT-OCR-2.0-hf", "d" * 40),
        ("docling-project/docling-models", "2bdc831fd1edeb61e6d0dfc8ae7596b0c30bdff4"),
    ]


def test_cli_check_dry_run_writes_nothing(tmp_path: Path, capsys) -> None:
    project = build_project(tmp_path)
    report_path = tmp_path / "preflight_check.json"
    code = preflight.main(
        [
            "--check",
            "--no-cuda",
            "--dry-run",
            "--project-root",
            str(project),
            "--path",
            str(project),
            "--out",
            str(report_path),
        ]
    )
    assert code in (0, 2)
    assert not report_path.exists()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["mode"] == "check"


def test_cli_check_writes_report_atomically(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    report_path = tmp_path / "preflight_check.json"
    code = preflight.main(
        ["--check", "--no-cuda", "--project-root", str(project), "--path", str(project), "--out", str(report_path)]
    )
    assert code in (0, 2)
    payload = json.loads(report_path.read_text())
    assert payload["exit_code"] == code
    assert not list(tmp_path.glob(".preflight_check.json.*.tmp"))


def test_cli_check_exit_code_matches_blocking_failure(tmp_path: Path) -> None:
    project = build_project(tmp_path)
    (project / "config/datasets/ohr_split.json").unlink()
    code = preflight.main(
        ["--check", "--no-cuda", "--dry-run", "--project-root", str(project), "--path", str(project)]
    )
    assert code == 1


def test_run_checks_explicit_missing_zip_fails_even_with_valid_root(
    tmp_path: Path, clean_env, monkeypatch
) -> None:
    project = build_project(tmp_path)
    pdf_root = tmp_path / "valid-root"
    pdf_root.mkdir()
    monkeypatch.setenv("FAAR_PDF_ROOT", str(pdf_root))
    monkeypatch.setenv("FAAR_PDF_ZIP", str(tmp_path / "missing.zip"))
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "fail"
    assert "FAAR_PDF_ZIP" in matching[0]["detail"]


def test_hf_checks_can_be_disabled_offline(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    monkeypatch.setenv("FAAR_PREFLIGHT_OFFLINE", "1")
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    hf_checks = [check for check in report["checks"] if check["name"] == "hf_model_access"]
    assert hf_checks and all(check["status"] == "skip" for check in hf_checks)


def test_cli_check_login_node_safe_without_cuda() -> None:
    completed = subprocess.run(
        [sys.executable, str(PREFLIGHT_PATH), "--check", "--no-cuda", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**__import__("os").environ, "FAAR_PREFLIGHT_OFFLINE": "1"},
    )
    assert completed.returncode in (0, 2)
    assert "cuda_visibility" in completed.stdout


def test_run_checks_inventory_file_is_blocking(tmp_path: Path, clean_env, monkeypatch) -> None:
    project = build_project(tmp_path)
    inventory = project / "OHR-Bench/data/retrieval_base/gt"
    inventory.rmdir()
    inventory.write_text("not a directory")
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    assert report["exit_code"] == 1
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "fail"
    assert "document inventory is not a directory" in matching[0]["detail"]


def test_run_checks_relative_inventory_resolves_against_project_root(
    tmp_path: Path, clean_env, monkeypatch
) -> None:
    project = build_project(tmp_path)
    outsider = tmp_path / "elsewhere"
    outsider.mkdir()
    monkeypatch.chdir(outsider)
    monkeypatch.setenv("FAAR_DOCUMENT_INVENTORY", "OHR-Bench/data/retrieval_base/gt")
    monkeypatch.setenv("FAAR_PDF_ROOT", "data/ohr_bench_raw")
    (project / "data/ohr_bench_raw").mkdir(parents=True, exist_ok=True)
    report = preflight.run_checks(gpu_payload(), project_root=project, path=project, cuda=False)
    matching = [check for check in report["checks"] if check["name"] == "dataset_paths"]
    assert matching[0]["status"] == "pass", matching[0]
