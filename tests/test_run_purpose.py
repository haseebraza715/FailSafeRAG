"""Run purpose (kind) and subset provenance of the answer-model run driver.

Every test uses the fake provider, through the factory seam for live mode. Nothing here sends a request.

Ways the driver could fail, written before the code. Each test below covers one.

Kind
  K1. A live run is recorded as a development pilot whatever it was for, so a transport check looks like a baseline.
  K2. A live run starts with no stated kind, or a fake run claims to be a development pilot.
  K3. The kind in run_config.json, the summary, the status output, the score summary and the registry disagree.
Subset provenance
  M1. A subset, a reordered or reworded selection or a replaced question counts as the frozen pilot.
  M2. Canonical is inferred from the pilot id or the question count, or from the manifest's path.
  M3. A development pilot starts on a manifest that is not the frozen one.
  M4. valid_baseline is true for an engineering check or for a non-canonical manifest.
Records written before the fields existed
  L1. A recorded run gains new keys in its exports.
"""

from __future__ import annotations

import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import pytest
import run_pilot_live as cli
from test_live_runner import (
    LIVE_READY_ENV,
    PILOT_ID,
    QUESTIONS,
    Project,
    Rig,
    default_project,
    summary_of,
    tripwire_clients,
    valid_live_config,
)

from faar import live_runner as lr
from faar.live_contract import RUN_KIND_DEVELOPMENT, RUN_KIND_ENGINEERING
from faar.pilot_runner import RunnerRefusal

ROOT = Path(__file__).resolve().parents[1]
_spec = spec_from_file_location("registry", ROOT / "scripts/experiments/registry.py")
registry = module_from_spec(_spec)
_spec.loader.exec_module(registry)



@pytest.fixture
def project(tmp_path: Path) -> Project:
    return default_project(tmp_path / "project")

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def live_config() -> lr.ProviderConfig:
    return lr.parse_provider_config(valid_live_config())


def live(
    project: Project,
    name: str = "live-a",
    *,
    run_kind: str | None,
    manifest: Path | None = None,
    rig: Rig | None = None,
) -> tuple[lr.LiveResult, Rig]:
    rig = rig or Rig()
    config = live_config()
    opts = lr.RunOptions(
        project_root=project.root,
        run_dir=project.root / "results" / "development" / name,
        mode=lr.MODE_LIVE,
        pilot_id=PILOT_ID,
        safety_ceiling=100.0,
        config=config,
        runtime_manifest_path=manifest,
        run_kind=run_kind,
    )
    result = lr.execute_run(
        opts,
        provider_factory=lambda ctx: lr.build_fake_provider(rig.steps, rig.default, config, ctx),
        descriptor=rig.descriptor,
        sleep=lambda s: None,
        environ=dict(LIVE_READY_ENV),
    )
    return result, rig


def live_dir(project: Project, name: str = "live-a") -> Path:
    return project.root / "results" / "development" / name


def frozen(project: Project) -> dict[str, Any]:
    return json.loads(project.runtime_manifest.read_text(encoding="utf-8"))


def write_variant(project: Project, name: str, payload: dict[str, Any] | str) -> Path:
    path = project.root / "derived" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=1), encoding="utf-8")
    return path


def variants(project: Project) -> dict[str, Path]:
    """Manifests that differ from the frozen one, most of them with the same pilot id and question count."""
    base = frozen(project)

    def altered(mutate: Any) -> dict[str, Any]:
        payload = json.loads(json.dumps(base))
        mutate(payload)
        return payload

    def swap(p: dict[str, Any]) -> None:
        p["questions"][0], p["questions"][1] = p["questions"][1], p["questions"][0]

    def reword(p: dict[str, Any]) -> None:
        p["questions"][0]["question"] += " Answer briefly."

    return {
        "reordered": write_variant(project, "reordered", altered(swap)),
        "reworded": write_variant(project, "reworded", altered(reword)),
        "subset": write_variant(project, "subset", altered(lambda p: p.update(questions=p["questions"][:3]))),
        "same_content_other_bytes": write_variant(project, "reformatted", json.dumps(base)),
    }


# ---------------------------------------------------------------------------
# K1 to K3: kind
# ---------------------------------------------------------------------------


def test_a_fake_run_is_an_engineering_check_in_every_record(project: Project) -> None:
    from test_live_runner import go

    result = go(project, Rig())
    run_dir = project.run_dir("run-a")
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["identity"]["run_kind"] == config["kind"] == RUN_KIND_ENGINEERING
    summary = summary_of(run_dir)
    assert summary["run_kind"] == RUN_KIND_ENGINEERING and result.summary["run_kind"] == RUN_KIND_ENGINEERING
    status = lr.run_status(run_dir)
    assert status["run_kind"] == RUN_KIND_ENGINEERING and status["kind"] == RUN_KIND_ENGINEERING


def test_a_fake_run_cannot_claim_to_be_a_development_pilot(project: Project) -> None:
    from test_live_runner import options

    rig = Rig()
    with pytest.raises(RunnerRefusal, match="engineering_check"):
        lr.execute_run(
            options(project, run_kind=RUN_KIND_DEVELOPMENT), provider_factory=rig.factory, descriptor=rig.descriptor, environ={}
        )
    assert rig.factory_calls == 0 and not project.run_dir("run-a").exists()


def test_a_live_run_needs_a_stated_kind_and_creates_nothing(project: Project) -> None:
    with pytest.raises(RunnerRefusal, match="run-kind"):
        live(project, run_kind=None)
    assert not live_dir(project).exists()
    with pytest.raises(RunnerRefusal, match="run-kind"):
        live(project, run_kind="scientific_evaluation")
    assert not live_dir(project).exists()


def test_a_live_engineering_check_is_labelled_a_transport_and_accounting_check(project: Project) -> None:
    result, _ = live(project, run_kind=RUN_KIND_ENGINEERING)
    config = json.loads((live_dir(project) / "run_config.json").read_text())
    assert config["kind"] == RUN_KIND_ENGINEERING == config["identity"]["run_kind"]
    label = " ".join(config["label"].split())
    assert "transport" in label and "accounting" in label and "not a baseline" in label
    assert result.summary["kind"] == RUN_KIND_ENGINEERING and result.summary["label"] == config["label"]


def test_a_live_development_pilot_keeps_its_label_and_kind(project: Project) -> None:
    live(project, run_kind=RUN_KIND_DEVELOPMENT)
    config = json.loads((live_dir(project) / "run_config.json").read_text())
    assert config["kind"] == RUN_KIND_DEVELOPMENT == config["identity"]["run_kind"]
    assert config["label"] == lr.LABEL_LIVE


def test_the_kind_is_identity_so_the_other_kind_cannot_resume_the_directory(project: Project) -> None:
    live(project, run_kind=RUN_KIND_ENGINEERING)
    with pytest.raises(RunnerRefusal, match="run_kind"):
        live(project, run_kind=RUN_KIND_DEVELOPMENT)


def test_status_and_the_run_line_name_the_kind_and_the_manifest(project: Project) -> None:
    result, _ = live(project, run_kind=RUN_KIND_ENGINEERING)
    status = lr.run_status(live_dir(project))
    assert status["run_kind"] == RUN_KIND_ENGINEERING and status["canonical_manifest"] is True
    text = lr.format_status(status)
    assert "engineering_check" in text.splitlines()[0]
    assert "canonical" in text


# ---------------------------------------------------------------------------
# M1 to M4: subset provenance
# ---------------------------------------------------------------------------


def test_the_identity_records_where_the_manifest_comes_from(project: Project) -> None:
    live(project, run_kind=RUN_KIND_DEVELOPMENT)
    identity = json.loads((live_dir(project) / "run_config.json").read_text())["identity"]
    manifest_sha = sha(project.runtime_manifest.read_bytes())
    ids = [q["question_id"] for q in frozen(project)["questions"]]
    assert identity["pilot_manifest"] == {
        "parent_pilot_id": PILOT_ID,
        "frozen_runtime_manifest_sha256": manifest_sha,
        "run_runtime_manifest_sha256": manifest_sha,
        "canonical": True,
        "question_count": len(QUESTIONS),
        "question_ids_sha256": sha("\n".join(ids).encode()),
    }
    assert identity["runtime_manifest_sha256"] == manifest_sha


@pytest.mark.parametrize("name", ["reworded", "same_content_other_bytes"])
def test_an_altered_selection_is_not_canonical_and_is_an_engineering_check_with_both_blockers(project: Project, name: str) -> None:
    manifest = variants(project)[name]
    result, _ = live(project, run_kind=RUN_KIND_ENGINEERING, manifest=manifest)
    block = json.loads((live_dir(project) / "run_config.json").read_text())["identity"]["pilot_manifest"]
    assert block["canonical"] is False
    assert block["frozen_runtime_manifest_sha256"] == sha(project.runtime_manifest.read_bytes())
    assert block["run_runtime_manifest_sha256"] == sha(manifest.read_bytes()) != block["frozen_runtime_manifest_sha256"]
    summary = result.summary
    assert summary["run_kind"] == RUN_KIND_ENGINEERING and summary["canonical_manifest"] is False
    assert summary["run_state"] == "complete" and summary["valid_baseline"] is False
    blockers = " | ".join(summary["valid_baseline_blockers"])
    assert "engineering_check" in blockers and "canonical" in blockers
    assert len(summary["valid_baseline_blockers"]) == 2, "only the fake-mode and kind blockers are absent here"


@pytest.mark.parametrize("name", ["reordered", "reworded", "same_content_other_bytes"])
def test_a_development_pilot_on_an_altered_selection_is_refused_before_anything_is_created(project: Project, name: str) -> None:
    manifest = variants(project)[name]
    rig = Rig()
    with pytest.raises(RunnerRefusal, match="canonical"):
        live(project, run_kind=RUN_KIND_DEVELOPMENT, manifest=manifest, rig=rig)
    assert rig.factory_calls == 0 and not live_dir(project).exists()
    assert not (project.root / "results" / "development").exists(), "the refusal comes before any directory is made"


def test_the_question_count_and_the_ids_hash_describe_the_run_not_the_frozen_pilot(project: Project) -> None:
    manifest = variants(project)["subset"]
    live(project, run_kind=RUN_KIND_ENGINEERING, manifest=manifest)
    block = json.loads((live_dir(project) / "run_config.json").read_text())["identity"]["pilot_manifest"]
    assert block["question_count"] == 3 and block["question_ids_sha256"] == sha(b"q1\nq2\nq3")
    assert block["parent_pilot_id"] == PILOT_ID


def test_canonical_follows_the_bytes_and_not_the_path(project: Project) -> None:
    copy = write_variant(project, "byte-copy", project.runtime_manifest.read_text(encoding="utf-8"))
    assert copy.read_bytes() == project.runtime_manifest.read_bytes()
    result, _ = live(project, run_kind=RUN_KIND_DEVELOPMENT, manifest=copy)
    assert result.summary["canonical_manifest"] is True and result.summary["valid_baseline"] is True


def test_a_missing_frozen_manifest_is_recorded_as_null_and_is_not_canonical(project: Project) -> None:
    custom = write_variant(project, "elsewhere", project.runtime_manifest.read_text(encoding="utf-8"))
    project.runtime_manifest.unlink()
    result, _ = live(project, run_kind=RUN_KIND_ENGINEERING, manifest=custom)
    block = json.loads((live_dir(project) / "run_config.json").read_text())["identity"]["pilot_manifest"]
    assert block["frozen_runtime_manifest_sha256"] is None and block["canonical"] is False
    assert result.summary["valid_baseline"] is False
    with pytest.raises(RunnerRefusal, match="canonical"):
        live(project, "live-b", run_kind=RUN_KIND_DEVELOPMENT, manifest=custom)


def test_a_clean_canonical_development_pilot_is_a_valid_baseline_and_says_only_that_it_is_mechanical(project: Project) -> None:
    result, _ = live(project, run_kind=RUN_KIND_DEVELOPMENT)
    summary = result.summary
    assert summary["valid_baseline"] is True and summary["valid_baseline_blockers"] == []
    assert summary["run_kind"] == RUN_KIND_DEVELOPMENT and summary["canonical_manifest"] is True
    text = " ".join(summary["valid_baseline_means"].split())
    assert "mechanical" in text and "does not mean" in text and "approved" in text


def test_the_fake_run_blockers_include_the_kind_and_the_manifest_only_when_they_apply(project: Project) -> None:
    from test_live_runner import go

    blockers = go(project, Rig()).summary["valid_baseline_blockers"]
    assert any("not a live run" in b for b in blockers) and any("engineering_check" in b for b in blockers)
    assert not any("canonical" in b for b in blockers), "the default manifest is the frozen one"


def test_the_score_summary_carries_the_kind_the_manifest_flag_and_the_baseline_flag(project: Project) -> None:
    live(project, run_kind=RUN_KIND_ENGINEERING)
    scored = lr.score_run_live(project_root=project.root, run_dir=live_dir(project))
    assert scored.summary["run_kind"] == RUN_KIND_ENGINEERING and scored.summary["canonical_manifest"] is True
    assert scored.summary["valid_baseline"] is False
    assert any("engineering_check" in b for b in scored.summary["valid_baseline_blockers"])
    on_disk = json.loads((live_dir(project) / "score_summary.json").read_text())
    assert on_disk == json.loads(json.dumps(scored.summary))


def test_a_subset_run_scores_against_a_subset_evaluation_manifest_and_says_it_is_not_canonical(project: Project) -> None:
    manifest = variants(project)["subset"]
    live(project, run_kind=RUN_KIND_ENGINEERING, manifest=manifest)
    evaluation = json.loads(project.evaluation_manifest.read_text())
    evaluation["questions"] = {k: v for k, v in evaluation["questions"].items() if k in ("q1", "q2", "q3")}
    subset_eval = write_variant(project, "eval-subset", evaluation)
    scored = lr.score_run_live(project_root=project.root, run_dir=live_dir(project), evaluation_manifest_path=subset_eval)
    assert scored.summary["canonical_manifest"] is False and scored.summary["valid_baseline"] is False


# --- CLI ------------------------------------------------------------------


def cli_live_args(project: Project, tmp: Path, *extra: str) -> list[str]:
    config = tmp / "provider.json"
    config.write_text(json.dumps(valid_live_config()))
    return [
        "run", "--mode", "live", "--pilot-id", PILOT_ID, "--project-root", str(project.root),
        "--run-dir", str(live_dir(project, "cli-live")), "--provider-config", str(config), "--safety-ceiling", "1", *extra,
    ]  # fmt: skip


def test_the_cli_refuses_a_live_run_without_run_kind_before_any_client_exists(
    project: Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built = tripwire_clients(monkeypatch)
    monkeypatch.setattr("os.environ", {**LIVE_READY_ENV})
    assert cli.main(cli_live_args(project, tmp_path)) == lr.EXIT_REFUSED
    assert "--run-kind" in capsys.readouterr().err
    assert built == [] and not live_dir(project, "cli-live").exists()


def test_the_cli_accepts_only_the_two_kinds(project: Project, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(cli_live_args(project, tmp_path, "--run-kind", "scientific_evaluation"))
    assert info.value.code == lr.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "--run-kind" in err


# ---------------------------------------------------------------------------
# K3: the registry
# ---------------------------------------------------------------------------


def registry_record(run_id: str, kind: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1, "run_id": run_id, "recorded_at": "2026-09-30T00:00:00Z", "kind": kind, "status": "completed",
        "purpose": "fixture", "dataset": {"name": "fixture", "split": None, "pilot_id": None}, "identity": {"files": {}, "settings": {}},
        "command": "x", "environment": None, "seed": None, "models": None,
        "attempts": [{"attempt": 1, "status": "completed", "code_commit": None, "dirty": None, "started_at": None, "ended_at": None, "note": None}],
        "outputs": outputs, "checkpoint": None, "related_runs": [], "limitations": [], "evidence": [],
    }  # fmt: skip


def registry_with(tmp_path: Path, record_kind: str, config: dict[str, Any] | None) -> list[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    outputs = []
    if config is not None:
        path = tmp_path / "runs" / "x" / "run_config.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(config))
        outputs.append({"path": "runs/x/run_config.json", "sha256": sha(path.read_bytes()), "in_git": True, "backup": "git"})
    reg = tmp_path / "registry.jsonl"
    reg.write_text(json.dumps(registry_record("kind-fixture-1", record_kind, outputs)) + "\n")
    return registry.check(tmp_path, reg)[0]


def test_the_registry_refuses_a_record_whose_kind_differs_from_its_run_config(tmp_path: Path) -> None:
    errors = registry_with(tmp_path, "development_pilot", {"kind": "engineering_check"})
    assert any("kind" in e and "run_config.json" in e for e in errors)


def test_the_registry_accepts_matching_kinds_and_records_without_a_run_config(tmp_path: Path) -> None:
    assert registry_with(tmp_path / "a", "engineering_check", {"kind": "engineering_check"}) == []
    assert registry_with(tmp_path / "b", "development_pilot", None) == []
    assert registry_with(tmp_path / "c", "development_pilot", {"no_kind_key": True}) == []
