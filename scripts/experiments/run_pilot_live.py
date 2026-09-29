"""Drive an answer model over the frozen pilot, with durable records and a safe resume.

    dry-run    prepare all requests and write the exact prompts, hashes and cost bounds; call no provider
    run        send pending requests, one event line per step; start or resume a run directory
    status     describe a run from its files
    export     rebuild predictions.jsonl and run_summary.json from the records
    reconcile  record how an attempt with an unknown outcome was resolved
    reopen     give an execution_failed question further attempts, after you fixed the cause
    score      join the exported predictions to the evaluation manifest

Fake mode (the default) uses a scripted in-process provider. It sends nothing and
measures no model, so its runs are engineering checks. Live mode needs
--provider-config, --safety-ceiling and an environment variable named at the end
of --help, and this script never starts it without all three.
Register a run yourself with scripts/experiments/registry.py; this script never
writes experiments/registry.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar import live_runner
from faar.answer_providers import SimulatedCrash
from faar.live_contract import ALLOWED_OPENAI_PARAMS
from faar.live_runner import (
    EXIT_BUDGET_LIMITED,
    EXIT_EXECUTION_FAILED,
    EXIT_INTERNAL_ERROR,
    EXIT_NEEDS_ATTENTION,
    EXIT_OK,
    EXIT_REFUSED,
    LIVE_ENV_NAME,
    LIVE_ENV_VALUE,
    RESOLUTIONS,
    RunOptions,
)
from faar.pilot_runner import RunnerRefusal

EPILOG = f"""\
exit codes:
  {EXIT_OK}  complete: every question has a terminal record and none is execution_failed
  {EXIT_REFUSED}  refusal, usage error or unreadable input; nothing was dispatched (also: another invocation holds the run lock)
  {EXIT_EXECUTION_FAILED}  the run is complete but at least one question is execution_failed
  {EXIT_INTERNAL_ERROR}  unexpected internal error (a traceback follows); the run directory keeps every record written so far
     (a fake script's crash_after_send step kills the process with SIGKILL instead, like a real crash)
  {EXIT_BUDGET_LIMITED}  budget_limited: the safety ceiling stopped the run; unserved questions are execution_failed with unserved true
  {EXIT_NEEDS_ATTENTION}  needs_reconciliation, stopped or incomplete: an attempt has an unknown outcome, a provider error
     stopped the run, or the invocation was interrupted; run status, then reconcile or run again

run directories:
  fake mode   results/engineering/<run_id>/ inside the project, or any directory outside it
  live mode   results/development/<run_id>/ inside the project, or any directory outside it
  never under results/pilots/

resume: run the same command again. A question that has a saved response is never sent again. The run
identity (pilot, inputs, retrieval, prompt template, model settings, prices, retry parameters, code) must match
exactly, or the run is refused before any dispatch. The safety ceiling is not identity: pass the current ceiling
again, or raise it with --raise-safety-ceiling and --authorization-note.

live mode needs all of: --mode live, --provider-config PATH, --safety-ceiling AMOUNT, and {LIVE_ENV_NAME}={LIVE_ENV_VALUE}
in the environment. Credentials are read only after those checks pass. Nothing in the test suite runs live mode.

live endpoint: the provider config's "endpoint" is the API base URL (for example https://api.openai.com/v1, not a
request path). The client sends only there, and the run identity records the client's base_url. Live mode refuses to
start while {", ".join(live_runner.REDIRECTING_ENV_NAMES)} is set, because those variables
can send the key to another URL or bill another organization or project. "params" in the provider config may hold
only: {", ".join(ALLOWED_OPENAI_PARAMS)}.
"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)


def _amount(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    def command(name: str, help_text: str, description: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name, help=help_text, description=description, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
        )

    def project(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--project-root", type=Path, default=REPO_ROOT, help="root that holds results/pilots and OHR-Bench (default: this checkout)"
        )

    def inputs(p: argparse.ArgumentParser) -> None:
        project(p)
        p.add_argument("--pilot-id", default="ohr_dev_v1", help="pilot whose runtime manifest is read (default: ohr_dev_v1)")
        p.add_argument(
            "--runtime-manifest", type=Path, help="runtime manifest path (default: results/pilots/<pilot-id>/runtime_manifest.json)"
        )

    dry = command(
        "dry-run",
        "prepare all requests and write the prompt preview; call no provider",
        "Prepare every request for the runtime questions and write prepared_requests.jsonl, prompt_preview.md and "
        "dry_run_summary.json. Calls no provider, needs no credentials and constructs no client. Refuses an output "
        "directory under results/pilots/.",
    )
    inputs(dry)
    dry.add_argument("--out", type=Path, required=True, help="output directory for the three dry-run files")
    dry.add_argument(
        "--provider-config",
        type=Path,
        help="optional provider config JSON for the output limit and prices; without it the study brief's unapproved values are used",
    )

    run = command(
        "run",
        "start or resume a run",
        "Prepare the requests, check the run identity, send every pending request in manifest order and export. "
        "Run the same command again to resume.",
    )
    inputs(run)
    run.add_argument("--run-dir", type=Path, required=True, help="run directory, for example results/engineering/<run_id>")
    run.add_argument("--run-id", help="run id, lowercase [a-z0-9._-] (default: the run directory name)")
    run.add_argument("--mode", choices=("fake", "live"), default="fake", help="fake (default) or live")
    run.add_argument("--fake-script", type=Path, help="JSON script for the fake provider (required with --mode fake)")
    run.add_argument(
        "--provider-config", type=Path, help="provider config JSON (required with --mode live; optional override in fake mode)"
    )
    run.add_argument(
        "--safety-ceiling",
        type=_amount,
        help="operating cost limit in the price table's currency; counts every attempt at its upper bound",
    )
    run.add_argument(
        "--raise-safety-ceiling",
        type=_amount,
        help="new higher ceiling for a resumed run; --safety-ceiling must name the current one",
    )
    run.add_argument("--authorization-note", help="who approved the higher ceiling and why; recorded in the event log")

    status = command("status", "describe a run from its files", "Print run state, counts, cost and the next step. Changes nothing.")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--json", action="store_true", help="print the status as JSON")

    export = command(
        "export",
        "rebuild predictions.jsonl and run_summary.json",
        "Rebuild the exports from requests.jsonl, attempts.jsonl and responses/. Safe to repeat.",
    )
    export.add_argument("--run-dir", type=Path, required=True)

    reconcile = command(
        "reconcile",
        "resolve an attempt with an unknown outcome",
        "Record the resolution of an attempt whose outcome is unknown. Check the provider's own records first. "
        "This never contacts the provider.",
    )
    reconcile.add_argument("--run-dir", type=Path, required=True)
    reconcile.add_argument("attempt_id", metavar="ATTEMPT_ID")
    reconcile.add_argument("--resolution", choices=RESOLUTIONS, required=True)
    reconcile.add_argument("--note", required=True, help="what you checked and what you found")

    reopen = command(
        "reopen",
        "give an execution_failed question further attempts",
        "Append a question_reopened event for a question whose status is execution_failed. The question gets up to "
        "max_attempts further attempts on the next run, counted from the event. Earlier attempts and their costs stay. "
        "Use it after fixing the cause. A pending question (unserved or budget-limited) needs no reopen: run again. "
        "A question with an unknown outcome needs reconcile. An answered question is never reopened. Refused without "
        "--note, while another invocation holds the lock, and on a scored run. This never contacts the provider.",
    )
    reopen.add_argument("--run-dir", type=Path, required=True)
    reopen.add_argument("question_id", metavar="QUESTION_ID")
    reopen.add_argument("--note", required=True, help="what you fixed and why the question may be tried again")

    score = command(
        "score",
        "score the exported predictions",
        "Join predictions to the evaluation manifest with faar.ohr_scoring. Scores only a run whose state is complete "
        "(every question answered, no_evidence or execution_failed; none pending, unknown or unserved). Scoring is "
        "final for a run: afterwards run, reconcile, reopen and any export that would change a file are refused.",
    )
    project(score)
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument(
        "--evaluation-manifest",
        type=Path,
        help="evaluation manifest path (default: results/pilots/<pilot_id of the run>/evaluation_manifest.json)",
    )
    return parser


def _check_same_checkout() -> None:
    """The run identity hashes the imported faar package, so it must come from this script's checkout."""
    package_dir = Path(live_runner.__file__).resolve().parent
    if package_dir != SRC.resolve() / "faar":
        raise RunnerRefusal(
            f"faar is imported from {package_dir}, not from this script's checkout {SRC / 'faar'}. "
            "Run the script with PYTHONPATH set to this checkout's src directory."
        )


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    try:
        _check_same_checkout()
        if args.command == "dry-run":
            config = live_runner.load_provider_config(args.provider_config) if args.provider_config else None
            result = live_runner.dry_run(
                project_root=args.project_root,
                out_dir=args.out,
                pilot_id=args.pilot_id,
                runtime_manifest_path=args.runtime_manifest,
                config=config,
                config_source=f"provider config file {args.provider_config}" if args.provider_config else None,
            )
        elif args.command == "run":
            live_runner.check_live_enablement(
                mode=args.mode,
                has_provider_config=args.provider_config is not None,
                safety_ceiling=args.safety_ceiling,
                environ=os.environ,
            )
            if args.safety_ceiling is None:
                raise RunnerRefusal("--safety-ceiling AMOUNT is required")
            config = (
                live_runner.load_provider_config(args.provider_config, simulated=args.mode == "fake")
                if args.provider_config
                else None
            )
            options = RunOptions(
                project_root=args.project_root,
                run_dir=args.run_dir,
                mode=args.mode,
                pilot_id=args.pilot_id,
                runtime_manifest_path=args.runtime_manifest,
                run_id=args.run_id,
                safety_ceiling=args.safety_ceiling,
                raise_safety_ceiling=args.raise_safety_ceiling,
                authorization_note=args.authorization_note,
                config=config,
                cli_script=Path(__file__).resolve(),
                code_root=REPO_ROOT,
                command=["scripts/experiments/run_pilot_live.py", *args_list],
            )
            factory, descriptor = live_runner.wire_provider(options, fake_script=args.fake_script, environ=os.environ)
            result = live_runner.execute_run(options, provider_factory=factory, descriptor=descriptor, environ=os.environ)
        elif args.command == "status":
            status = live_runner.run_status(args.run_dir)
            print(json.dumps(status, indent=2) if args.json else live_runner.format_status(status))
            return EXIT_OK
        elif args.command == "export":
            result = live_runner.export_run(args.run_dir)
        elif args.command == "reconcile":
            result = live_runner.reconcile_attempt(
                run_dir=args.run_dir, attempt_id=args.attempt_id, resolution=args.resolution, note=args.note
            )
        elif args.command == "reopen":
            result = live_runner.reopen_question(run_dir=args.run_dir, question_id=args.question_id, note=args.note)
        else:
            result = live_runner.score_run_live(
                project_root=args.project_root, run_dir=args.run_dir, evaluation_manifest_path=args.evaluation_manifest
            )
    except RunnerRefusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except SimulatedCrash as exc:
        # A fake script's crash_after_send step. Die the way a killed process does: no cleanup, no exit code.
        print(f"simulated crash: {exc}", file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        raise
    except Exception:
        traceback.print_exc()
        return EXIT_INTERNAL_ERROR
    print(result.message)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
