"""Run the frozen pilot offline: generate predictions from the runtime manifest, then score them.

    generate   read the runtime manifest and MinerU text, retrieve within each
               document, and save one prediction per question
    score      join saved predictions to the evaluation manifest and save scores

This is an engineering path. It uses a rule-based answer backend, local-hash
retrieval and no model or network call. Its settings are not the approved
scientific protocol. Register a run yourself with scripts/experiments/registry.py;
this script never writes experiments/registry.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from faar import pilot_runner
from faar.pilot_runner import EXIT_EXECUTION_FAILED, EXIT_OK, EXIT_REFUSED, RunnerRefusal

EPILOG = f"""\
exit codes:
  {EXIT_OK}  every question has a terminal record and none is execution_failed,
     or an existing identical run was verified with no execution_failed
  {EXIT_REFUSED}  refusal, usage error or unreadable input; nothing was written
  {EXIT_EXECUTION_FAILED}  the run is complete but at least one question is execution_failed
     (for score: the scored run contains execution_failed questions)

overwrite policy (generate and score never overwrite):
  new or empty --run-dir                 the run is written
  complete run, different fingerprint    refused
  complete run, same fingerprint         regenerated in memory and compared; identical
                                         bytes exit as "already complete, verified
                                         identical" without a write; different bytes
                                         are refused as nondeterministic
  partial, corrupt or unrecognised dir   refused
  a --run-dir under results/pilots/      refused

The fingerprint covers the src/faar code, this script, the runtime manifest, every
MinerU file hash, the retrieval settings, the answer backend and the injected failures.
"""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-dir", type=Path, required=True, help="run directory, for example results/engineering/<run_id>")
        p.add_argument("--project-root", type=Path, default=REPO_ROOT, help="root that holds results/pilots and OHR-Bench (default: this checkout)")

    generate = sub.add_parser(
        "generate",
        help="generate and save predictions",
        description="Generate predictions from the runtime manifest and MinerU text. Never opens the evaluation manifest.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common(generate)
    generate.add_argument("--run-id", help="run id, lowercase [a-z0-9._-] (default: the run directory name)")
    generate.add_argument("--pilot-id", default="ohr_dev_v1", help="pilot whose runtime manifest is read (default: ohr_dev_v1)")
    generate.add_argument("--runtime-manifest", type=Path, help="runtime manifest path (default: results/pilots/<pilot-id>/runtime_manifest.json)")
    generate.add_argument(
        "--inject-failure",
        action="append",
        default=[],
        metavar="QUESTION_ID[:STAGE]",
        help="force an execution_failed record for this question at STAGE (load, retrieve or answer; default answer). Repeatable. Recorded in run_config.json and the fingerprint.",
    )

    score = sub.add_parser(
        "score",
        help="score saved predictions",
        description="Join saved predictions to the evaluation manifest with faar.ohr_scoring and save scores.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common(score)
    score.add_argument("--evaluation-manifest", type=Path, help="evaluation manifest path (default: results/pilots/<pilot_id of the run>/evaluation_manifest.json)")
    return parser


def _check_same_checkout() -> None:
    """The fingerprint hashes the imported faar package, so it must come from this checkout."""
    package_dir = Path(pilot_runner.__file__).resolve().parent
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
        if args.command == "generate":
            result = pilot_runner.generate_run(
                project_root=args.project_root,
                run_dir=args.run_dir,
                run_id=args.run_id,
                pilot_id=args.pilot_id,
                runtime_manifest_path=args.runtime_manifest,
                inject_failures=args.inject_failure,
                cli_script=Path(__file__).resolve(),
                code_root=REPO_ROOT,
                command=["scripts/experiments/run_pilot_offline.py", *args_list],
            )
        else:
            result = pilot_runner.score_run(
                project_root=args.project_root,
                run_dir=args.run_dir,
                evaluation_manifest_path=args.evaluation_manifest,
            )
    except RunnerRefusal as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(result.message)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
