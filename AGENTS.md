# FAAR agent instructions

These are the shared project rules for every coding agent in this repository.
The user's instructions in the conversation take precedence over this file.

FAAR asks whether diagnosing failures in OCR-backed document question answering
and choosing a matching repair answers more questions correctly than simpler
recovery policies at comparable cost. The research value of this repository
depends on keeping evaluations clean and records honest, so the research rules
below apply even when they make a task slower.

## Where things are

- Research direction: [docs/research/study-brief.md](docs/research/study-brief.md). It is the current plan. [docs/research/aaai-plan.md](docs/research/aaai-plan.md) is the fixed protocol. Where the two conflict, follow the study brief and tell the user about the conflict.
- Current data and pilot reports: [docs/reports/](docs/reports/index.md).
- Run records and their format: [experiments/README.md](experiments/README.md).
- Paper: [paper/main.tex](paper/main.tex), with rules in [paper/README.md](paper/README.md).
- Cluster procedure: [SUPERVISOR_HANDOFF.md](SUPERVISOR_HANDOFF.md) and [docs/operations/runbook.md](docs/operations/runbook.md).
- History: [docs/history/](docs/history/README.md). These are records, not instructions. Older files there say to work on `main` or describe other layouts. Do not follow them.

## Research rules

- Label every run by what it measured, using the kinds in `experiments/README.md` (`data_preparation`, `engineering_check`, `development_pilot` and `scientific_evaluation`). Each kind supports a different claim. Mock-backend and smoke runs are engineering checks even when they report EM or F1.
- Keep gold answers, evidence-page labels, evidence contexts and clean ground-truth text out of runtime retrieval, gating, diagnosis and recovery. Clean text may serve only as a diagnostic reference condition.
- Do not edit source locks, the fixed split, the locked QA file or a frozen pilot selection (`config/`, `OHR-Bench/data/qas_v2.json`, `results/pilots/`). A different selection needs a new `pilot_id` and output directory.
- Treat missing or empty OCR as a recorded condition of the noisy input. Never exclude a question for it automatically, and never fill it from ground truth.
- Do not tune on test outcomes. Do not keep changing and re-running an experiment in search of a positive result. A null or negative result is evidence.
- Keep failed, interrupted and negative runs and their records.
- Record an unknown historical fact as unknown (`null` in records). Do not reconstruct timestamps, commands, seeds or outcomes.

## Experiment records

Follow [experiments/README.md](experiments/README.md) for every run that produces evidence. Keep each run's identity and output provenance intact. Resuming an unchanged run adds an attempt under the same `run_id`. Changed data, measurement code or experimental settings need a new run. A recorded hash identifies a file. It is not a backup, so say where a copy exists or record that none does.

## Implementation

- Read the relevant code first. Reproduce a bug before changing code when a reproduction is feasible.
- Keep changes inside the requested task. Report other problems you notice instead of fixing them in passing.
- Choose checks that exercise the change. Record a failure that exists before your change separately from a regression you caused. Known local issues are listed in [README.md](README.md#local-checks-and-known-issues).
- Keep tests away from real experiment logs, annotations and result files. Use temporary directories.
- Launch GPU or cluster jobs, paid API calls or large downloads only when the task authorizes them.
- Work in this session only. Do not delegate to subagents or other agents unless the user asks for that.

Checks that are safe to run locally:

```bash
.venv-aaai/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py
.venv-aaai/bin/ruff check .
.venv-aaai/bin/python scripts/experiments/registry.py check
.venv-aaai/bin/python scripts/data/build_pilot.py --project-root . --config config/pilots/ohr_dev_v1.json --no-render
```

The last command revalidates one frozen pilot and writes nothing. Run it for each config in `config/pilots/`. To re-check the data audit, pass `--out-dir` a temporary directory so the committed audit stays unchanged. The paper builds with `latexmk` as described in [paper/README.md](paper/README.md).

## Writing

For documentation, reports, PR descriptions, commit messages and manuscript prose, read and apply [.claude/skills/technical-writing/SKILL.md](.claude/skills/technical-writing/SKILL.md). It applies [.claude/skills/unslop/SKILL.md](.claude/skills/unslop/SKILL.md) to the same text. Both files are plain Markdown that any agent can read.

- The paper follows scientific manuscript structure. Apply the sentence-level rules to it, not a software-documentation layout.
- Keep verified facts, interpretations, proposals and pending results visibly distinct. Report results only from registered runs.
- Keep every command, path and count true at the commit that lands it.
- When you report progress to the user, use plain language: what changed, what you checked, and what is still open.

## Git

- Work on the task branch the user names, or create one. Do not work directly on `main`.
- Keep unrelated working-tree changes intact and out of your commits. Stage files by name after reviewing the diff.
- Commit only when the task authorizes it. Use one logical change per commit and a one-line conventional message (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Do not push, force-push or rewrite history without the user's authorization.
- Do not commit `.env`, secrets, virtual environments, model caches or bulk data. `.gitignore` covers the usual paths. Check any new large or binary file before staging it.
- Add no AI attribution, `Co-authored-by` trailers or generated-by footers.
