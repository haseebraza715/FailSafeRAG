# Audit scripts

These scripts re-run the offline checks behind the [live-baseline readiness review](../../docs/reports/live-baseline-readiness-2026-09-30.md). They call no model, open no network connection and read no credential. Run them from the repository root with the pinned virtual environment.

Each script takes explicit input paths, defaults to paths relative to the repository root, and writes only into the directory given by `--out`. When a required local asset is missing, it prints a one-line `error:` message and exits with code 2. A failed check exits with code 1. No script writes into `results/pilots/`, and the scripts that write files refuse an `--out` directory under it. No output holds document text.

The dry-run directory (`prepared_requests.jsonl`, `dry_run_summary.json`) holds full document text and stays local, as do the MinerU files. Create the directory with `scripts/experiments/run_pilot_live.py dry-run --out <dir>`. The examples below use `.local/work/dry-run` and write to `.local/work/audits`.

| Script | Reads | Answers |
| --- | --- | --- |
| `audit_prompts.py` | dry-run directory, runtime manifest, MinerU files | Do the saved prompts match the repository's prompt builder, the manifest and the MinerU text? Are the blocks well formed, the document name absent outside the evidence, the input and cost bounds right, the skip reasons true, and no evidence text aimed at a model? Runtime data only. |
| `audit_prompt_leakage_diagnostic.py` | dry-run directory, `evaluation_manifest.json`, clean reference text | Does any gold string, evidence context or reference passage appear in a prompt outside the evidence? A diagnostic that reads evaluation data. Its output must never feed runtime preparation. |
| `cost_and_ceiling.py` | dry-run directory | Input-bound totals, one-attempt and multi-attempt cost bounds, an expected-cost estimate from stated assumptions, and a replay of the safety ceiling against `faar.request_budget.SafetyLedger`. |
| `select_live_check.py` | dry-run directory, runtime manifest | Applies the fixed rule `faar-live-check-v1` and writes the selected ids and a derived runtime manifest. |
| `check_subset.py` | frozen and derived runtime manifests | Is the derived manifest an exact subset? Prints its sha256, question count and question-id digest. |
| `verify_live_check.py` | a live-check run directory | Do the run's records meet the mechanical success criteria (readiness report section 5)? Fields that only newer runner code records are read when present and reported as `absent` otherwise. |
| `case_review_counts.py` | `docs/reports/ohr-dev-v1-agent-review-2026-09-30.cases.json` | Counts of the agent review of the 20 inspection cases. `--check-markdown` compares every row with the report's summary table. |

## Commands

```bash
.local/venv-prebaseline/bin/python scripts/audits/audit_prompts.py --dry-run-dir .local/work/dry-run --out .local/work/audits
.local/venv-prebaseline/bin/python scripts/audits/audit_prompts.py --dry-run-dir .local/work/dry-run --out .local/work/audits --trace-dry-run --provider-config <config.json>
.local/venv-prebaseline/bin/python scripts/audits/audit_prompt_leakage_diagnostic.py --dry-run-dir .local/work/dry-run --out .local/work/audits
.local/venv-prebaseline/bin/python scripts/audits/cost_and_ceiling.py --dry-run-dir .local/work/dry-run --out .local/work/audits --fast-multiplier 1.7
.local/venv-prebaseline/bin/python scripts/audits/select_live_check.py --dry-run-dir .local/work/dry-run --out .local/work/livecheck
.local/venv-prebaseline/bin/python scripts/audits/check_subset.py --derived .local/work/livecheck/runtime_manifest.livecheck-v1.json
.local/venv-prebaseline/bin/python scripts/audits/verify_live_check.py results/development/<run_id>
.local/venv-prebaseline/bin/python scripts/audits/case_review_counts.py --check-markdown docs/reports/ohr-dev-v1-agent-review-2026-09-30.md
```

`--trace-dry-run` runs a dry run under a Python audit hook and lists the data files it opens. Importing `faar.settings` reads `config/model_revisions.json`, and the trace lists that read separately.

## Assumptions in the expected cost

The bounds in `cost_and_ceiling.py` are facts about the prepared requests. The expected cost is an estimate, and every assumption is an argument that the output records:

- Input tokens come from the repository's heuristic (`faar.answer_prompt.estimate_tokens`), not a tokenizer. The low and high variants use `--chars-per-token-low`, `--chars-per-token-high`, `--cjk-tokens-low`, `--cjk-tokens-high` and `--framing-tokens`. They are a sensitivity check, not an envelope.
- Output tokens per request are `--output-tokens` (with `--output-tokens-low` and `--output-tokens-high`).
- `--retry-rate` adds billed retries as a multiplier. `--brief-output-tokens` and `--brief-retry-rate` give the study brief's assumptions.
- `--fast-multiplier` scales all three rates for the Fast-tier scenario.

## What the scripts do not preserve

Full prompts, OCR text and page images stay local by policy. The scripts regenerate the audit figures from those local files, and only compact outputs (ids, counts, hashes) can be committed.

## Tests

`tests/test_audits.py` runs offline on synthetic data:

```bash
.local/venv-prebaseline/bin/python -m pytest -q -p no:cacheprovider tests/test_audits.py
```
