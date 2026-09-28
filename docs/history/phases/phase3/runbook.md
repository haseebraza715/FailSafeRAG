# Phase 3 Runbook

## Reproducibility Policy

- Default seed: `42`
- Every run writes:
  - run ID
  - timestamp
  - profile name
  - seed
  - selected example IDs
  - model/back-end settings

## API Mode

Phase 3 is API-enabled.

- `vlm_backend` can be `openai` for visual fallback profiles.
- If no API key is present, fallback path reports explicit skipped status in logs.
- Budget control is enforced via run-level `max_examples` and profile-level settings.

## Commands

Run one profile:

```bash
faar-demo run-benchmark --profile faar_full --max-examples 20 --seed 42
```

Run all profiles:

```bash
faar-demo run-benchmark-all --max-examples 20 --seed 42
```

Run a stratified slice that keeps one example per manual failure label:

```bash
faar-demo run-benchmark-all --max-examples 40 --stratify-by manual_failure_type --examples-per-stratum 1 --seed 42
```

Run only table-heavy finance examples:

```bash
faar-demo run-benchmark --profile faar_full --max-examples 40 --doc-types finance --evidence-sources table --seed 42
```

Run only manually labeled OCR-corruption cases:

```bash
faar-demo run-benchmark-all --manual-failure-types text_corruption,structure_corruption --max-examples 40 --seed 42
```

Aggregate outputs:

```bash
faar-demo aggregate-phase3
```

## Output Paths

- logs: `logs/phase3/<profile>/`
- profile summaries: `artifacts/phase3/<profile>_summary.json`
- global summary: `artifacts/phase3/metrics_summary.json`
- tables: `artifacts/phase3/retrieval_metrics.csv`, `artifacts/phase3/qa_metrics.csv`

## Slice Selection Notes

- `--doc-types` filters on manifest `doc_type`.
- `--evidence-sources` filters on manifest `evidence_source`.
- `--manual-failure-types` filters on Phase 0 manual labels.
- `--stratify-by` supports `doc_type`, `evidence_source`, or `manual_failure_type`.
- `--examples-per-stratum` keeps the slice balanced when stratification is enabled.
- `run-benchmark-all` resolves the slice once and reuses the same example IDs for every profile so comparisons remain fair.
