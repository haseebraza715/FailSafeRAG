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

Aggregate outputs:

```bash
faar-demo aggregate-phase3
```

## Output Paths

- logs: `logs/phase3/<profile>/`
- profile summaries: `artifacts/phase3/<profile>_summary.json`
- global summary: `artifacts/phase3/metrics_summary.json`
- tables: `artifacts/phase3/retrieval_metrics.csv`, `artifacts/phase3/qa_metrics.csv`
