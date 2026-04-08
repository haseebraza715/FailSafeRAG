# Phase 3 Ablation Analysis

## Mandatory Ablations

This phase includes three required ablations:

1. `faar_no_backtrack`
2. `faar_no_vlm`
3. `faar_no_diagnosis`

## Observed Comparison vs `faar_full`

From `artifacts/phase3/ablation_summary.json` and `artifacts/phase3/metrics_summary.json`:

- `faar_no_backtrack` matches `faar_full` on this run slice.
- `faar_no_vlm` matches `faar_full` on this run slice.
- `faar_no_diagnosis` differs from `faar_full` and aligns with `naive_rag` metrics on this run slice.

## Interpretation

- On the current data slice, VLM and backtracking did not change aggregate metrics.
- Diagnosis removal changed routing behavior, producing a different retrieval/answer profile.
- Final conclusions require broader runs across larger sample sizes.
