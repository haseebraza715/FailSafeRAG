# Phase 3 Ablation Analysis

## Mandatory Ablations

This phase includes three required ablations:

1. `faar_no_backtrack`
2. `faar_no_vlm`
3. `faar_no_diagnosis`

## Observed Comparison vs `faar_full`

From `artifacts/phase3/ablation_summary.json` and `artifacts/phase3/metrics_summary.json`
(40-example offline slice, mock VLM, seed 42):

- All five profiles report identical `EM`, `F1`, `NDCG@5`, and `Recall@5`.
- `faar_full` and `faar_no_backtrack` route the same 1/40 example to the mock visual
  fallback; the other profiles route none.

| Profile | NDCG@5 | Recall@5 | EM | F1 | Visual fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `faar_full` | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0250 |
| `faar_no_backtrack` | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0250 |
| `faar_no_vlm` | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0000 |
| `faar_no_diagnosis` | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0000 |
| `naive_rag` | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0000 |

## Measured recovery outcome

Each per-example log now records a direct-answer counterfactual alongside the
recovered answer. On this slice the measured outcome is:

- recovery changed the final answer in `0` of `15` routed recovery cases for `faar_full`;
- the `em` and `f1` effect distributions are `equal` for all 15 routed recovery cases.

## Interpretation

- On the current data slice, the typed-recovery profiles are at **parity** with the
  naive baseline: recovery does not change any predicted answer.
- The ablation differences are limited to routing (which examples would invoke the
  visual fallback) and cost, not answer quality.
- The mock visual fallback is a no-op, so real-VLM answer quality is **untested**.
- Final conclusions require broader runs across larger sample sizes and a real VLM.
