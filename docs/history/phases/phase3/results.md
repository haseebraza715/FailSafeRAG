# Phase 3 Results

## Scope

Results below are aggregated from the current 40-example committed
`logs/phase3/` offline fixture artifacts using the Phase 3 metrics protocol.
The earlier five-row pilot table is historical and has been superseded by this
reproducible aggregation (fresh run, current code, mock VLM, ByT5 disabled for
offline reproducibility, seed 42).

## Main Metrics Table

| Profile | Count | NDCG@5 | Recall@5 | EM | F1 | Visual fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `faar_full` | 40 | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0250 |
| `faar_no_backtrack` | 40 | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0250 |
| `faar_no_vlm` | 40 | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0000 |
| `faar_no_diagnosis` | 40 | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0000 |
| `naive_rag` | 40 | 0.4533 | 0.5000 | 0.1500 | 0.2036 | 0.0000 |

## Measured recovery outcome

Per-example logs (`logs/phase3/faar_full/*.json`) now record a
`recovery_metrics` block comparing the direct-answer counterfactual with the
recovered answer. On this slice recovery changed **0 of 15** routed recovery
cases and the EM/F1 effect was `equal` in all 15 cases. Profiles are therefore at parity with
the naive baseline.

## Retrieval and QA Artifact References

- `artifacts/phase3/metrics_summary.json`
- `artifacts/phase3/retrieval_metrics.csv`
- `artifacts/phase3/qa_metrics.csv`

## Notes

- Regeneration command:
  `faar-demo run-benchmark-all --project-root . --max-examples 40 --seed 42 --vlm-backend mock --no-api-enabled --no-enable-byt5`
- The committed summaries are pinned by `tests/test_evidence_tripwire.py` so
  they cannot silently go stale relative to the current code.
- The visual fallback is a mock no-op in these runs; real-VLM answer quality is
  untested. Word-level ByT5 correction is disabled in this reproducible run.
