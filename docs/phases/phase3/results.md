# Phase 3 Results

## Scope

Results below are aggregated from current `logs/phase3/` artifacts using the Phase 3 metrics protocol.

## Main Metrics Table

| Profile | Count | NDCG@5 | Recall@5 | EM | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `faar_full` | 5 | 0.6000 | 0.6000 | 0.0000 | 0.0124 |
| `faar_no_backtrack` | 5 | 0.6000 | 0.6000 | 0.0000 | 0.0124 |
| `faar_no_vlm` | 5 | 0.6000 | 0.6000 | 0.0000 | 0.0124 |
| `faar_no_diagnosis` | 5 | 0.7262 | 0.8000 | 0.0000 | 0.0161 |
| `naive_rag` | 5 | 0.7262 | 0.8000 | 0.0000 | 0.0161 |

## Retrieval and QA Artifact References

- `artifacts/phase3/metrics_summary.json`
- `artifacts/phase3/retrieval_metrics.csv`
- `artifacts/phase3/qa_metrics.csv`

## Notes

- Current run slice is small and intended to validate full pipeline wiring and reporting.
- Larger-scale benchmark claims should rely on extended Phase 3 runs with broader manifest coverage.
