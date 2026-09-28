# Phase 3 Acceptance Checklist

## Readiness Locks

- [x] Dataset scope fixed in `experiment_matrix.md`
- [x] System profile scope fixed (`naive_rag`, `faar_full`, and three ablations)
- [x] Metrics protocol fixed in `metrics_protocol.md`
- [x] Reproducibility and API mode fixed in `runbook.md`

## Implementation Completion Criteria

- [x] All five profiles run end-to-end on configured example slice.
- [x] Metrics (`NDCG@5`, `Recall@5`, `EM`, `F1`) generated for all profiles.
- [x] Ablation outputs generated and documented.
- [x] Results exported to `artifacts/phase3/`.
- [x] Logs archived under `logs/phase3/`.
- [x] Results docs completed (`results.md`, `ablation_analysis.md`, `error_analysis.md`).
- [x] Phase report and archive index updated.
- [x] `python -m pytest` passes.
