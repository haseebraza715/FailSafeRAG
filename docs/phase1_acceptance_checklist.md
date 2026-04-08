# Phase 1 Acceptance Checklist

This checklist maps the required Phase 1 prototype requirements to implementation evidence and validation status.

## Requirement Matrix

| Requirement | Evidence | Status | Notes |
| --- | --- | --- | --- |
| LangGraph controller with policy routing | `src/faar/graph.py` | Complete | Routes `answer_direct`, `correct_text`, `retry_retrieval`, `invoke_vlm`. |
| OCR-backed source text available to pipeline | `src/faar/data.py`, `artifacts/phase0/ocr_text/` | Complete (Phase 1 scope) | Pipeline consumes real OCR outputs produced in Phase 0 artifacts. |
| Chunking + metadata | `src/faar/chunking.py`, `src/faar/types.py` | Complete | Includes `doc_name`, `page_id`, `chunk_id`, `example_id`. |
| Retrieval (BM25 + dense + fused ranking) | `src/faar/retrieval.py` | Complete | BM25 + sentence-transformer + FAISS inner-product retrieval. |
| Quality gate | `src/faar/quality.py::quality_gate` | Complete | Computes quality score + reasons. |
| Diagnostic layer (`semantic`, `word_level`, `structural`) | `src/faar/quality.py::diagnose_failure` | Complete | Gate outcomes route to failure types. |
| Recovery actions (ByT5 / retry / VLM fallback) | `src/faar/recovery.py`, `src/faar/graph.py` | In progress | Recovery hardening and explicit statuses implemented in this phase update. |
| Answer generation + per-example logs | `src/faar/answering.py`, `src/faar/cli.py`, `logs/phase1/*.json` | In progress | Log schema will be extended with reproducibility metadata. |

## Baseline Branch Behavior (Pre-hardening Snapshot)

- `answer_direct`: observed in `logs/phase1/a57a9c82-aeae-4da7-9913-3e39bde1b1d2.json`
- `retry_retrieval`: observed in `logs/phase1/446d159e-b5c2-45dc-91cc-faaa931f3649.json`
- `correct_text`: observed in `logs/phase1/c132e7f1-47cb-4531-82f9-139fbf36e46f.json`
- `invoke_vlm`: observed in `logs/phase1/848d6744-2489-473b-b5fa-9dc0938d4510.json`

## Completion Criteria For This Implementation Round

1. Cross-platform path/config defaults work without hardcoded machine paths.
2. Recovery paths emit explicit status fields and reasons.
3. Logs include reproducibility metadata and action outcomes.
4. Unit + integration tests pass.
5. One end-to-end Phase 1 run is executed and archived.
