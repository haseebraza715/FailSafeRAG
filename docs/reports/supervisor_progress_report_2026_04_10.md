# Supervisor Progress Report

## FAAR: Failure-Aware Agentic Recovery for OCR-RAG

**Date:** April 10, 2026  
**Repository:** [failure-aware-ocr-rag](https://github.com/haseebraza715/failure-aware-ocr-rag)  
**Current branch:** `main`  
**Report purpose:** Provide a clear, evidence-based update on completed work, current results, and project direction.

---

## 1. Executive Summary

This project has completed Phases 0 through 4.

- Phase 0 established a manually inspected OCR-grounded dataset slice.
- Phase 1 delivered a working FAAR pipeline with typed recovery routing.
- Phase 2 formalized the framework (`Q`, `F`, `pi`, `U`) and mapped theory to code.
- Phase 3 delivered benchmark execution, ablations, and metrics artifacts.
- Phase 4 delivered claim refinement with explicit support/not-support decisions based on measured evidence.

Current state:

- The controller and routing logic are operational and well-instrumented.
- Cost-control behavior via selective visual fallback is supported.
- Diagnostic-policy consistency is supported.
- Quality-improvement claims over the current naive baseline are not yet supported on the present run slice.

---

## 2. Research Goal and Current Claim Status

### Original research intent

Build a failure-aware OCR-RAG system that improves QA quality under OCR corruption while controlling multimodal cost.

### Current claim status after Phase 4

- Supported:
  - quality-gated selective recovery controls expensive fallback usage
  - typed diagnostic routing is internally consistent with executed policy
- Not currently supported on current evaluation slice:
  - overall answer-quality gain vs naive baseline
  - selective VLM quality gain vs no-VLM profile

Evidence:

- [Phase 4 Claim Assessment](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)
- [Phase 4 Summary](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)

---

## 3. Work Completed So Far (Phase-by-Phase)


| Phase   | Status   | What was completed                                                                 | Primary evidence                                                                                                                                                                                                                                                             |
| ------- | -------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 0 | Complete | 40-example grounding sample, OCR/manual inspection, failure-type labels            | [manual_labels.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/manual_labels.csv), [sample_manifest.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/sample_manifest.csv)                                 |
| Phase 1 | Complete | End-to-end FAAR graph pipeline with quality gate and typed recovery actions        | [Phase 1 Completion Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase1/completion_report.md)                                                                                                                                        |
| Phase 2 | Complete | Formalization of `Q(c)`, `F(c)`, `pi(s)`, `U(a)` and code-to-math mapping          | [Methodology Formalization](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase2/methodology_formalization.md), [Code-Math Mapping](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase2/code_math_mapping.md) |
| Phase 3 | Complete | Benchmark runner, profile ablations, metrics exports, reporting                    | [Phase 3 Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase3_report.md), [metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)                             |
| Phase 4 | Complete | Claim refinement engine, supervisor-facing assessment artifacts, verification docs | [Phase 4 Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase4_report.md), [phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)                               |


---

## 4. Technology Stack Used

### Core runtime and orchestration

- Python 3.12+
- LangGraph (`langgraph`) for controller orchestration
- Typer (`typer`) for CLI workflows
- Pydantic (`pydantic`) for typed settings/config

### Retrieval and indexing

- Hybrid retrieval:
  - lexical: BM25 (`rank-bm25`)
  - dense: SentenceTransformer embeddings (`sentence-transformers`)
  - vector index: FAISS (`faiss-cpu`)
- Fusion strategy combines normalized BM25, dense similarity, and reciprocal-rank signal.

### Recovery stack

- Word-level correction: ByT5 (`transformers`, model default `google/byt5-small`)
- Semantic recovery: retrieval backtrack query expansion
- Structural recovery: selective visual fallback
  - backend modes: `mock` and OpenAI vision (`openai`, default model setting `gpt-4o`)

### Data and experiment processing

- Pandas (`pandas`) for data handling where needed
- JSON/CSV artifact outputs for reproducible reporting

### Testing and quality

- Pytest-based automated test suite
- Current project test status: `18 passed` on latest run

Reference:

- [pyproject.toml](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/pyproject.toml)

---

## 5. Experimental Setup Snapshot

- Primary benchmark context: OHR-Bench-derived Phase 0/3 workflow.
- Current analyzed run slice: 40 examples per profile.
- Profiles evaluated:
  - `naive_rag`
  - `faar_full`
  - `faar_no_backtrack`
  - `faar_no_vlm`
  - `faar_no_diagnosis`
- Metrics:
  - NDCG@5
  - Recall@5
  - EM
  - F1
  - Visual fallback rate

Metrics protocol:

- [Phase 3 Metrics Protocol](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase3/metrics_protocol.md)

---

## 6. Results

### 6.1 Phase 0 failure-type grounding labels (n=40)


| Label                  | Count |
| ---------------------- | ----- |
| `no_issue`             | 33    |
| `text_corruption`      | 5     |
| `structure_corruption` | 2     |


Source:

- [manual_labels.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/manual_labels.csv)

### 6.2 Main profile metrics (Phase 3 artifacts)


| Profile             | Count | NDCG@5 | Recall@5 | EM     | F1     | Visual Fallback Rate |
| ------------------- | ----- | ------ | -------- | ------ | ------ | -------------------- |
| `faar_full`         | 40    | 0.3065 | 0.3250   | 0.0000 | 0.0627 | 0.0750               |
| `faar_no_backtrack` | 40    | 0.3065 | 0.3250   | 0.0000 | 0.0627 | 0.0750               |
| `faar_no_vlm`       | 40    | 0.3065 | 0.3250   | 0.0000 | 0.0627 | 0.0000               |
| `faar_no_diagnosis` | 40    | 0.4789 | 0.5250   | 0.0000 | 0.0722 | 0.0000               |
| `naive_rag`         | 40    | 0.4789 | 0.5250   | 0.0000 | 0.0722 | 0.0000               |


Source:

- [Phase 3 metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)

### 6.3 Phase 4 claim-assessment outcomes


| Question                                                        | Status          | Key measured value(s)                                                                     |
| --------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------- |
| Q1: Does the quality gate reduce expensive recovery?            | `supported`     | `faar_full_visual_fallback_rate = 0.075`; estimated savings vs always-on visual = `0.925` |
| Q2: Does typed recovery outperform naive fallback?              | `not_supported` | `delta_f1_vs_naive_rag = -0.0095`; `delta_em_vs_naive_rag = 0.0`                          |
| Q3: Does selective VLM improve quality vs no-VLM at lower cost? | `not_supported` | `delta_f1_vs_faar_no_vlm = 0.0`; `delta_em_vs_faar_no_vlm = 0.0`                          |
| Q4: Do diagnostic categories align with observed behavior?      | `supported`     | diagnosis-policy consistency = `1.0`; non-pass rate = `0.475`                             |


Source:

- [claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)

### 6.4 Routing/failure distribution highlights (`faar_full`)

- `pass`: 21/40 (`0.525`)
- `word_level`: 14/40 (`0.350`)
- `structural`: 3/40 (`0.075`)
- `semantic`: 2/40 (`0.050`)
- Executed actions:
  - `answer_direct`: 21
  - `correct_text`: 14
  - `invoke_vlm`: 3
  - `retry_retrieval`: 2

Source:

- [phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)

### 6.5 Case-study signal

Phase 4 extracted both improvements and regressions:

- Improvement examples exist (word-level correction helps in some cases).
- Multiple regressions also exist, including severe negative F1 deltas in some word-level correction cases.

Source:

- [case_studies.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/case_studies.json)

---

## 7. Testing, Debugging, and Verification

### What was verified

- Dedicated Phase 4 unit tests for:
  - malformed JSON handling
  - claim logic and status outcomes
  - case-study selection behavior
- CLI test for artifact generation and CSV output
- Full project regression suite

### Latest verification outcome

- `./.venv/bin/python -m pytest -q` -> `18 passed`
- Phase 4 command execution succeeded:
  - `./.venv/bin/faar-demo run-phase4 --project-root .`

### Example issue resolved during Phase 4

- A claim-test fixture initially expected `not_supported` while generating no visual fallback usage, which correctly produced `inconclusive`.
- Fixture was corrected to include a structural/invoke-vlm case, and tests were re-run successfully.

Reference:

- [Phase 4 Testing and Verification](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase4/testing_and_verification.md)

---

## 8. Current Status (Supervisor View)

The project is in a strong engineering/documentation state and a mixed research-results state:

- Engineering maturity: good (modular pipeline, reproducible artifacts, automated tests, clear docs).
- Claim maturity: medium (some claims supported, key accuracy claim not yet supported on current slice).
- Immediate bottleneck: answer quality and correction quality, not orchestration.

In short:

- We can reliably run and analyze the system.
- We now have precise evidence showing what works and what still does not.

---

## 9. Main Risks and Constraints

1. EM remains `0.0` across compared profiles, limiting paper-strength QA claims.
2. Current results are based on a 40-example slice; broader coverage is needed for stronger statistical confidence.
3. Some correction paths degrade answers, especially in formula-heavy/noisy OCR cases.
4. Selective VLM currently shows cost-control value but no measured quality lift on this slice.

---

## 10. Recommended Next Steps

1. Improve answer-generation module for exact-match-sensitive QA.
2. Audit and harden word-level correction guardrails to reduce harmful rewrites.
3. Re-run larger benchmark slices and stratify by failure type for stronger evidence.
4. Add per-failure-type recovery-success metrics into standard artifacts.
5. Move into Phase 5 writing with evidence-constrained claim language from Phase 4.

---

## 11. Appendix: Key Links

### Project and architecture

- [README.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/README.md)
- [ARCHITECTURE.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/ARCHITECTURE.md)

### Phase reports

- [Phase 1 Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase1_report.md)
- [Phase 2 Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase2_report.md)
- [Phase 3 Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase3_report.md)
- [Phase 4 Report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase4_report.md)

### Core result artifacts

- [Phase 3 metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)
- [Phase 4 phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)
- [Phase 4 claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)
- [Phase 4 case_studies.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/case_studies.json)
- [Phase 4 profile_deltas_vs_baseline.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/profile_deltas_vs_baseline.csv)