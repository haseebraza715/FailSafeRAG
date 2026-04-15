# Supervisor Progress Report

## FAAR: Failure-Aware Agentic Recovery for OCR-RAG

**Status date:** April 15, 2026  
**Repository:** [failure-aware-ocr-rag](https://github.com/haseebraza715/failure-aware-ocr-rag)  
**Branch:** `main`  
**Purpose:** Provide a supervisor-ready, evidence-based summary of what has been done, what is currently being done, and what results have been achieved.

---

## 1. Executive Summary

The project has completed Phases 0 through 4 and is now in a transition from implementation/validation into research-quality improvement and Phase 5 writing.

What is complete:

- End-to-end FAAR pipeline with typed failure diagnosis and recovery routing.
- Formal method-to-code mapping (`Q`, `F`, `pi`, `U`) and reproducible experiment stack.
- Phase 3 benchmark runs with ablations and standardized metrics artifacts.
- Phase 4 claim-refinement analysis with explicit supported/not-supported outcomes.

What is currently being done:

- tightening answer quality for exact-match-sensitive QA;
- hardening word-level correction to reduce harmful rewrites;
- preparing broader evaluation slices for stronger statistical confidence;
- converting current evidence into claim-safe thesis/report writing.

Current conclusion:

- Engineering readiness is strong.
- Cost-control and policy-consistency claims are supported.
- Main quality-improvement claim versus current naive baseline is not yet supported on the 40-example slice.

---

## 2. What We Built and What We Did

### Phase 0 to Phase 4 delivery status

| Phase | Status | What was done | Evidence |
| --- | --- | --- | --- |
| Phase 0 | Complete | Built and manually inspected a 40-example OCR-grounded slice with failure labels. | [manual_labels.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/manual_labels.csv), [sample_manifest.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/sample_manifest.csv) |
| Phase 1 | Complete | Implemented FAAR controller graph with quality gate and typed actions. | [phase1 completion report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase1/completion_report.md) |
| Phase 2 | Complete | Formalized framework and mapped theory directly to implementation. | [methodology formalization](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase2/methodology_formalization.md), [code-math mapping](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase2/code_math_mapping.md) |
| Phase 3 | Complete | Executed benchmark matrix, ablations, and consolidated metrics exports. | [phase3 report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase3_report.md), [metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json) |
| Phase 4 | Complete | Added claim-assessment module and supervisor-facing claim evidence artifacts. | [phase4 report](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase4_report.md), [claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json) |

### Core system elements implemented

- Retrieval: hybrid lexical + dense + rank-fusion.
- Quality diagnosis: pass/semantic/word_level/structural classification.
- Recovery actions: direct answer, retrieval retry, OCR correction, selective visual fallback.
- Evaluation: per-profile logs, summary metrics, ablation outputs, claim-assessment outputs.
- Tooling and reliability: Typer CLI workflows and passing automated tests (`18 passed` on latest verification run).

---

## 3. Results Snapshot (What the Data Shows)

### 3.1 Failure-label grounding (Phase 0, n=40)

| Label | Count |
| --- | --- |
| `no_issue` | 33 |
| `text_corruption` | 5 |
| `structure_corruption` | 2 |

Source: [manual_labels.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/manual_labels.csv)

### 3.2 Main benchmark profile outcomes (Phase 3, n=40/profile)

| Profile | NDCG@5 | Recall@5 | EM | F1 | Visual fallback rate |
| --- | --- | --- | --- | --- | --- |
| `faar_full` | 0.3065 | 0.3250 | 0.0000 | 0.0627 | 0.0750 |
| `faar_no_backtrack` | 0.3065 | 0.3250 | 0.0000 | 0.0627 | 0.0750 |
| `faar_no_vlm` | 0.3065 | 0.3250 | 0.0000 | 0.0627 | 0.0000 |
| `faar_no_diagnosis` | 0.4789 | 0.5250 | 0.0000 | 0.0722 | 0.0000 |
| `naive_rag` | 0.4789 | 0.5250 | 0.0000 | 0.0722 | 0.0000 |

Source: [metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)

### 3.3 Claim assessment outcomes (Phase 4)

| Question | Status | Key value(s) |
| --- | --- | --- |
| Does quality gating reduce expensive fallback usage? | `supported` | visual fallback rate `0.075`, estimated savings vs always-on visual `0.925` |
| Does typed recovery outperform naive baseline? | `not_supported` | `delta_f1_vs_naive_rag = -0.0095`, `delta_em_vs_naive_rag = 0.0` |
| Does selective VLM improve quality vs no-VLM while controlling cost? | `not_supported` | `delta_f1_vs_faar_no_vlm = 0.0`, `delta_em_vs_faar_no_vlm = 0.0` |
| Do diagnosis categories align with executed behavior? | `supported` | diagnosis-policy consistency `1.0`, non-pass rate `0.475` |

Source: [claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)

### 3.4 Routing behavior highlights (`faar_full`)

- pass: `21/40` (`0.525`)
- word_level: `14/40` (`0.350`)
- structural: `3/40` (`0.075`)
- semantic: `2/40` (`0.050`)
- executed actions: `answer_direct=21`, `correct_text=14`, `invoke_vlm=3`, `retry_retrieval=2`

Source: [phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)

---

## 4. Current Status and What Is Being Worked On Now

### Current status

- Implementation and documentation milestones are complete through Phase 4.
- The system is runnable, testable, and well-instrumented.
- Research claims are evidence-constrained and transparently labeled.

### Active workstream

1. Improve answer-generation for EM-sensitive questions.
2. Add stronger guardrails to word-level correction to avoid regressions.
3. Re-run larger and stratified benchmark slices.
4. Extend reporting with per-failure-type recovery-success metrics.
5. Prepare Phase 5 writing using claim-safe language grounded in Phase 4 outcomes.

---

## 5. Risks and Constraints

1. EM is currently `0.0` across compared profiles.
2. Current evidence is from a 40-example slice; confidence will improve with larger runs.
3. Some OCR correction paths can degrade quality on formula-heavy/noisy cases.
4. Selective VLM currently shows cost-control value without quality lift on this slice.

---

## 6. Verification and Reproducibility

- Full regression suite previously verified: `./.venv/bin/python -m pytest -q` -> `18 passed`.
- Phase 4 analysis command available and validated: `./.venv/bin/faar-demo run-phase4 --project-root .`.
- All key decisions in this report are traceable to committed JSON/CSV artifacts.

---

## 7. Key Links

### Project overview

- [README.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/README.md)
- [ARCHITECTURE.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/ARCHITECTURE.md)

### Phase reports

- [phase1_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase1_report.md)
- [phase2_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase2_report.md)
- [phase3_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase3_report.md)
- [phase4_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase4_report.md)

### Core artifacts used in this update

- [artifacts/phase3/metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)
- [artifacts/phase4/phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)
- [artifacts/phase4/claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)
- [artifacts/phase4/case_studies.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/case_studies.json)
- [artifacts/phase4/profile_deltas_vs_baseline.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/profile_deltas_vs_baseline.csv)
