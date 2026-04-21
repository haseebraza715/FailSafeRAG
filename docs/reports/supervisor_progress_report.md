# Supervisor Progress Report

## FAAR: Failure-Aware Agentic Recovery for OCR-RAG

**Status date:** April 21, 2026  
**Repository:** [failure-aware-ocr-rag](https://github.com/haseebraza715/failure-aware-ocr-rag)  
**Branch:** `main`  
**Purpose:** Provide a supervisor-ready, evidence-based summary of what has been built, what was updated in the latest cycle, and what the current results show.

---

## 1. Executive Summary

The project remains complete through Phases 0 to 4 and has now added a Phase 5 writing handoff plus another implementation-improvement cycle focused on answer extraction, OCR-correction safety, evaluation control, and reproducibility.

What is complete:

- end-to-end FAAR pipeline with typed failure diagnosis and recovery routing;
- formal method-to-code mapping (`Q`, `F`, `pi`, `U`);
- benchmark runner, ablation profiles, metrics exports, and claim-assessment artifacts;
- Phase 5 claim-safe writing scaffolding and repository CI test automation.

What was updated in the latest cycle:

- improved answer extraction for exact-match-sensitive questions;
- added guarded word-level correction acceptance to reduce harmful rewrites;
- added broader benchmark-slice selection and stratification controls;
- added GitHub Actions testing so verification no longer relies only on local runs.

Current conclusion:

- engineering readiness is stronger than the previous report;
- exact-match and F1 on the 40-example slice improved substantially after the extraction update;
- typed-recovery profiles no longer trail the naive baseline on that slice, but they still do not outperform it;
- cost-control and diagnosis-policy consistency claims remain supported;
- broader quality-improvement claims are still not yet supported.

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

### Latest implementation cycle completed

1. **Answer extraction upgrade**
   - Added question-aware extractive answer selection for yes/no, numeric, range, and date-like questions.
   - Main effect: reduced long-span answers for EM-sensitive benchmark items.

2. **Word-level correction hardening**
   - Added correction guardrails to skip non-Latin and formula-like content.
   - Added acceptance checks for numeric drift, excessive length shifts, and low token preservation.

3. **Evaluation control upgrade**
   - Added benchmark filtering by `doc_type`, `evidence_source`, and manual failure labels.
   - Added optional stratification by `doc_type`, `evidence_source`, or `manual_failure_type`.

4. **Writing and reproducibility upgrade**
   - Added Phase 5 writing-plan and claim-safe outline documents.
   - Added GitHub Actions `pytest` workflow for automated regression checks on push and PR.

---

## 3. Results Snapshot (Current Data)

### 3.1 Failure-label grounding (Phase 0, n=40)

| Label | Count |
| --- | --- |
| `no_issue` | 33 |
| `text_corruption` | 5 |
| `structure_corruption` | 2 |

Source: [manual_labels.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/data/phase0/manual_labels.csv)

### 3.2 Latest benchmark rerun (April 21, 2026, n=40/profile)

Rerun command used for current artifacts:

```bash
./.venv/bin/faar-demo run-benchmark-all --project-root . --max-examples 40 --seed 42 --vlm-backend mock --no-api-enabled
```

| Profile | NDCG@5 | Recall@5 | EM | F1 | Visual fallback rate |
| --- | --- | --- | --- | --- | --- |
| `faar_full` | 0.4789 | 0.5250 | 0.1250 | 0.2036 | 0.0750 |
| `faar_no_backtrack` | 0.4789 | 0.5250 | 0.1250 | 0.2036 | 0.0750 |
| `faar_no_vlm` | 0.4789 | 0.5250 | 0.1250 | 0.2036 | 0.0000 |
| `faar_no_diagnosis` | 0.4789 | 0.5250 | 0.1250 | 0.2036 | 0.0000 |
| `naive_rag` | 0.4789 | 0.5250 | 0.1250 | 0.2036 | 0.0000 |

Source: [metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)

### 3.3 What changed versus the previous reported run

- EM improved from `0.0000` to `0.1250`.
- F1 improved from `0.0627` to `0.2036` for `faar_full`.
- The typed-recovery profiles no longer underperform the naive baseline on this slice.
- The profile deltas are now flat rather than negative:
  - `delta_f1_vs_naive_rag = 0.0`
  - `delta_em_vs_naive_rag = 0.0`

Interpretation:

- the latest extraction work improved output quality globally;
- current improvements are not unique to the typed-recovery path;
- retrieval-level differentiation between FAAR and naive remains unresolved on this slice.

### 3.4 Current claim assessment outcomes (Phase 4 rerun)

Rerun command used for current claim artifacts:

```bash
./.venv/bin/faar-demo run-phase4 --project-root .
```

| Question | Status | Key value(s) |
| --- | --- | --- |
| Does quality gating reduce expensive fallback usage? | `supported` | visual fallback rate `0.075`, estimated savings vs always-on visual `0.925` |
| Does typed recovery outperform naive baseline? | `not_supported` | `delta_f1_vs_naive_rag = 0.0`, `delta_em_vs_naive_rag = 0.0` |
| Does selective VLM improve quality vs no-VLM while controlling cost? | `not_supported` | `delta_f1_vs_faar_no_vlm = 0.0`, `delta_em_vs_faar_no_vlm = 0.0` |
| Do diagnosis categories align with executed behavior? | `supported` | diagnosis-policy consistency `1.0`, non-pass rate `0.475` |

Source: [claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)

### 3.5 Routing behavior highlights (`faar_full`)

- pass: `21/40` (`0.525`)
- word_level: `14/40` (`0.350`)
- structural: `3/40` (`0.075`)
- semantic: `2/40` (`0.050`)
- executed actions: `answer_direct=21`, `correct_text=14`, `invoke_vlm=3`, `retry_retrieval=2`

Source: [phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)

### 3.6 Case-study signal

- Phase 4 comparison currently shows `0` improvements, `0` regressions, and only ties between `faar_full` and `naive_rag` on the latest rerun.
- This is consistent with the updated benchmark table: the latest extraction changes lifted all profiles together rather than creating a measurable FAAR-only advantage.

Source: [case_studies.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/case_studies.json)

---

## 4. Current Status and Active Work

### Current status

- Implementation and documentation milestones remain complete through Phase 4.
- The system is runnable, tested, and better instrumented than in the previous report.
- Phase 5 writing scaffolding now exists and is aligned with the actual evidence.
- The main thesis claim should still be framed conservatively.

### Active workstream

1. Run larger or stratified benchmark slices using the new slice-selection tooling.
2. Investigate why extraction gains lift all profiles equally rather than improving FAAR selectively.
3. Add per-failure-type recovery-success reporting so typed-recovery value can be measured more directly.
4. Evaluate visual fallback in a fully enabled multimodal setting when available, rather than only the current offline mock configuration.
5. Convert the current evidence into supervisor-ready thesis drafting without overstating quality gains.

---

## 5. Risks and Constraints

1. The current 40-example slice is still small for strong research claims.
2. The latest rerun used `mock` visual fallback with `--no-api-enabled`, so current quality comparisons remain an offline reproducible evaluation rather than a real multimodal-quality test.
3. Typed-recovery benefit is currently neutral rather than positive against the naive baseline on this slice.
4. Word-level OCR recovery remains a risk area for formulas, multilingual text, and layout-heavy pages despite the new guardrails.

---

## 6. Verification and Reproducibility

- Full regression suite verified after the latest implementation cycle:
  - `./.venv/bin/python -m pytest -q` -> `26 passed`
- Current benchmark artifacts regenerated from the latest code:
  - `./.venv/bin/faar-demo run-benchmark-all --project-root . --max-examples 40 --seed 42 --vlm-backend mock --no-api-enabled`
- Current claim artifacts regenerated from the latest logs:
  - `./.venv/bin/faar-demo run-phase4 --project-root .`
- Automated CI is now present:
  - `.github/workflows/tests.yml`

---

## 7. Supervisor Takeaway

The project made meaningful engineering progress in this cycle. Answer quality on the 40-example slice improved substantially after the extraction upgrade, and the previous negative gap between FAAR and the naive baseline disappeared. However, the latest rerun still does not show a positive typed-recovery advantage, only parity. The strongest supported claims remain selective-cost control and diagnosis-policy consistency.

The most defensible thesis position at this point is:

- the FAAR framework is implemented, reproducible, and operationally coherent;
- the current system now avoids the earlier measured quality deficit;
- larger or stratified reruns are still needed before claiming answer-quality improvement over naive OCR-RAG.

---

## 8. Key Links

### Project overview

- [README.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/README.md)
- [ARCHITECTURE.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/ARCHITECTURE.md)

### Phase and report links

- [phase1_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase1_report.md)
- [phase2_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase2_report.md)
- [phase3_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase3_report.md)
- [phase4_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase4_report.md)
- [Phase 5 Writing Plan](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase5/writing_plan.md)

### Core artifacts used in this update

- [artifacts/phase3/metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)
- [artifacts/phase4/phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)
- [artifacts/phase4/claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)
- [artifacts/phase4/case_studies.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/case_studies.json)
- [artifacts/phase4/profile_deltas_vs_baseline.csv](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/profile_deltas_vs_baseline.csv)
