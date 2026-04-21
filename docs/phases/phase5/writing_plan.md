# Phase 5 Writing Plan

## Objective

Turn Phases 0 through 4 into a supervisor-ready draft while keeping the narrative strictly aligned with supported evidence.

## Current Writing Position

- The implementation stack is mature enough to describe concretely.
- The cost-control claim is supported.
- Diagnosis-policy alignment is supported.
- The main quality-improvement claim over `naive_rag` is not yet supported on the current 40-example slice.
- Selective VLM fallback is currently justified as a selective-cost mechanism, not as a demonstrated quality booster.

## Writing Priorities

1. Write the method and system sections from implemented code and committed artifacts.
2. Write experiments and analysis from Phase 3 and Phase 4 outputs without smoothing away negative findings.
3. Keep the contribution framing focused on selective recovery control, typed routing, and evidence transparency.
4. Mark larger or stratified reruns as active next-step work, not already-achieved evidence.

## Required Inputs

- [README.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/README.md)
- [ARCHITECTURE.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/ARCHITECTURE.md)
- [docs/phases/phase2/methodology_formalization.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase2/methodology_formalization.md)
- [docs/reports/phase3_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase3_report.md)
- [docs/reports/phase4_report.md](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/reports/phase4_report.md)
- [artifacts/phase3/metrics_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase3/metrics_summary.json)
- [artifacts/phase4/claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)

## Section Guidance

### Abstract

- Lead with OCR-induced answerability failures in document QA.
- Describe FAAR as a selective recovery controller.
- Report supported cost-control and routing-consistency findings.
- Avoid claiming answer-quality gains over naive OCR-RAG on the current slice.

### Introduction

- Frame the problem around brittle OCR-heavy QA.
- Motivate typed recovery as a control problem, not only a quality problem.
- Preview that current evidence is mixed: strong for control, incomplete for quality lift.

### Methodology

- Map `Q`, `F`, `pi`, and `U` directly to the implementation.
- Describe answer extraction and conservative word-level correction guardrails as practical components of the current system.
- Explain why selective multimodal fallback is triggered only for diagnosed structural cases.

### Experiments

- Report the exact slice size and selection policy.
- Separate benchmark scope from future planned reruns.
- Include ablations and routing rates alongside QA metrics.

### Analysis

- Highlight supported versus unsupported claims explicitly.
- Use case studies to explain where answer extraction and correction still fail.
- Call out EM sensitivity and regression-heavy word-level cases as active engineering risks.

### Conclusion

- Emphasize what the system reliably demonstrates today.
- State that larger or stratified reruns are needed before promoting broader accuracy claims.

## Exit Criteria

- Every headline claim maps to a committed artifact.
- Unsupported claims are written as limitations or future work.
- The draft can be handed to a supervisor without needing verbal caveats to correct overstatement.
