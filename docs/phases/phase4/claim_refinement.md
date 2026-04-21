# Phase 4 Claim Refinement Decisions

## Source

Primary artifact:

- [artifacts/phase4/claim_assessment.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/claim_assessment.json)

Supporting artifact:

- [artifacts/phase4/phase4_summary.json](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/artifacts/phase4/phase4_summary.json)

## Decision Table

| Question | Status | Evidence | Decision |
| --- | --- | --- | --- |
| Does the quality gate reduce unnecessary expensive recovery? | `supported` | `faar_full` visual fallback rate = `0.075` | Keep selective fallback routing; cost-control objective is met. |
| Does typed recovery outperform naive direct fallback? | `not_supported` | `delta_f1_vs_naive_rag = -0.0095`, `delta_em_vs_naive_rag = 0.0` | Do not claim quality gain over naive baseline in current draft. |
| Does selective VLM fallback improve quality relative to no VLM at lower cost? | `not_supported` | `delta_f1_vs_faar_no_vlm = 0.0`, visual fallback used at `0.075` | Keep as cost-control mechanism only; remove quality-improvement claim. |
| Do diagnostic categories align with observed routing behavior? | `supported` | diagnosis-policy consistency = `1.0`, non-pass rate = `0.475` | Keep typed diagnosis framing as operationally valid. |

## Claim Language Update

Use this safer claim framing for next draft:

- FAAR currently demonstrates **controlled selective recovery cost behavior** and **internally consistent typed diagnosis routing**.
- Current run does **not yet show answer-quality gains** over naive baseline on this benchmark slice.

## Recommended Next Phase 4 Follow-ups

1. Improve answer generation quality (current EM is zero across compared profiles).
2. Revisit word-level correction behavior where case studies show regressions.
3. Expand and stratify evaluation slices before promoting quality-improvement claims.

## Phase 5 Handoff

Phase 5 writing should treat the current evidence as:

- strong support for selective-cost control and policy-consistent routing;
- insufficient support for quality-improvement claims over the naive baseline on the current 40-example slice;
- motivation for larger or stratified reruns before thesis-level claim escalation.

Use [Phase 5 Writing Plan](https://github.com/haseebraza715/failure-aware-ocr-rag/blob/main/docs/phases/phase5/writing_plan.md) as the default handoff document for drafting.
