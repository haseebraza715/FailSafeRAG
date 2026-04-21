# Claim-Safe Outline

## Recommended Thesis/Paper Positioning

Use language of this form:

- FAAR is a failure-aware OCR-RAG controller that selectively applies recovery actions when quality signals indicate likely answerability failure.
- On the current evaluation slice, FAAR demonstrates controlled visual-fallback usage and diagnosis-policy consistency.
- Current evidence does not yet establish answer-quality gains over the naive baseline, so quality-improvement claims remain provisional.

## Claims To Keep

- selective recovery reduces always-on multimodal usage;
- typed diagnosis maps consistently onto executed actions;
- the implementation is reproducible, testable, and artifact-backed.

## Claims To Avoid For Now

- FAAR improves answer quality over naive OCR-RAG in general;
- selective VLM fallback improves quality on the present benchmark slice;
- word-level correction is broadly beneficial across OCR-corruption cases.

## Safe Contribution Framing

1. A working failure-aware OCR-RAG controller with typed recovery routing.
2. A concrete mapping from quality gate and failure taxonomy to executable policy actions.
3. An evidence-backed analysis showing where selective recovery is operationally useful and where quality gains remain unresolved.

## Writing Checklist

- Name the slice size every time results are summarized.
- Pair every positive result with the relevant metric and artifact.
- State unsupported findings directly instead of hiding them in limitations.
- Keep future-work language separate from achieved results.
