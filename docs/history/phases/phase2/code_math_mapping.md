# Phase 2 Code-to-Math Mapping

## Purpose

This document maps formal methodology symbols to current FAAR implementation anchors.

## Mapping Table

| Formal object | Meaning | Implementation anchor | Current status |
| --- | --- | --- | --- |
| `Q(c)` | quality gate decision (`pass` or `diagnose`) | `src/faar/quality.py::quality_gate` | Implemented |
| `w(c)` | OCR token/character corruption score | `src/faar/quality.py::weird_char_ratio` | Implemented |
| `s(c)` | layout-structure signal count | `src/faar/quality.py::layout_signals` + `quality_gate` | Implemented |
| `F(c)` | failure type (`semantic`, `word_level`, `structural`) | `src/faar/quality.py::diagnose_failure` | Implemented |
| `pi(s)` | policy decision from gate/diagnosis state | `src/faar/graph.py::route_after_gate`, `diagnose_node`, `route_after_diagnosis` | Implemented |
| `a=answer_direct` | direct answer path | `src/faar/graph.py::direct_answer` | Implemented |
| `a=correct_text` | word-level recovery | `src/faar/graph.py::word_level_recovery`, `src/faar/recovery.py::ByT5Corrector.correct` | Implemented |
| `a=retry_retrieval` | semantic recovery via backtrack | `src/faar/graph.py::semantic_recovery`, `src/faar/recovery.py::semantic_backtrack` | Implemented |
| `a=invoke_vlm` | structural recovery via visual fallback | `src/faar/graph.py::structural_recovery`, `src/faar/recovery.py::VisualFallback.answer` | Implemented |
| `U(a)` | expected accuracy minus weighted cost | `docs/phases/phase2/methodology_formalization.md` | Defined; measured in Phase 3 |

## Implemented Now vs Planned Next

Implemented now:

- deterministic gate/diagnosis/policy execution
- typed recovery actions and explicit action outcomes
- reproducibility metadata and threshold logging

Planned for Phase 3:

- empirical estimation of `E[Accuracy(a)]` on benchmark metrics
- action-conditioned cost table using measured latency/API usage
- utility-based comparative analysis across baselines and ablations
