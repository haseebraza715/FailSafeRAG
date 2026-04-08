# FAAR Documentation Style Standard

## Purpose

This standard defines how FAAR documents are written, structured, and evidenced across all phases.

## Required Structure For Phase Documents

1. Purpose
2. Scope
3. Inputs and Dependencies
4. Method or Process
5. Evidence and Artifacts
6. Validation
7. Risks and Limitations
8. Next Actions

## Writing Rules

- Use short paragraphs and flat bullet lists.
- Keep terminology consistent with code labels (`word_level`, `invoke_vlm`, `retry_retrieval`).
- Distinguish measured facts from interpretation.
- Link each completion claim to evidence in `logs/`, `artifacts/`, `src/`, or `tests/`.

## Math and Notation Rules

- Include a symbol table before equations.
- Keep one concept per equation block.
- Explain each equation in plain language immediately below it.
- Map symbols to implementation anchors.

## Diagram Rules

- Include one pipeline diagram in architecture/method docs.
- Include one decision diagram in routing/policy docs.
- Avoid decorative-only diagrams.

## Evidence Rules

Allowed status labels:

- `Complete`
- `In progress`
- `Planned`

Do not mark `Complete` without implementation and validation evidence.

## Templates

- Phase 3 experiment report template
- Phase 4 claim-refinement memo template
- Phase 5 paper drafting support template

Canonical template details are maintained in phase docs under `docs/phases/`.