# Historical: repository architecture docs

Leftover Diataxis layout from an earlier docs migration. Current navigation is
[docs/README.md](../../README.md). For cluster work start at
[SUPERVISOR_HANDOFF.md](../../../SUPERVISOR_HANDOFF.md).

- Tutorial: learn by doing
- How-to: complete specific tasks
- Reference: exact components and interfaces
- Explanation: design rationale and architecture

## Navigation

Canonical modular docs entrypoint:

- [Docs Home](/docs/README.md)

- [Repository Handbook Index](/docs/history/repo_handbook/index.md)
- [Architecture Overview](/docs/history/repo_handbook/architecture_overview.md)
- [Runtime Components](/docs/history/repo_handbook/runtime_components.md)
- [Phase 2 Methodology Formalization](/docs/history/phases/phase2/methodology_formalization.md)
- [Phase 2 Code-to-Math Mapping](/docs/history/phases/phase2/code_math_mapping.md)
- [Phase 2 Acceptance Checklist](/docs/history/phases/phase2/acceptance_checklist.md)
- [Documentation Style Standard](/docs/history/repo_handbook/documentation_style_standard.md)

## Documentation Map

```mermaid
flowchart TD
    index[Index]
    tutorial[Tutorial]
    howto[HowTo]
    reference[Reference]
    explanation[Explanation]
    index --> tutorial
    index --> howto
    index --> reference
    index --> explanation
```


