# Repository Architecture Docs (Legacy)

This folder documents the FAAR repository using a Diataxis-oriented layout:

- Tutorial: learn by doing
- How-to: complete specific tasks
- Reference: exact components and interfaces
- Explanation: design rationale and architecture

## Navigation

Canonical modular docs entrypoint:

- [Docs Home](../index.md)

- [Repository Handbook Index](../repo_handbook/index.md)
- [Architecture Overview](../repo_handbook/architecture_overview.md)
- [Runtime Components](../repo_handbook/runtime_components.md)
- [Phase 2 Methodology Formalization](../phases/phase2/methodology_formalization.md)
- [Phase 2 Code-to-Math Mapping](../phases/phase2/code_math_mapping.md)
- [Phase 2 Acceptance Checklist](../phases/phase2/acceptance_checklist.md)
- [Documentation Style Standard](../repo_handbook/documentation_style_standard.md)

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



