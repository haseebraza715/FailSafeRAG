# Documentation index

Current cluster work starts at the repository root:

- [Supervisor handoff](../SUPERVISOR_HANDOFF.md)
- [Shared-cluster runbook](operations/runbook.md)

Those two files are the operational source of truth.

## Current

| Path | Contents |
| --- | --- |
| [research/study-brief.md](research/study-brief.md) | Current study design: question, comparisons, scoring and decisions |
| [research/aaai-plan.md](research/aaai-plan.md) | Earlier AAAI plan, kept as evidence; not the current protocol |
| [research/aaai-reproducibility.md](research/aaai-reproducibility.md) | Environment, model pins, and identity checks |
| [architecture/overview.md](architecture/overview.md) | Current system design |
| [operations/runbook.md](operations/runbook.md) | Shared-cluster procedure |
| [reports/](reports/index.md) | Current data audit and pilot readiness reports |
| [../experiments/README.md](../experiments/README.md) | Run registry and run-record format |
| [../paper/README.md](../paper/README.md) | Current paper source and table/figure provenance |
| [../cluster/README.md](../cluster/README.md) | Launcher, templates, and scheduler mechanics |
| [../annotation/README.md](../annotation/README.md) | Label study commands, after a real B0 exists |
| [evidence/evidence-manifest.tsv](evidence/evidence-manifest.tsv) | Provenance of committed prototype assets |

## Historical

[history/](history/README.md) holds superseded plans, the prototype phase
records and reports (including the 40-example mock numbers), the August 2026
status report, and the record of the 2026-09-28 reorganisation with its
old-to-new path map. Do not use these as cluster instructions.

## Command-line tools

User-facing commands are grouped under `scripts/` by purpose: experiments,
data preparation, annotation, smoke checks, and release checks. Cluster launchers
and scheduler templates remain under `cluster/`.
