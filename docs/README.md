# Documentation index

## Current status

The FAAR study is finished. Diagnosis-based repair did not beat the gated image repair on the final test (26.6%
against 31.5% exact match on 946 multi-page test questions). Start with the
[FAAR study report](reports/faar-study-report.md).

## Cluster work

The FAAR study used the OpenAI API and no cluster. The cluster commands from the earlier plan were never submitted.
Their procedure is in the [supervisor handoff](../SUPERVISOR_HANDOFF.md) and the
[shared-cluster runbook](operations/runbook.md).

## Current

| Path | Contents |
| --- | --- |
| [reports/faar-study-report.md](reports/faar-study-report.md) | The whole FAAR study: method, results of every stage, limitations |
| [research/faar-vs-gated-preregistration.md](research/faar-vs-gated-preregistration.md) | Pre-registration: policies, decision rule, frozen method, test plan |
| [research/faar-practice-rounds.md](research/faar-practice-rounds.md), [faar-confirmation.md](research/faar-confirmation.md), [faar-final-test.md](research/faar-final-test.md) | Stage reports with run ids and costs |
| [research/study-brief.md](research/study-brief.md) | Study design the pre-registration builds on: question, comparisons, scoring |
| [research/repair-analysis-plan.md](research/repair-analysis-plan.md), [visual-repair-design.md](research/visual-repair-design.md), [ohr-dev-v2-proposal.md](research/ohr-dev-v2-proposal.md), [longer-document-sample-options.md](research/longer-document-sample-options.md) | Development plans and decisions that led to the pre-registration, kept as records |
| [research/aaai-plan.md](research/aaai-plan.md) | Earlier AAAI plan, kept as evidence; not the current protocol |
| [research/aaai-reproducibility.md](research/aaai-reproducibility.md) | Environment, model pins, and identity checks |
| [architecture/overview.md](architecture/overview.md) | The pipeline the study ran, step by step |
| [operations/runbook.md](operations/runbook.md) | Shared-cluster procedure |
| [reports/](reports/index.md) | The study report, plus the data audit, pilot readiness and pre-baseline engineering reports |
| [../experiments/README.md](../experiments/README.md) | Run registry and run-record format |
| [../paper/README.md](../paper/README.md) | Current paper source and table/figure provenance |
| [../cluster/README.md](../cluster/README.md) | Launcher, templates, and scheduler mechanics |
| [../annotation/README.md](../annotation/README.md) | Label study commands; the label study was not run |
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
