# Architecture Overview

## System Goal

Improve OCR-heavy document QA quality while controlling multimodal cost through selective, typed recovery.

## High-Level Pipeline

```mermaid
flowchart TD
    input[QuestionAndExample] --> load[LoadPhase0Example]
    load --> chunk[ChunkAndMetadata]
    chunk --> retrieve[HybridRetrieve]
    retrieve --> gate[QualityGate]
    gate -->|pass| answerDirect[AnswerDirect]
    gate -->|diagnose| diagnose[FailureDiagnosis]
    diagnose --> policy[RecoveryPolicy]
    policy --> answerRecovered[RecoveredAnswer]
    answerDirect --> log[StructuredLog]
    answerRecovered --> log
```



## Why This Design

- quality gate separates easy from risky cases early
- typed diagnosis supports action specialization
- structured logs enable per-example analysis and reproducibility

