# Phase 3 Error Analysis

## Source

Examples are sampled from `artifacts/phase3/error_analysis_examples.json`.

## What Was Inspected

- profile path (`profile`)
- predicted failure type (`failure_type`)
- selected policy (`policy_action`)
- predicted answer vs gold answer

## High-Level Observations

- `EM` remains low across current profiles, indicating extractive answering remains the main bottleneck.
- Retrieval-oriented metrics (`NDCG@5`, `Recall@5`) are materially higher than exact answer match.
- This suggests the pipeline often retrieves relevant context without producing exact final spans.

## Actionable Next Steps

- strengthen answer generation module for exact-match-sensitive questions
- expand Phase 3 run slice for more stable failure-type distributions
- add per-failure-type confusion and recovery-success counts in next reporting pass
