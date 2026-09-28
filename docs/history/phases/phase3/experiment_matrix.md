# Phase 3 Experiment Matrix

## Dataset Scope

- Source: OHR-Bench examples available through `data/phase0/sample_manifest.csv`
- Phase 3 execution slice: all examples available in manifest by default
- Optional control: run on first `N` examples via CLI limit flag for quick validation

## System Scope (Locked)

This implementation round includes:

1. `naive_rag`
2. `faar_full`
3. `faar_no_backtrack`
4. `faar_no_vlm`
5. `faar_no_diagnosis`

No external baseline approximation is included in this round by design.

## Comparison Protocol

- Same example set across all profiles
- Same metric definitions and normalization pipeline
- Same seed handling policy per profile run

## Output Policy

- Per-example outputs under `logs/phase3/<profile>/`
- Aggregated profile outputs under `artifacts/phase3/`
