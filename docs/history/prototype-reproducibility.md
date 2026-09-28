# Historical: prototype reproducibility notes

This page describes the older offline prototype path (`local-hash-v1`,
`constraints-py312.txt`). Paper runs follow
[docs/research/aaai-reproducibility.md](../research/aaai-reproducibility.md).

---

# Reproducibility and asset provenance

## Runtime

The verified target is Python 3.12 with `constraints-py312.txt`. Offline mock
runs use `local-hash-v1`, a deterministic 256-dimensional feature hasher in
`src/faar/retrieval.py`; it performs no model or network lookup. Research runs
may opt into `sentence-transformers/all-MiniLM-L6-v2` at Hugging Face revision
`8b3219a92973c328a8e22fadcfa821b5dc75636a`. The backend and revision must be
recorded with each new run.

## OHR-Bench

`OHR-Bench/` is a vendored research benchmark snapshot, not original FAAR
code. Upstream project: OpenBMB/OHR-Bench, paper/data/code released for
research use. The exact imported upstream commit was not recorded in the
original checkout, so publication must describe its provenance as
**incomplete** until the owner supplies that commit and confirms the upstream
license. No nested benchmark data or result has been rewritten during cleanup.

## Phase assets

`data/phase0/`, `logs/`, and `artifacts/` contain sampled inputs, manual labels,
and derived offline evidence. Their per-file Git object ids and classifications
are listed in `docs/evidence/evidence-manifest.tsv`. New experiment runs must use versioned
destinations and must never overwrite a cited artifact.

## Capacity bounds

`RetrievalSettings.max_chunks` defaults to 10,000 and rejects larger corpora
before allocating dense vectors. `embedding_batch_size` defaults to 64. Raise
these only after recording peak memory/runtime for the target corpus.
