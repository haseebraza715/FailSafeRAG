# Runtime Components

## Data Access

- `src/faar/data.py`: loads phase assets and returns `Phase0Example`.

## Chunking

- `src/faar/chunking.py`: creates overlapping chunk windows with metadata.

## Retrieval

- `src/faar/retrieval.py`: BM25 + dense FAISS retrieval with fused ranking.

## Quality + Diagnosis

- `src/faar/quality.py`: computes quality score and failure type.

## Recovery

- `src/faar/recovery.py`:
  - `word_level` -> ByT5 correction
  - `semantic` -> retrieval retry with semantic backtrack
  - `structural` -> selective visual fallback

## Orchestration

- `src/faar/graph.py`: LangGraph state machine routing by gate + diagnosis.

## Answering

- `src/faar/answering.py`: extractive answer heuristic over context hits.

## Logging and CLI

- `src/faar/cli.py`: emits JSON logs with gate, diagnosis, policy, action outcome, and reproducibility metadata.
