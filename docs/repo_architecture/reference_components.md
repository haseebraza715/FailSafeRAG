# Component Reference

## Ingestion and Data Access

- Source module: `src/faar/data.py`
- Primary responsibility: load Phase 0 manifest/summary and return `Phase0Example`.

```mermaid
flowchart LR
    manifest[sample_manifest.csv]
    summary[phase0_asset_summary.json]
    ocr[artifacts phase0 ocr_text]
    repo[Phase0Repository]
    example[Phase0Example]
    manifest --> repo
    summary --> repo
    ocr --> repo
    repo --> example
```



## Chunking

- Source module: `src/faar/chunking.py`
- Splits OCR page text into overlapping chunks with IDs and page metadata.

```mermaid
flowchart LR
    pageText[OCRPageText]
    tokenize[TokenizeWords]
    window[SlidingWindow]
    chunks[ChunkObjects]
    pageText --> tokenize --> window --> chunks
```



## Retrieval and Embedding

- Source module: `src/faar/retrieval.py`
- Hybrid retriever combining BM25, dense retrieval, and fused scoring.

```mermaid
flowchart TD
    query[Query]
    bm25[BM25Score]
    dense[DenseScore FAISS]
    fuse[FusedRanking]
    hits[TopK RetrievalHit]
    query --> bm25
    query --> dense
    bm25 --> fuse
    dense --> fuse
    fuse --> hits
```



## Quality Gate and Diagnostics

- Source module: `src/faar/quality.py`
- Computes quality score and routes to failure taxonomy.

```mermaid
flowchart TD
    hit[TopHitTextAndScores]
    signals[NoiseAndLayoutSignals]
    qscore[QualityScore]
    diag[FailureType]
    hit --> signals --> qscore --> diag
```



## Recovery

- Source module: `src/faar/recovery.py`
- Implements typed recovery actions:
  - `word_level`: ByT5 correction
  - `semantic`: query backtrack/retry
  - `structural`: selective visual fallback

```mermaid
flowchart LR
    failure[FailureType]
    byt5[ByT5Correct]
    retry[SemanticRetry]
    vlm[VisualFallback]
    failure -->|word_level| byt5
    failure -->|semantic| retry
    failure -->|structural| vlm
```



## Answering

- Source module: `src/faar/answering.py`
- Uses extractive overlap heuristic over retrieved/recovered context.

```mermaid
flowchart LR
    q[Question]
    hits[CandidateSentences]
    score[OverlapScoring]
    answer[BestAnswerSpan]
    q --> score
    hits --> score --> answer
```



## Controller and Orchestration

- Source module: `src/faar/graph.py`
- LangGraph state machine connecting all modules with conditional routing.

```mermaid
flowchart TD
    load[LoadExample]
    prep[PrepareRetrieval]
    ret[Retrieve]
    gate[Gate]
    diag[Diagnose]
    direct[AnswerDirect]
    recover[AnswerRecovered]
    load --> prep --> ret --> gate
    gate -->|pass| direct
    gate -->|fail| diag --> recover
```



## Logging and CLI

- Source module: `src/faar/cli.py`
- Produces per-run JSON output with:
  - controller path
  - action outcome status
  - run metadata for reproducibility

```mermaid
flowchart LR
    run[CLI Run]
    state[GraphResultState]
    payload[JSONPayload]
    log[logs phase1]
    run --> state --> payload --> log
```



