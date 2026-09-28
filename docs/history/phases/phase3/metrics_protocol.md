# Phase 3 Metrics Protocol

## Retrieval Metrics

### Recall@5

Definition:

- `Recall@5 = 1` if any top-5 retrieved chunk contains normalized gold answer string
- `Recall@5 = 0` otherwise

Reported value is mean across examples.

### NDCG@5

Definition:

- Relevance at rank `i` is binary (`1` if chunk contains normalized gold answer, else `0`)
- `DCG@5 = sum((2^rel_i - 1) / log2(i + 1))`
- `IDCG@5` computed from ideal ordering of same relevance vector
- `NDCG@5 = DCG@5 / IDCG@5` (0 if `IDCG@5 = 0`)

Reported value is mean across examples.

## QA Metrics

### Exact Match (EM)

- Compare normalized prediction and normalized gold string.
- `EM = 1` if exact normalized match, else `0`.

### Token F1

- Tokenize normalized prediction and gold.
- Compute overlap precision and recall.
- `F1 = 2PR/(P+R)` with zero-safe handling.

Reported value is mean across examples.

## Normalization Rules

Normalization used consistently for retrieval and QA checks:

- lowercase
- trim leading/trailing spaces
- collapse internal whitespace
- remove punctuation except alphanumerics and spaces

## Reporting

Required profile-level outputs:

- `NDCG@5`
- `Recall@5`
- `EM`
- `F1`
- example count
