# FAAR — Failure-Aware Agentic Recovery for OCR-RAG
## Supervisor-Aligned Execution Plan

**Research title:** Failure-Aware Agentic Recovery for OCR-Induced Errors in Document QA  
**Target venue:** EMNLP 2026 / ACL Findings / EACL  
**Execution principle:** Build a working system first, then run real experiments, then formalize and write cleanly.

---

## The one-sentence claim

> A failure-aware OCR-RAG system that routes document chunks through typed recovery policies based on OCR quality and failure diagnosis can improve QA accuracy over standard OCR-RAG while using less multimodal cost than always-on visual pipelines.

Everything in the plan should support this claim.

---

## What Changed From The Original Plan

The original plan was a lightweight MVP path. This updated version keeps the completed **Phase 0** work, but upgrades the remaining phases to match supervisor direction:

- keep the manual Phase 0 grounding step
- implement the controller in **LangGraph**
- formalize the framework mathematically
- use a real OCR backend and vector store
- include **quality gate + diagnostic layer + ByT5 correction + selective VLM fallback**
- run stronger benchmark experiments on **OHR-Bench**
- include baselines, retrieval metrics, and ablations

This is now a paper-oriented implementation plan rather than a minimal prototype plan.

---

## Phase 0 — Ground Yourself In The Actual Data
**Status:** Completed  
**Purpose:** Build a real, manually inspected development set before designing the system

### What was done

1. Downloaded and inspected **OHR-Bench**.
2. Created a fixed random sample of **40 QA examples**.
3. Extracted relevant source page(s) for each example.
4. Ran **Tesseract OCR** on the sampled pages.
5. Compared OCR text against clean benchmark page text.
6. Checked whether each answer was recoverable from OCR alone.
7. Manually labeled each example as:
   - `text_corruption`
   - `structure_corruption`
   - `no_issue`

### Phase 0 results

| Label | Count |
|---|---:|
| `no_issue` | 33 |
| `text_corruption` | 5 |
| `structure_corruption` | 2 |

### Interpretation

- OCR failures are real, but not dominant in every random example.
- Harder OCR failures were concentrated in formulas, multilingual pages, and layout-sensitive tables.
- This dev set is useful for grounding, but later detector validation may need a harder stress subset with more structure-heavy examples.

### Deliverables

- [manual_labels.csv](/Users/x/Downloads/Thesis-Paper/Code/data/phase0/manual_labels.csv)
- [sample_manifest.csv](/Users/x/Downloads/Thesis-Paper/Code/data/phase0/sample_manifest.csv)
- raw OCR outputs, clean page text, and rendered page images under `artifacts/phase0/`

Do not discard this phase. It remains the anchor for later detector analysis.

---

## Phase 1 — Build The Working Prototype
**Priority:** Highest  
**Duration:** Immediate next phase  
**Goal:** Produce a working end-to-end system that demonstrates the orchestration

This is now the most important implementation phase.

### Required stack

| Component | Choice |
|---|---|
| Agentic controller | LangGraph |
| OCR backend | Tesseract or PaddleOCR |
| Vector store | FAISS or LanceDB |
| Text retriever | BM25 + dense retrieval |
| Dense embeddings | sentence-transformers or equivalent |
| Text repair module | ByT5 |
| Visual fallback | GPT-4o or Gemini |
| Logging | JSON per example |

### System modules to implement

1. **OCR ingestion**
   - extract page images or PDFs
   - run OCR
   - save raw OCR text

2. **Chunking and indexing**
   - split OCR text into chunks
   - attach `doc_id`, `page_id`, and chunk metadata
   - build vector index and lexical retrieval store

3. **Quality gate**
   - compute chunk-level confidence or quality signals
   - decide whether a chunk passes directly to retrieval/answering or enters diagnosis

4. **Diagnostic layer**
   - classify chunk or retrieved context into failure types
   - working taxonomy for implementation:
     - `semantic`
     - `word_level`
     - `structural`

5. **Recovery actions**
   - `word_level` → ByT5 post-OCR correction
   - `structural` → selective VLM page fallback
   - `semantic` → retrieval refinement or controller retry

6. **Answer generation**
   - answer from retrieved context
   - log path taken by the controller

### Minimum LangGraph flow

```text
Input document/question
    ↓
OCR node
    ↓
Chunk + index node
    ↓
Retrieve node
    ↓
Quality gate Q(c)
    ↓
Diagnostic node F(c)
    ↓
Policy node pi(s)
    ├── direct answer
    ├── ByT5 correction
    ├── retrieval retry / backtrack
    └── VLM page fallback
    ↓
Final answer + logs
```

### Deliverable

- A working LangGraph prototype
- OCR to answer pipeline runs end to end
- Quality gate, diagnostic layer, ByT5 correction, and selective VLM fallback are wired in
- Per-example structured logs exist

Do not over-optimize here. The goal is not elegance first. The goal is to show the orchestration works.

---

## Phase 2 — Formalize The Framework Mathematically
**Priority:** High, but after the prototype is running  
**Goal:** Convert the system logic into a clean methodology section

This phase should be written only after the prototype is concrete enough that the math reflects the actual system.

### Required formal definitions

1. **Quality gate**

Define a decision function:

```text
Q(c): chunk -> {pass, diagnose}
```

with explicit threshold conditions over confidence, noise, retrieval confidence, or layout signals.

2. **Failure taxonomy**

Define:

```text
F(c): chunk -> {semantic, word-level, structural}
```

where:
- `semantic` means the content is insufficient, mismatched, or retrieval-relevant meaning is lost
- `word-level` means OCR corruption damages tokens, characters, or short text spans
- `structural` means layout loss damages answerability, especially for tables, forms, or multi-column pages

3. **Recovery policy**

Define:

```text
pi(s): state -> action
```

or equivalently:

```text
pi(failure_type) -> {answer_direct, correct_text, retry_retrieval, invoke_vlm}
```

4. **Utility function**

Define:

```text
U(a) = E[Accuracy(a)] - lambda * Cost(a)
```

where:
- `a` is an action or recovery strategy
- `E[Accuracy(a)]` is expected answer quality under action `a`
- `Cost(a)` is computational or API cost of that action
- `lambda` is a trade-off hyperparameter controlling accuracy-vs-cost preference

### Deliverable

- Formal notation for the paper methodology
- Notation aligned with the actual implementation
- Clean mapping between theory and system modules

---

## Phase 3 — Run Real Experiments
**Priority:** Very high  
**Goal:** Fill the biggest gap: evidence

This phase determines whether the work is actually publishable.

### Main dataset

Use **OHR-Bench (ICCV 2025)** as the main benchmark.

### Systems to evaluate

At minimum:
- **Naive RAG**
- **FAAR (ours)**

Supervisor-requested reference baselines:
- **CRAG**
- **Self-RAG**
- **VisRAG**

Practical note:
- If reproducing every baseline fully is too heavy, document what is reproduced directly, what is approximated, and what uses reported settings.
- But do not skip the comparison discussion.

### Metrics

Report:
- **NDCG@5**
- **Recall@5**
- **Exact Match (EM)**
- **F1**

Optional but still useful:
- latency
- visual fallback rate
- cost per query

### Minimum ablations

Run at least these three:

1. **Remove History-Aware Backtracking**
2. **Disable Visual Fallback**
3. **Remove Diagnostic Layer**

If implementation complexity forces a rename, keep the ablation logic equivalent and note it clearly.

### Additional analysis to include

- performance by failure type
- where ByT5 helps and where it does not
- where VLM fallback triggers
- recovery success rate of visual fallback
- examples of false positives and false negatives from the diagnostic layer

### Deliverable

- Benchmark results table
- Retrieval metrics table
- Ablation table
- Error analysis examples

---

## Phase 4 — Refine Experimental Design And Claims
**Goal:** Make sure the story supported by the experiments is precise and defensible

### Questions this phase must answer

1. Does the quality gate reduce unnecessary expensive recovery?
2. Does typed recovery outperform naive direct fallback?
3. Does selective visual fallback get closer to visual-RAG performance at lower cost?
4. Do the diagnostic categories correspond to real observed errors?

### What to check

- whether `structural` predictions actually benefit from VLM fallback
- whether `word_level` cases actually benefit from ByT5 correction
- whether `semantic` cases are mostly retrieval/controller failures rather than OCR corruption

### Deliverable

- tightened experimental claims
- refined thresholds
- a small number of clear case studies for the paper

---

## Phase 5 — Write The Paper Properly
**Goal:** Convert implementation + experiments into a conference-ready paper structure

Use this structure:

1. **Abstract**
2. **Introduction**
3. **Related Work**
4. **Methodology**
5. **Experiments**
6. **Analysis**
7. **Conclusion**

### Writing guidance by section

**Abstract**
- state the problem
- state the method
- state the key result
- state the efficiency/accuracy trade-off

**Introduction**
- motivate OCR failures in document QA
- explain why generic RAG is not enough
- position selective recovery as the contribution

**Related Work**
- OCR-aware QA
- agentic RAG
- document understanding
- retrieval correction / self-refinement

**Methodology**
- formalize `Q`, `F`, `pi`, and `U`
- describe LangGraph controller and system modules
- explain quality gate, diagnosis, recovery actions

**Experiments**
- dataset
- baselines
- metrics
- implementation details
- main results

**Analysis**
- ablations
- failure breakdown
- recovery success analysis
- qualitative examples

**Conclusion**
- summarize what works
- note limitations honestly
- suggest future work

### Deliverable

- Supervisor-ready draft
- Figures: architecture diagram, result tables, ablations

---

## Updated Contribution Framing

If the system works as intended, the contributions can be framed as:

1. A **failure-aware OCR-RAG controller** implemented with LangGraph
2. A formal **quality gate and typed failure taxonomy** for OCR-RAG recovery
3. A selective recovery pipeline combining **ByT5 correction** and **page-level VLM fallback**
4. An empirical study on **OHR-Bench** showing when typed recovery improves QA accuracy and when it is worth the cost

---

## What Not To Lose From The Original Plan

Do not lose these strengths from the earlier roadmap:

- keep Phase 0 as the grounding step
- keep the page-level, not full-document, visual fallback idea
- keep logging and per-example analysis
- keep the habit of measuring before complicating the system

This upgraded plan is more ambitious, but it should still be executed with the same discipline.

---

## Immediate Next Steps

### Step 1
Build the **Phase 1 working prototype** first:
- LangGraph
- OCR
- indexing
- retrieval
- quality gate
- diagnostic layer
- ByT5
- VLM fallback

### Step 2
Once the system runs, formalize:
- `Q(c)`
- `F(c)`
- `pi(s)`
- `U(a)`

### Step 3
Then run experiments on OHR-Bench and start filling result tables.

This order is important:

```text
implementation first
    ↓
experiments second
    ↓
formalization and writing third
```

That is the safest path to a paper-quality result without getting stuck in planning.

---

## Final Honest Advice

The project now has stronger requirements, but it is also more aligned with a publishable paper. The main risk is trying to build every idea at once.

Move forward by treating the updated prototype as the backbone:
- make it run
- measure it
- refine it
- then formalize and write

Phase 0 is already done. The next real milestone is a working Phase 1 controller-based system.
