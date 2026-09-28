# FAAR -> AAAI: Full Experimental Plan

Please work top to bottom, do not skip any phases.

## Stack Decisions

Pick one per row before starting.

| Component | Recommended | Alternate A | Alternate B | Free / Local |
| --- | --- | --- | --- | --- |
| VLM | claude-sonnet-4-5 | gpt-4o | Gemini 1.5 Flash | Qwen2-VL-7B |
| Embeddings | NV-Embed-v2 | text-embedding-3-large | Voyage-3 | Nomic-embed-text-v1.5 |
| Reranker | BGE-reranker-v2-m3 | Cohere rerank-3 | Jina-reranker-v2 | FlashRank |
| OCR | GOT-OCR 2.0 | Surya | Nougat | PaddleOCR v4 |
| PDF Pre-processing | Docling | MinerU | LlamaParse | PyMuPDF4LLM |
| Visual RAG Base | ColPali | VisRAG | FLMR | MuRAG |
| Eval Framework | RAGAS | ARES | DeepEval | TruLens |
| Experiment Track | Weights & Biases | LangSmith | MLflow | Comet ML |
| Annotation Tool | Label Studio | Argilla | Prodigy | Doccano |

If compute is limited use this stack, fully open, zero API cost, fully reproducible: Qwen2-VL-7B + Nomic-embed-text-v1.5 + Surya + Docling + ARES.

## Phase 0: Setup

Selected VLM decision for this repository: OpenAI Alternate A using the dated
`gpt-4o-2024-11-20` snapshot. Do not use the floating `gpt-4o` alias.

- Fix train/val/test split on OHR-Bench, save indices to `config/datasets/ohr_split.json`, never change again.
- Add second dataset: MP-DocVQA val set.
- Add third dataset: ArXivQA.
- Swap mock VLM backend with chosen VLM from the stack table above.
- Add API call counter and cost logger to VLM wrapper.
- Write `scripts/experiments/evaluate.py` using RAGAS. It must output `EM`, `F1`, `vlm_rate`, and `harm_rate`.

Commands:

```bash
python scripts/data/split_dataset.py --seed 42 --out config/datasets/ohr_split.json
export VLM_BACKEND=openai
export OPENAI_MODEL=gpt-4o-2024-11-20
export EMBED_MODEL=NV-Embed-v2
export RERANKER=bge-reranker-v2-m3
export LOG_VLM_CALLS=true
export WANDB_PROJECT=faar-aaai
```

## Phase 1: Baselines

Run in this exact order. Save every output before touching FAAR.

```bash
python scripts/experiments/run.py --gate off --recovery off --out results/b0.json
python scripts/experiments/run.py --gate off --recovery always_vlm --out results/b1.json
python scripts/experiments/run.py --gate on --recovery random_type --out results/b2.json
python scripts/experiments/run.py --mode colpali --out results/b3.json
python scripts/experiments/run.py --mode visrag --out results/b4.json
```

If FAAR does not beat B2 by a clear margin, fix the diagnosis module before proceeding.

## Phase 2: Quality Gate Specification

- Use BGE-reranker-v2-m3 score on top-1 retrieved chunk as the gate signal.
- If using local stack, use cosine similarity of top-1 chunk vs query.
- Run theta grid search on validation only for `0.3`, `0.4`, `0.5`, `0.6`, `0.7`.
- Pick theta with best gate-F1 and lock it.
- Target: gate precision >= `0.75`, gate recall >= `0.70`.

## Phase 3: Annotation Study

```bash
python scripts/annotation/sample_failures.py --baseline results/b0.json --n 100 --out annotation/
python scripts/annotation/extract_ocr.py --engine got-ocr-2 --samples annotation/samples.json --out annotation/ocr_texts/
```

Load `annotation/ocr_texts/` into Label Studio. Two people independently label each example as exactly one of `semantic`, `word_level`, `structural`, or `other`. Cohen's kappa must be at least `0.65`.

## Phase 4: Main FAAR Runs

```bash
python scripts/experiments/run.py --mode faar --dataset ohrbench --split test --ocr got-ocr-2 --embed NV-Embed-v2 --reranker bge-reranker-v2-m3 --vlm openai --out results/faar_ohr.json
python scripts/experiments/run.py --mode faar --dataset mpdocvqa --split val --out results/faar_mpdocvqa.json
python scripts/experiments/run.py --mode faar --dataset arxivqa --split val --out results/faar_arxivqa.json
```

Log per run: EM, F1, `vlm_rate`, `cost_usd`, `runtime_sec`.

## Phase 5: Ablations

```bash
python scripts/experiments/run.py --ablate no_gate --out results/a1.json
python scripts/experiments/run.py --ablate no_diagnosis --out results/a2.json
python scripts/experiments/run.py --ablate no_wordlevel_llm --wordlevel_fallback symspell --out results/a3.json
python scripts/experiments/run.py --ablate no_semantic_retry --out results/a4.json
```

## Phase 6: Analysis

- Compute bootstrap confidence intervals.
- Compute per failure type breakdown.
- Compute harm rate.
- Export Plotly cost-accuracy Pareto curve as PDF.

## Phase 7: Paper Tables

- Generate Table 1 main results.
- Generate Table 2 ablation.
- Generate Table 3 per failure type.
- Generate Figure 1 cost-accuracy Pareto curve.

## Before Submitting

```bash
grep -r "haseebraza" .
grep -r "your-real-name" .
pdfinfo paper.pdf | grep Author
```

Checklist:

- [ ] `config/datasets/ohr_split.json` committed to repo
- [ ] `scripts/experiments/evaluate.py` outputs all 4 metrics
- [ ] B0 / B1 / B2 / B3 / B4 all run and saved
- [ ] Gate threshold locked from val set not test set
- [ ] Cohens kappa >= 0.65
- [ ] FAAR run on all 3 datasets
- [ ] All 4 ablations complete
- [ ] Bootstrap 95% CI computed
- [ ] Pareto figure exported as PDF
- [ ] All model versions pinned in paper
- [ ] Repo anonymized, no real GitHub link in PDF

## Model Version Pins

| Role | Recommended |
| --- | --- |
| VLM | gpt-4o-2024-11-20 |
| Embeddings | NV-Embed-v2 |
| Reranker | bge-reranker-v2-m3 |
| OCR | GOT-OCR 2.0 |
| PDF Prep | Docling 1.x |
| Eval | RAGAS 0.2.x |
| Tracking | wandb 0.17.x |
| Annotation | Label Studio 1.x |
