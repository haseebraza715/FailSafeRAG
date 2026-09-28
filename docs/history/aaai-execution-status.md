# Historical: AAAI execution status notes

This file is a dated implementation log, not the current cluster procedure.
Start at [SUPERVISOR_HANDOFF.md](../../SUPERVISOR_HANDOFF.md) and
[the shared-cluster runbook](../operations/runbook.md). The current experiment protocol is
[docs/research/aaai-plan.md](../research/aaai-plan.md).

---

# FAAR AAAI experimental execution plan

This plan mirrors the fixed Phase 0-7 experimental plan. Each phase is a checkpoint: implement, run, verify against the phase success criteria, commit, then continue.

## Phase 0: Setup

- Fix the OHR-Bench train/val/test split with seed 42 and save it to `split.json`.
- Treat `split.json` as immutable after creation.
- Add dataset registry entries for OHR-Bench, MP-DocVQA val, and ArXivQA val.
- Wire the recommended stack through the existing interfaces:
  - VLM: `gpt-4o-2024-11-20` (user-approved Alternate A)
  - Embeddings: `NV-Embed-v2`
  - Reranker: `bge-reranker-v2-m3`
  - OCR: `GOT-OCR 2.0`
  - PDF pre-processing: `Docling`
  - Visual RAG base: `ColPali`
  - Eval framework: `RAGAS`
  - Experiment tracking: `Weights & Biases`
  - Annotation tool: `Label Studio`
- Add API call counting and cost logging before any paid VLM call.
- Add `evaluate.py`, which always emits `EM`, `F1`, `vlm_rate`, and `harm_rate`.
- Verify the setup commands and required outputs.
- Commit as `phase0: complete protocol setup`.

Status (2026-07-23):
- **Implementation complete:** all three dataset registration paths, shared-corpus retrieval, pinned model loaders, Docling adapter, visual baseline dispatch, OpenAI request accounting, four-metric evaluation, and model provenance are implemented. Graph construction under pinned `langgraph==0.3.18` uses node id `quality_gate` while keeping the GraphState/result key `"gate"` (fix commit `1dda8c81842090417610484231b3fb7727a23918`).
- **Locked AAAI environment verified:** CPython 3.12.13 on macOS arm64 in ignored `.venv-aaai/` via `pip install -c constraints-aaai.txt -e '.[aaai]'`. Existing `.venv` preserved. `.venv-aaai/bin/python -m pip check` passes. Required imports succeed. Path-safe package listing is at `results/environment/pip-freeze.txt` with source-code commit `1dda8c81842090417610484231b3fb7727a23918` (raw `pip freeze` must not be committed because editable-install metadata embeds a personal GitHub URL).
- **Locked-env test status:** focused graph/ablation/shared-corpus suite **10 passed**; full `.venv-aaai` suite **84 passed**; legacy `.venv` suite **84 passed**. No dependency pins were changed.
- **Split protected:** `split.json` remains unchanged; SHA-256 is `64583a532c5db5aa31e4cbb5cd9c7d894c7a2d5e8aa49f1a7f6041f54e714f53`.
- **Model cache ready:** all six locked Hugging Face snapshots are downloaded locally at the exact revisions in `config/model_revisions.json`; the selected OpenAI snapshot remains fixed to `gpt-4o-2024-11-20`. No paid experiment or Phase 1 baseline was started.
- **Asset integrity contracts enforced:** manifests must use project-relative paths; loaders reject path traversal; shared corpora require a complete `document_inventory` (evidence pages cannot define retrieval). `prepare_benchmark_assets.py` can plan/execute one OHR document (PDF extract → Docling audit → PNG → pinned GOT-OCR) with resumable provenance.
- **OHR inventory gaps resolved as naming mismatches:** `textbook_needrop_en_*` ↔ `jiaocai_needrop_en_*`; one Springer title is Unicode spelling drift. No fabricated page counts; inventories load from real gt JSON after alias resolution.
- **One-document smoke validated** on val doc `academic/DUDE_157911e3080d18f4d799a122aaeb33fb` (1 page, MPS). GOT-OCR ≈ 24.7 min/page in this run — full val/test prep is compute-bound (~2.9k / ~2.8k OCR-hours at that rate).
- **ArXivQA preparation advanced:** the supervisor-approved fixed subset has 500 QA rows across 497 full papers. All 497 PDFs are downloaded and parsed, and all 8,955 pages are rendered and hash-validated in a preparation-only inventory. Evidence candidates still require human confirmation, and every page still requires pinned GOT-OCR provenance before registration.
- **Still not paper-run ready:** complete OHR val/test OCR assets are not built; MP-DocVQA val still requires an authenticated RRC DocVQA Task 4 download; ArXivQA still requires GPU GOT-OCR and human-confirmed page mappings; no Phase 1 baseline has started.
- **Supervisor correction recorded:** ArXivQA is evaluated by remapping figure-level QA to the original full paper. `remap_arxivqa_full_papers.py` requires explicit QA-to-paper/page mappings and emits a complete-paper source; it rejects missing mappings and never makes evidence pages the retrieval corpus.
- **One-document B0 end-to-end smoke (post asset-prep):** `run_b0_one_doc_smoke.py` loads prepared smoke PDF/OCR/image assets through `BenchmarkRepository`, runs B0 (`naive_rag`: gate off, recovery off), exercises retrieval → quality gate → diagnosis/recovery routing probe → answer generation → serialization, and writes a clearly marked non-paper payload with `EM`, `F1`, `vlm_rate`, `harm_rate`, `cost_usd`, and `runtime_sec`. Default retrieval is a local lexical mock (no HF download, no paid VLM). `--use-pinned-models` loads locked NV-Embed-v2 + bge-reranker-v2-m3 when available. Not a Phase 1 baseline.

## Phase 1: Baselines

- Run B0 text-only RAG, gate off, recovery off.
- Run B1 always-VLM.
- Run B2 random recovery with diagnosis off and random recovery type.
- Run B3 ColPali.
- Run B4 VisRAG.
- Use one worker subagent per independent baseline.
- Wait for all baseline outputs before moving on.
- Verify outputs are saved and FAAR can later be compared against B2.
- Commit as `phase1: baselines`.

## Phase 2: Quality Gate Specification

- Use the BGE-reranker-v2-m3 top-1 score as the gate signal.
- Run the theta grid on the validation split only: `0.3`, `0.4`, `0.5`, `0.6`, `0.7`.
- Log precision, recall, and F1 for each theta.
- Lock the winning theta in config and write it verbatim to `docs/history/runlog.md`.
- Verify gate precision is at least `0.75` and gate recall is at least `0.70`.
- Commit as `phase2: gate threshold`.

Implementation status: complete; execution is pending a real validation B0 result file produced with the pinned stack.

## Phase 3: Annotation Study

- Sample 100 examples where B0 got the wrong answer.
- Extract raw OCR text with GOT-OCR 2.0.
- Prepare Label Studio import assets with exactly the four label choices: `semantic`, `word_level`, `structural`, `other`.
- Stop for real double annotation by humans rather than fabricating labels.
- After labels exist, compute Cohen's kappa and require `kappa >= 0.65`.
- Commit prepared annotation assets as `phase3: annotation setup`.

Implementation status: complete; execution is pending B0 failures, GOT-OCR runtime/model access, and two independent human annotations.

## Phase 4: Main FAAR Runs

- Run FAAR on OHR-Bench test split.
- Run FAAR on MP-DocVQA val split.
- Run FAAR on ArXivQA val split.
- Log EM, F1, `vlm_rate`, `cost_usd`, and `runtime_sec` for each run.
- Verify all result JSON files exist and contain the required metrics.
- Commit as `phase4: main faar runs`.

Implementation status: complete; execution is pending full registered OHR-Bench/MP-DocVQA/ArXivQA OCR and image assets plus the pinned-model credentials.

## Phase 5: Ablations

- Run A1 no gate, always recover.
- Run A2 no diagnosis, random recovery type.
- Run A3 no word-level LLM, SymSpell fallback.
- Run A4 no semantic retry.
- Use one worker subagent per independent ablation.
- Verify all ablation rows are saved.
- Commit as `phase5: ablations`.

Implementation status: complete; execution is pending the locked gate, complete benchmark assets, and pinned-model credentials.

## Phase 6: Analysis

- Compute bootstrap 95% confidence intervals.
- Compute per-failure-type breakdowns for `semantic`, `word_level`, and `structural`.
- Compute harm rate.
- Generate the Plotly cost-accuracy Pareto figure and export it as PDF.
- Verify numbers are loaded from result JSON files.
- Commit as `phase6: analysis`.

Implementation status: complete; execution is pending real saved result JSON files and the Plotly/Kaleido PDF export dependency.

## Phase 7: Paper Tables and Pre-Submission Checklist

- Generate Table 1 main results from result JSON files.
- Generate Table 2 ablations from result JSON files.
- Generate Table 3 per-failure-type results from result JSON files.
- Run the before-submitting checklist verbatim:
  - grep for `haseebraza`
  - grep for `your-real-name`
  - check PDF metadata author field
  - confirm anonymization checklist items
- Commit as `phase7: paper tables`.

Implementation status: complete; tables and Figure 1 require completed Phase 4-6 artifacts. The source-tree anonymization scan is clean except for intentional checklist text and local `.git` metadata; PDF metadata cannot be checked until `paper.pdf` exists and `pdfinfo` is available.
