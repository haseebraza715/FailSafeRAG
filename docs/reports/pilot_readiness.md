# Pilot readiness: OHR development pilot `ohr_dev_v1`

Built 2026-09-28 from the locked QA source, split and PDF archive, and from
the asset audit with content sha256
`7bb6df50f543f2c4130409f46e98865a34d2316e7c7e6ceba4e87f0c2557e04b`. The
frozen selection digest is
`2e2d5061debfb935eb5cdea27cea61273c8064b308e5af5f018d351f458cdd92`.

This is a development pilot for mechanism work. It is not a final benchmark and
not an unseen-publication evaluation. Nothing here measures answer quality,
diagnosis accuracy or repair effectiveness. No model, OCR, retrieval or answer
generation was run.

## Reproduce

```bash
.venv-aaai/bin/python scripts/data/audit_assets.py --project-root . --out-dir results/data_audit --pdf-zip data/ohr_bench_raw/pdfs.zip
.venv-aaai/bin/python scripts/data/build_pilot.py --project-root . --config config/pilots/ohr_dev_v1.json
```

The builder refuses to run when the QA source, split, archive or text-tree
hashes differ from the audit or the lock (exit 1). When a frozen record
already exists, it validates the record instead of rewriting it (`"status":
"reproduced"`). If the config or inputs would produce a different pilot, it
exits 3 and requires a new `pilot_id` and `output_dir`. It never overwrites an
existing `annotations.csv`. `--no-render` freezes or validates the selection
without rendering.

| Artifact | Contents |
| --- | --- |
| `config/pilots/ohr_dev_v1.json` | Seed, size, pool, stratification, inspection budget |
| `results/pilots/ohr_dev_v1/selection_record.json` | Algorithm, pool, allocation, exclusions, composition, source families, source hashes, manifest digests; volatile metadata under `volatile` |
| `results/pilots/ohr_dev_v1/runtime_manifest.json` | Question text, document identity, PDF member and hash, complete PDF page inventory with per-page MinerU status and text hash |
| `results/pilots/ohr_dev_v1/evaluation_manifest.json` | Gold answers, evidence pages and contexts, gt references, analysis flags |
| `results/pilots/ohr_dev_v1/inspection/` | `index.html`, `annotations.csv`, `inspection_cases.json`, `render_record.json`, `pages/` (git-ignored; regenerated) |

The runtime manifest contains no answers, evidence labels, evidence contexts,
gt text or gt paths. The builder enforces this, and the tests check it.

## What was selected and why

- **Identity.** A document's canonical ID is the NFC form of the gt inventory key its QA name resolves to. The resolutions are those the audit recorded: exact, the documented `textbook_needrop → jiaocai_needrop` alias, or Unicode NFC.
- **Pool.** Documents all of whose questions are in the train split: 343 documents. That matches the audit's train-exclusive count.
- **Eligibility.** A document is eligible when it has:
  - one raw QA name,
  - exactly one source PDF candidate with audit status `verified_complete`, and
  - a gt reference for evaluation.

  Missing or empty OCR never excludes a document. All 343 are eligible, with no source exclusions.
- **Selection.** Seed 42, 30 documents, stratified by document type:
  1. Each available type gets one slot.
  2. The remaining slots go by largest remainder against proportional quotas. Ties break by larger pool, then type name.
  3. Within a type, documents are taken in ascending `sha256(seed, "select", doc_id)` order. This does not depend on input order.

  Only identity, type and source status were used, never answers, evidence strings, string-match results or model outputs. All 70 train questions of the selected documents are included.

| Type | Eligible | Proportional quota | Selected |
| --- | ---: | ---: | ---: |
| academic | 10 | 0.87 | 1 |
| administration | 22 | 1.92 | 2 |
| finance | 4 | 0.35 | 1 |
| law | 12 | 1.05 | 1 |
| manual | 18 | 1.57 | 1 |
| news | 89 | 7.78 | 8 |
| textbook | 188 | 16.44 | 16 |

The selection was frozen before any page was rendered. It was not resampled.

One correction happened after freezing. After the first freeze I added a second
source-family naming rule (below). That changed only the descriptive family
section of the record, and the builder's frozen check refused the new record
(exit 3). Because that first record was minutes old and had not been used, I
replaced it. The selected documents, question IDs and manifest digests are
identical, and the earlier record is kept in the working scratchpad.

## Composition (a description of the sample, not results)

| | Eligible pool | Pilot |
| --- | ---: | ---: |
| Documents / questions / pages | 343 / 692 / 715 | 30 / 70 / 54 |
| Documents by page count: 1 / 2–5 / 6–20 / 21–50 | 277 / 43 / 22 / 1 | 23 / 5 / 2 / 0 |
| Evidence type: text / table / formula / reading order / multi / chart | 483 / 73 / 75 / 57 / 4 / 0 | 42 / 11 / 3 / 12 / 2 / 0 |
| MinerU pages ok / empty / missing | 582 / 131 / 2 | 48 / 6 / 0 |
| Documents with empty / missing MinerU pages | 105 / 2 | 6 / 0 |
| Questions whose evidence page has empty MinerU text | 95 | 6 |
| Questions containing Han characters | 109 | 22 |
| List-valued evidence (all also multi-page) | 46 | 9 |

In the pilot, the 6 empty MinerU pages are the single pages of four GNHK
documents and two `omnidocbench_notes` documents. The one I viewed (case 1) is
handwriting. That is an observation, not a diagnosis of why the text is
empty.

**What this sample represents.** A proportional draw from train-exclusive
documents. That pool differs sharply from the benchmark as a whole:

- 277 of its 343 documents are single pages.
- It contains no chart questions.
- Multi-page documents appear mostly in the validation and test splits, which is why they are rare here.

In the pilot, 23 of 30 documents are single pages, and only 7 have more than
one page (2, 2, 2, 3, 5, 8 and 9 pages). Within-document retrieval is therefore
trivial for most pilot questions. The pilot can exercise word-level and
structural repair, but it gives little room to test semantic retrieval retry.
See open decision 1.

## Source-family overlap (uncertain)

The families here come from naming conventions, not upstream metadata:

- `<publication>.pdf_<page>` is a page excerpt of one publication.
- `omnidocbench_<collection>_<32-hex id>_<page>` is a page of one source.

No other grouping was attempted, for example company filings or GNHK writers.

- 12 pilot documents fall into 10 such families.
- The three `omnidocbench_notes_1ba14cb3…` pilot documents (pages 74, 121 and 124) come from one source. They are not independent.
- Three families are shared with validation or test:
  - `omnidocbench_notes_1ba14cb3…`: 3 pilot documents, 12 validation/test documents
  - `omnidocbench_notes_f7f010b7…`: 1 pilot document, 9 validation/test documents
  - `omnidocbench_newspaper_2a6b4fa0…`: 1 pilot document, 9 validation/test documents

Using the pilot for tuning could therefore carry information about those
publications into validation and test. The split was not changed.

## Inspection packet

- **Sample.** 20 cases drawn only from the pilot, as a purposive sample; its proportions are not prevalence estimates.
- **Procedure.** Questions are ranked by `sha256(seed, "inspect", question_id)`. Category quotas are filled in this order:
  1. OCR gap
  2. multi-page evidence
  3. multi evidence source
  4. Han script
  5. table
  6. formula
  7. chart
  8. reading order
  9. text

  Remaining slots are topped up in rank order. A case whose new pages would exceed the 60-page budget would be skipped for the next-ranked one; none needed to be.
- **Coverage (cases covering / available in pilot):**

  | Category | Covered / available |
  | --- | ---: |
  | OCR gap | 3 / 6 |
  | multi-page | 4 / 9 |
  | multi source | 2 / 2 |
  | Han | 4 / 22 |
  | table | 3 / 11 |
  | formula | 2 / 3 |
  | reading order | 2 / 12 |
  | text | 11 / 42 |
  | chart | unavailable (0 in the pilot and the pool) |

  Categories overlap.
- **Shown per case:** question, document and PDF member hash, the evidence pages with page_idx and PDF page number, MinerU text or an explicit missing/empty state, and gold labels and gt text in collapsed "evaluation only" sections.
- **Evidence-page caveat.** The packet chooses pages using gold evidence labels, and it says so at the top. This is not a retrieval result and must not become a runtime shortcut.
- **Annotation.** `annotations.csv` has one blank row per case and page, with the five observation categories plus uncertainty and notes columns. No field is pre-filled.

**Rendering (measured).**

- **Renderer:** pypdfium2 4.30.0 (PDFium build 6462) and Pillow 10.4.0.
- **Format:** 150 DPI PNG. 150 DPI is an inspection choice, not a model-input setting.
- **Volume:** 17 unique pages, 31,555,437 bytes in total, 251,130 / 1,131,471 / 5,412,255 bytes per page (min / median / max). Dimensions range from 774 px wide to 3,438 px tall.
- **Time:** 7.89 s including archive reads.
- **Reproducibility:** only the needed members were read from the archive, and PNG hashes were identical across two runs.
- **Crops:** none were needed.

This small purposive sample does not support a full-dataset storage or compute
estimate.

**Visual check.** I served a byte-identical copy of the packet on localhost and opened it in the built-in browser:

- All 24 images loaded. The page has no scripts and no remote references.
- Image/text pairing was correct in cases 1 (handwritten page, empty MinerU), 4 (two-page news article) and 8 (Chinese table).
- The 150 DPI news page was readable at display size.

I did not review the other 17 cases one by one. No observation was recorded;
the annotations are for a human to make.

## What a human should inspect next

1. For each case, record which observation categories apply and how certain you are. Do not infer a cause from absent text alone.
2. Pay particular attention to:
   - the empty-OCR handwritten and notes pages,
   - the Chinese table and reading-order pages,
   - the two-page law and news cases, where the pages must be read together.
3. Note wherever 150 DPI is too coarse, so crops or a higher DPI can be justified.

## Discrepancies with earlier documents

- The audit report's proposed contract stratified by `doc_type` and `evidence_source`. The lead's decision, implemented here, stratifies by document type only. Evidence types are reported, not controlled.
- The study brief lists "gate-triggered visual recovery" and "random recovery" comparisons. The pilot's short documents constrain the semantic-retry arm (open decision 1).

## Open decisions before the baseline

1. **Semantic retry coverage.** Accept a pilot where most documents are single pages, or define a second, explicitly versioned pilot (for example multi-page train-exclusive documents only). Do not resample this one.
2. **Answer generator.** Choose one common generator and prompt for text and visual conditions.
3. **Answer scoring.** Choose between EM/F1, normalised numeric matching and a judged metric, given the Short/Numeric/Yes-No answer forms.
4. **Chunking and budget.** Decide the page or chunk unit and the evidence budget for within-document retrieval.
5. **Empty and missing OCR at runtime.** Decide what an empty page presents to the text path. It must never be gt text.
6. **Family overlap.** Decide whether pilot tuning results may inform validation runs, given the three shared families.

## Limitations

- The pool is small and structurally unrepresentative (short documents, no charts).
- The families are heuristic.
- The MinerU engine version is unknown.
- Page counts come from pypdfium2; page content was inspected only for three cases.
- The inspection sample is purposive.

Focused checks: `tests/test_build_pilot.py` (12) and `tests/test_audit_assets.py` (21).
