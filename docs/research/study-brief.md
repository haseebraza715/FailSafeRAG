# First study: diagnosis-based recovery for document QA

Working proposal and local audit, 2026-09-28. This brief narrows the first study; it does not claim completed benchmark results or replace the locked protocol without review.

## Research question

When OCR-backed document question answering fails, does diagnosing the failure and choosing a matching recovery improve answers compared with a simple recovery policy, at comparable cost?

Test two links separately: whether the diagnosis matches an independently assessed failure, and whether its selected action repairs the answer. Routing consistency alone proves neither. A null result is valid evidence; do not tune on test outcomes until a positive result appears.

## Proposed first scope

Use OHR-Bench first because its local QA records include answers, evidence pages, document type and evidence source, and its local text assets include ground truth and MinerU outputs. Audit the coverage and provenance of those outputs before deciding whether new OCR is necessary. Defer MP-DocVQA, ArXivQA, ColPali and VisRAG from the initial mechanism pilot. They can later test generalisation and competitiveness.

Compare no recovery, gate-triggered visual recovery, random recovery, and typed recovery. Hold the retriever, answer model, input corpus and evidence budget constant wherever applicable. Include a budget-matched comparison because random and typed policies may invoke different numbers of visual calls. Use multiple predetermined random seeds. Measure paired answer accuracy, successful repairs, damage to initially correct answers, diagnosis accuracy, calls, latency and cost. Report uncertainty; do not set a success margin after seeing test results.

Use clean text as a diagnostic reference condition, never as an input to a noisy-text recovery branch. Keep gold answers and evidence-page labels out of retrieval and routing. Search complete documents or a declared fixed corpus, not only gold evidence pages.

## Verified local inventory

Read-only recursive file inventory, excluding symbolic links. Logical bytes below differ from allocated disk space reported by du. These counts describe this checkout, not remote machines or external caches.

| Directory | Files | Logical bytes | Meaning |
| --- | ---: | ---: | --- |
| OHR-Bench/data | 3124 | 72961487 | QA JSON and retrieval text, including gt and MinerU directories (includes one 6,148-byte .DS_Store that the audit excludes) |
| data/external | 16 | 144955432 | Mostly ArXivQA staging; not proof of complete paper assets |
| data/benchmark_prep | 19 | 7732067 | Small preparation assets; one PDF and one PNG in this tree |
| artifacts/phase0 | 125 | 62487080 | Prototype assets: 80 text files and 45 PNGs |

The locked qas_v2.json contains 8,498 questions and 1,117 distinct raw document names. Its SHA-256 matches the documented lock: 2446db28741fa9f392067ee7aae7f3b05e0d85c584069a50ddd5b1b5bc783f58. No pdfs.zip was found beneath Code/data at the time of this inventory. The official archive has since been obtained and locked (config/ohr_pdf_source_lock.json, 1,516,951,813 bytes); all 1,117 QA documents map to one archive PDF whose page count matches the gt page set (docs/reports/data_audit.md). The 200 GB figure remains unmeasured. Measure compressed inputs, extracted PDFs, page images, OCR text and model caches separately.

The split contains 5,948 train, 1,274 validation and 1,276 test questions. Validation and test share 342 raw document names. The split is question-disjoint but not document-disjoint. Shared documents are leakage only if something is trained or tuned on the training questions or documents, but the split cannot support an unseen-document claim. Alias normalisation may change document counts. Preserve the existing lock. Decide whether the target claim is new questions on familiar documents or unseen documents; the latter requires a separately versioned, document-disjoint protocol.

## Decisions before implementation

1. Confirm the first claim and retrieval scope: known-document QA or search across a fixed collection.
2. Audit MinerU text coverage, source version and alignment to QA and ground truth. Reuse it if it meets the protocol; regenerate OCR only for a stated reason.
3. Locate and hash the actual PDF source (done: see the PDF source lock). Establish image availability for every pilot document.
4. Choose a common answer generator so visual fallback is not compared only against hand-written answer extraction.
5. Define failure annotation rules, including mixed and unknown failures. Annotators must not see the controller prediction before labelling.
6. Select a development pilot without using test answer outcomes; keep its source documents complete (done: ohr_dev_v1, 30 train-exclusive documents; see docs/reports/pilot_readiness.md, including its note that most pilot documents are single pages).

## First implementer assignment

Build a read-only, repeatable asset audit that emits a JSON manifest and concise report. Include file counts and bytes by asset class, source hashes, QA-to-text and QA-to-image coverage, missing assets, page-ID mismatches, raw and normalised document overlap across splits, and provenance gaps. Do not download models, call paid APIs, regenerate splits, delete data or run the full pipeline. Verify counts against the actual files. This report determines the preparation work and subsequent compute request.

After the data contract and comparison design are agreed, fix and verify the answer/recovery paths, beginning with ByT5 device placement and actual correction outcomes. GPU sizing follows measured preparation and model-stage pilots. The existing 108-page job measures preparation only, not scientific effectiveness or the peak memory of every baseline.
