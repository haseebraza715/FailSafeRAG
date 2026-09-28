# Data audit: FAAR assets and OHR-Bench readiness

This audit covers the local checkout as of 2026-09-28. It was taken at commit
`22be5e29be5e75191c7979b9414dc9f041b72987`, with uncommitted documentation and
audit changes. It was re-run after the official OHR PDF archive was obtained.

All figures come from `results/data_audit/asset_manifest.json` (content sha256
`7bb6df50f543f2c4130409f46e98865a34d2316e7c7e6ceba4e87f0c2557e04b`).
`results/data_audit/asset_audit_summary.md` repeats them in table form. The
audit does not cover other machines, cluster scratch, or caches that are not
configured.

```bash
.venv-aaai/bin/python scripts/data/audit_assets.py --project-root . --out-dir results/data_audit --pdf-zip data/ohr_bench_raw/pdfs.zip
```

The run completed with exit status 0 and no integrity failures. The manifest
reports `audit_execution` and `experimental_readiness` separately:

- **Audit execution:** completed, exit 0, no integrity failures.
- **Experimental readiness:** `no_recorded_blockers`. This means no issue is
  recorded that blocks preparation. It does not mean the benchmark is ready:
  3 warnings and the scope decisions in section 7 still apply.

The script header documents the exit codes:

- **0:** the audit completed. Findings of any severity may exist.
- **1:** the audit could not run.
- **2:** an integrity failure. That covers a lock mismatch (including the PDF archive), malformed or duplicate QA, or split overlap.

The audit uses the standard library and the repository's `faar.ohr_inventory`
alias rules. It uses `pypdfium2` for page counts and needs Python 3.11 or later.

## 1. What is verified now

| Asset | Status | Evidence |
| --- | --- | --- |
| `qas_v2.json` | verified | SHA-256 matches `config/split_checksums.json`. Git blob `6fc22e11…` equals upstream GitHub `opendatalab/OHR-Bench` `data/qas_v2.json` at `1f421eb4…`, last changed by "fix qas_v2" (2025-03-14). |
| OHR split | verified | SHA-256 matches its lock. The file is unchanged. |
| `retrieval_base/gt`, `retrieval_base/MinerU` | bytes verified against upstream | All 18 category tree ids match upstream git (2 after a recorded name normalisation, below). |
| OHR PDF archive | verified | 1516951813 bytes. SHA-256 `f9bc65f3…` equals the Hugging Face LFS checksum. The ZIP CRC check passed. Locked in `config/ohr_pdf_source_lock.json`. |
| Complete-document page sets | verified for 1117 of 1117 QA documents | Each document maps to exactly one archive PDF, which `pypdfium2` opens. Its pages 0..n-1 equal the gt `page_idx` set, 8400 pages in total. |

"Verified" here means byte identity with the upstream host or repository. It
does not authenticate who produced the data. It also does not check whether gt
is correct; upstream describes gt as human-verified.

## 2. PDF source acquisition record

- **Source.** Hugging Face dataset `opendatalab/OHR-Bench` at revision
  `7f833e3eda9a571a9ea545a8f6d476fa1685033d`. That is main at lookup, and the
  server confirmed it in `X-Repo-Commit`. The file is `pdfs.zip`, last changed by commit
  `80aca2bb…` (2025-03-10) and unchanged since.
- **Before download.** The advertised size was 1516951813 bytes (1.41 GiB),
  under the 5 GiB limit. Free local space was about 140 GB.
- **Download.** `curl -L` of the pinned resolve URL to
  `data/ohr_bench_raw/pdfs.zip.partial`.
- **Validation.** Observed size 1516951813 bytes. SHA-256
  `f9bc65f383172c4ea47940c47dfab01dd36c03a120bc0450d7a962917098c783`. This
  equals the published LFS checksum and the `X-Linked-ETag` header. `testzip`
  found no bad member.
- **Publication.** Renamed to `data/ohr_bench_raw/pdfs.zip`, the default path
  (git-ignored by `data/`). The file was retrieved at 2026-09-28T15:59:56Z.
  Nothing was overwritten or extracted.
- **Contents.**
  - 1261 PDF members in 7 category directories, all at depth one.
  - No encrypted members.
  - No canonically equivalent member names.
  - Declared uncompressed bytes 1618187174; declared compressed member bytes 1516661149.
  - 144 PDFs map to no QA document: textbook 112, news 20, administration 5, academic 4, manual 3.
- **Relationship to the locked QA.** The archive and the retrieval base were
  both last changed on 2025-03-10, during the upstream "OHRBench-v2" update. The
  QA fix followed on 2025-03-14. Upstream publishes no manifest tying the
  archive to a QA version. Compatibility therefore rests on the checks here:
  every QA document maps to one PDF whose page count matches gt.

The 1.4 GB figure in older documentation turns out to match the compressed
archive (1.52 GB, or 1.41 GiB). No expanded-corpus or rendered-image size has
been measured. This audit does not extrapolate one from the archive size.

## 3. What data do we have?

This section counts regular files by logical bytes, without following symbolic
links. Hard links are counted once. `.DS_Store`, `__pycache__` and the audit output are excluded.

| Root | Files | Logical bytes | Contents |
| --- | ---: | ---: | --- |
| `OHR-Bench/data` | 3123 | 72955339 | 3 QA JSON files and 3120 retrieval-text files (1560 gt, 1560 MinerU) |
| `data/ohr_bench_raw` | 1 | 1516951813 | Official PDF archive (compressed input) |
| `data/external` | 16 | 144955432 | ArXivQA QA sources, dataset cards, download metadata |
| `data/benchmark_prep` | 19 | 7732067 | One-document OHR smoke preparation, ArXivQA staging manifests |
| `artifacts`, `logs` | 140, 206 | 62517123, 1471913 | Prototype outputs and logs |

The prepared smoke PDF (1129395 bytes) is kept separate from the source. Its hash
equals the matching archive member. Page images exist only for 47 questions (the
prototype and smoke sets). No model cache is configured, so none was inventoried.

## 4. Documents, text and OCR conditions

**PDF mapping.** Of the 1117 QA documents:

- 1107 match an archive member by exact name.
- 10 match through the existing `textbook_needrop_en_ → jiaocai_needrop_en_` alias.
- No document has an ambiguous, duplicate, missing or unreadable candidate.
- The 149 excerpt documents (`<publication>.pdf_<page>`) are single-page cuts of larger publications. Each is a complete benchmark document with its own PDF, and all 149 verify. The larger publications themselves are not in the archive.

**Name normalisation.** Upstream stores one Springer textbook name in NFD in the
QA file, the retrieval base and the archive. This checkout stores it in NFC. The
file contents are identical. The lock records the name form, and the audit
compares names after NFC. The earlier "filesystem-dependent name" warning was
caused by the local re-encoding of this one file name, not by upstream.

**Text coverage.** The denominator is 8498 questions. A question counts as
covered when its document resolves and every evidence page exists with
non-empty text.

- **gt:** covers all 8498. gt itself has 299 empty pages in 111 files, none of them an evidence page.
- **MinerU:** covers 8256. 237 questions have an empty MinerU evidence page, and 5 have a missing one.

**MinerU conditions.** These are recorded and nothing is excluded. The
per-document lists are under `ohr.coverage.ocr_conditions.MinerU`.

| Condition | Documents | Pages |
| --- | ---: | ---: |
| Pages absent from MinerU but present in the PDF and gt | 20 | 23 |
| Empty MinerU pages | 263 | 478 |
| Of those, gt has text | — | 284 |
| Any condition | 275 of 1117 | — |

A missing or empty MinerU page where gt has text may be a real OCR failure. It
may also be an extraction or packaging artefact. Only visual inspection of the
page can tell which.

**Page convention.** The code reads `evidence_page_no` as a 0-based
`page_idx`. The evidence context was found on the labelled gt page for 5474
questions. It was found only on the preceding page for 1 question and only on the
following page for none. Every evidence page lies within the verified PDF page range.

For 2981 questions the context is not verbatim page text: 526 of the 765 chart
questions and all 135 `multi` questions fall in this group. The check says
nothing about those questions.

**MinerU evidence string match (diagnostic only).** This covers the 5474
questions whose evidence context occurs verbatim on the gt evidence page. For
3035 of them, the same normalised string also occurs on the MinerU page.

| Evidence type | Found in MinerU / checked |
| --- | ---: |
| text | 1699 / 2434 |
| table | 883 / 1680 |
| formula | 357 / 897 |
| reading order | 59 / 241 |
| chart | 37 / 222 |

This is an exact-substring diagnostic, not a measure of OCR accuracy or a
failure label:

- Changes in table markup, LaTeX, reading order or hyphenation cause mismatches even when the content survives.
- Charts, formulas and multi-page evidence often have no continuous-text form, so they are under-represented in the denominator.
- Neither a match nor a mismatch shows whether the question is answerable from MinerU.

Classifying actual failures needs visual inspection.

**Question scripts.** The audit reports which scripts appear in each question.
This is a character check, not language identification. The ranges checked are:

- **Han:** U+3400–4DBF, U+4E00–9FFF, U+F900–FAFF, U+20000–2FA1F
- **Kana:** U+3040–30FF, U+31F0–31FF, U+FF66–FF9F
- **Hangul:** U+1100–11FF, U+3130–318F, U+A960–A97F, U+AC00–D7FF

972 questions contain Han characters. None contain kana or Hangul. Han alone does not tell Chinese from Japanese. The flags are recorded per question and exclude nothing.

## 5. Split integrity

The split is question-disjoint, with no question in two splits. It is not
document-disjoint.

| Pair | Shared documents (raw) | After alias normalisation |
| --- | ---: | ---: |
| train & val | 487 | 487 |
| train & test | 509 | 509 |
| val & test | 342 | 342 |

1206 of 1274 validation questions and 1214 of 1276 test questions concern
documents that also appear in train. Only 343 documents (692 questions) appear
exclusively in train. Validation has 53 exclusive documents and test has 49.

Shared documents are leakage only if something was trained or tuned on the
training questions or their documents. That depends on the experiment, not the
split. The existing split cannot support an unseen-document claim. The locked
split was not modified.

## 6. Documentation reconciliation

| Claim | Location | Status |
| --- | --- | --- |
| Archive "about 1.4 GB"; "200 GB expanded corpus" | `SUPERVISOR_HANDOFF.md` at HEAD line 43; `runbook.md` at HEAD line 275 | Compressed size now measured (1516951813 bytes). 200 GB remains unmeasured. The uncommitted wording now cites the lock. |
| "Use the existing OHR archive" | `cluster/README.md:217` | Now points to the locked archive and its default path. |
| `FAAR_PDF_ROOT` default | `cluster/README.md:418` | Corrected earlier to "unset". |
| Val 549 docs / 7,037 pages; test 567 / 6,849 | `runbook.md:173`, `cluster/README.md:306-310` | Reproduced, and now verified against the source PDFs. |
| ArXivQA: 497 PDFs downloaded, 8,955 pages rendered | `docs/history/aaai-execution-status.md:43`, `docs/history/runlog.md:74`, `docs/paper/faar_aaai_findings.tex:77` | The 18407 referenced files are absent from this checkout. That does not show the work never happened elsewhere. History is preserved. The status paper now carries a one-sentence qualification. |
| All locked model snapshots downloaded | `aaai-execution-status.md:39`, `runlog.md:73` | Not checked; no cache configured. |

## 7. Decisions for the research lead

These questions concern the lead's proposed first experiment: OHR only, a
training-document pilot, retrieval within each complete source document, and
MinerU as a fixed noisy-text condition.

1. **Pilot pool.** Decide whether "training documents" means documents that appear only in train (343 documents, 692 questions) or any train document (1006). The second choice shares documents with validation and test.
2. **MinerU as a fixed condition.** The bytes match upstream, but the engine version is unknown. Decide whether to accept that and state it as a limitation.
3. **Handling recorded conditions.** Decide how to analyse, not exclude, these groups:
   - questions with empty or missing MinerU evidence pages (242),
   - list-valued evidence (761) and `multi` evidence (135),
   - Han-script questions (972).
4. **Failure labelling.** Define the visual-inspection protocol that decides whether a MinerU gap or mismatch is a real OCR failure.
5. **Renderer.** Choose the renderer and DPI for page images. Measure storage on a sample before rendering at scale.

## 8. Proposed first-study data contract (revised; not adopted)

- **Sources.** The locked `qas_v2.json`, split and `pdfs.zip`, hash-checked before every run. Retrieval-base bytes are pinned by the upstream tree ids in the lock.
- **Pilot selection.** Select from training documents per decision 1, by document, stratified by `doc_type` and `evidence_source`, without using answer outcomes. Nothing is selected or frozen in this assignment.
- **Source eligibility.** Include a document when:
  - its identity and page mapping are defensible: one source PDF, no ambiguous candidates, the page count equals the gt page set, and evidence pages are in range;
  - its text sources resolve by a recorded rule (exact, documented alias, or Unicode NFC).

  A document fails eligibility only when its source data is unusable (missing, ambiguous or unreadable PDF, or a page-set mismatch). Every exclusion is recorded with its reason and its effect on the doc_type and evidence_source mix.
- **OCR conditions.** Missing or empty MinerU pages are recorded conditions of the noisy input, not exclusions. Noisy text is never filled from gt.
- **Per-document inputs.**
  - source PDF bytes (by member hash),
  - page images rendered from the PDF,
  - MinerU text as the noisy input,
  - gt text as a diagnostic reference only, never as a recovery input.
- **Retrieval corpus.** All pages of each complete benchmark document, never evidence pages only.
- **Gold labels.** Answers, evidence pages and evidence contexts stay out of retrieval, gating and routing.
- **Provenance.** Each derived file records its source hash, tool, version and parameters. Each result bundle records this audit's `content_sha256`.

## 9. Limitations

- The MinerU engine version and settings are unknown upstream.
- Upstream offers no manifest linking the archive to a QA version.
- Page coverage depends on `pypdfium2` page counts. Page content was not rendered or inspected.
- The string-match and alignment checks under-represent charts, formulas and multi-page evidence.
- The model cache, other checkouts and cluster storage were not inspected.
- Focused checks: 21 fixture tests in `tests/test_audit_assets.py`.
