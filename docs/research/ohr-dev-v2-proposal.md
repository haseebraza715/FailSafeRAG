# Proposal: a larger development pilot, ohr_dev_v2

**Status: frozen on 2026-10-31 at 201 documents and 414 questions, as the confirmation sample.** The lead approved
the rule below with one change: about 400 questions instead of 285. Section "As frozen" records the change and the
result. The rest of this page is the proposal as written before approval, kept unchanged. `ohr_dev_v1`,
`config/datasets/ohr_split.json` and `OHR-Bench/data/qas_v2.json` stay unchanged.

## As frozen

**Why the size changed.** The lead raised the target from about 285 to about 400 questions. With perfect page choice,
the gold-page image raised EM by about 7 points on `ohr_dev_v1`. A repair that chooses pages at runtime will
probably gain 5 to 6 points, and 285 questions detect a gain of that size only about half the time.

**What changed.** Only `selection.documents`, from 138 to 201, and the purpose text. The count follows the rule of
step 6 below: 400 / (622 / 313) = 201.3, rounded to 201. The seed, pool, stratification, exclusion and inspection
block are as proposed. The seed was fixed before either draw.

**Result** (`config/pilots/ohr_dev_v2.json`, `results/pilots/ohr_dev_v2/selection_record.json`, selection digest
`0d36bf39…`):

| | `ohr_dev_v2` | Left in the train-only pool afterwards |
| --- | ---: | ---: |
| Questions | 414 | 208 |
| Documents | 201 | 112 |
| Pages | 416 | 245 |
| Questions on multi-page documents | 128 | |
| Questions whose evidence page has empty MinerU text | 56 | 33 |
| Han-script questions | 62 | 25 |
| Questions with evidence on several pages | 25 | 12 |

Documents by type: academic 6, administration 13, finance 2, law 7, manual 11, news 52, textbook 110.

**Role.** `ohr_dev_v2` is the confirmation sample. Development stays on `ohr_dev_v1`. Each pre-registered comparison
runs on `ohr_dev_v2` once, and nobody inspects its outputs before the comparison is computed. The 208 train-only
questions left in the pool are not used by any run.

The tables below this point describe the 285-question proposal, not the frozen pilot.

## Why a second pilot

`ohr_dev_v1` has 70 questions. Detecting a 5-point paired EM gain with McNemar's test (alpha 0.05, power 0.80)
needs about 190 questions when the two policies disagree on 6% of questions and about 310 at 10%, before the
correction for questions that share a document (failure-analysis report, section 5). No contrast on 70 questions
can show such a gain.

## How many questions the training split can give

The fixed split assigns questions, not documents, to train, validation and test. Most documents carry questions
from more than one split:

| Documents whose questions are in | Documents |
| --- | ---: |
| train only | 343 |
| train, validation and test | 333 |
| train and test | 176 |
| train and validation | 154 |
| validation or test only, or both | 111 |

`ohr_dev_v1` drew only from the 343 train-only documents, so no development document appears in validation or
test. This proposal keeps that rule. The 343 documents hold 692 questions. `ohr_dev_v1` used 30 documents and 70
questions, which leaves 313 documents and 622 questions.

Taking train questions from shared documents would give far more questions (5,948 train questions in all), but the
method would then be developed on documents whose other questions are in validation and test. The proposal does not
do that.

## Selection rule

The draw reuses `stratified-hash-rank-v1`, the algorithm that froze `ohr_dev_v1`, with one addition that is
committed and tested: `selection.exclude_documents_of_pilots` removes the documents of named frozen pilots from the
eligible pool before allocation (`scripts/data/build_pilot.py`, `excluded_pilot_documents`).

1. Pool: train-only documents with one source PDF and a clean reference text, as for `ohr_dev_v1`. Missing or
   empty OCR never excludes a document or a question.
2. Remove the 30 `ohr_dev_v1` documents. The two pilots share no document, so `ohr_dev_v2` results are not
   shaped by anything read on `ohr_dev_v1`, including the v2 prompt.
3. Stratify by document type (`doc_type`). Give every type at least one document, then allocate the rest in
   proportion to the pool by largest remainder.
4. Within a type, take documents in ascending `sha256(seed, "select", doc_id)`. The seed is 20261031.
5. Take every question of each selected document.
6. Document count: 138. The target was about 275 questions, and the pool averages 622 / 313 = 1.99 questions per
   document, so 275 / 1.99 rounds to 138. The count was set from the pool average before the draw was previewed.

The proposed config:

```json
{
  "schema_version": 1,
  "pilot_id": "ohr_dev_v2",
  "purpose": "second development pilot, sized to detect a 5-point paired EM difference at a low discordant rate; document-disjoint from ohr_dev_v1; not a final benchmark and not an unseen-publication evaluation",
  "dataset": "OHR-Bench",
  "qa_source": "OHR-Bench/data/qas_v2.json",
  "split": "config/datasets/ohr_split.json",
  "audit_manifest": "results/data_audit/asset_manifest.json",
  "pdf_source_lock": "config/ohr_pdf_source_lock.json",
  "noisy_text_source": "MinerU",
  "reference_text_source": "gt",
  "output_dir": "results/pilots/ohr_dev_v2",
  "selection": {
    "algorithm": "stratified-hash-rank-v1",
    "seed": 20261031,
    "documents": 138,
    "pool": "train_exclusive",
    "stratify_by": "doc_type",
    "minimum_per_available_stratum": 1,
    "exclude_documents_of_pilots": ["ohr_dev_v1"]
  },
  "inspection": {
    "algorithm": "purposive-quota-hash-rank-v1",
    "cases": 20,
    "page_budget": 60,
    "dpi": 150,
    "image_format": "png",
    "category_quotas": [
      ["ocr_gap", 3], ["multi_page_evidence", 2], ["multi_evidence_source", 2], ["han_script", 2], ["table", 2],
      ["formula", 2], ["chart", 2], ["reading_order", 2], ["text", 3]
    ]
  }
}
```

The inspection block is the `ohr_dev_v1` block unchanged. It picks 20 questions for the human inspection packet that
study brief section 10 describes. The pool has no chart questions, so the chart quota stays unfilled, as it did for
`ohr_dev_v1`.

## What the draw gives

An in-memory preview of this rule (metadata only: document identity, type, page count, OCR status and question
flags; no answers and no model output) selects:

| | `ohr_dev_v2` | Left in the train-only pool afterwards |
| --- | ---: | ---: |
| Questions | 285 | 337 |
| Documents | 138 | 175 |
| Pages | 274 | 387 |
| Questions on multi-page documents | 91 | 96 |
| Questions whose evidence page has empty or missing MinerU text | 41 | 48 |
| Han-script questions | 43 | 44 |
| Questions with evidence on several pages | 22 | 15 |
| Evidence source: text, formula, reading order, table, several | 200, 32, 31, 20, 2 | 241, 40, 14, 42, 0 |

| Document type | Pool documents | Selected documents | Selected questions |
| --- | ---: | ---: | ---: |
| academic | 9 | 4 | 10 |
| administration | 20 | 9 | 32 |
| finance | 3 | 1 | 5 |
| law | 11 | 5 | 25 |
| manual | 17 | 7 | 12 |
| news | 81 | 36 | 93 |
| textbook | 172 | 76 | 108 |

The freeze recomputes these numbers and records them in `results/pilots/ohr_dev_v2/selection_record.json`.

## What 285 questions can detect

With the v2 prompt, the gold-page-image condition disagreed with the no-recovery baseline on 12.9% of the
`ohr_dev_v1` questions (`results/analysis/2026-10-31-ohr-dev-v1-terra-answer-v2/report.md`, section 4). At that rate,
285 questions detect a paired EM gain of about 6 points with McNemar's test (alpha 0.05, power 0.80), or about 7
points with a design effect of 1.45. A 5-point gain needs about 400 questions at that rate. The pool cannot give
400 train-only questions and still leave anything for confirmation, so 285 is a compromise: it detects a 6 to 7
point gain, not a 5-point one.

The design effect is uncertain. The intraclass correlation of EM within documents was 0.40 on the v1 baseline and
0.02 on the v2 baseline, both from the same 30 documents. `ohr_dev_v2` averages 2.07 questions per document, so the
design effect lies between about 1.0 and 1.45.

## Expected cost

Freezing the pilot calls no model. Rendering its inspection packet costs local time only.

A later no-recovery baseline on `ohr_dev_v2` with `faar-answer-v2` and `gpt-6.1-terra` would send about 245
requests (questions whose evidence pages have no text are not sent). The v2 baseline on `ohr_dev_v1` cost 0.095867
USD for 63 requests, about 0.0015 USD each, so the expected cost is about 0.40 USD. The runner's upper bound for one
attempt per request is about 1.6 USD, so a safety ceiling of 0.60 USD leaves room for every request while capping a
fault. That run needs its own approval.

## A consequence for the confirmation sample

The draft repair analysis plan (`docs/research/repair-analysis-plan.md`, section 2) puts a confirmation sample in
the train split, document-disjoint from development. After `ohr_dev_v1` and `ohr_dev_v2`, the train-only pool
holds 337 questions. That is near the lower end of the 290 to 480 questions the plan needs, and it would leave no
train-only questions for anything else. Two ways forward:

- Use `ohr_dev_v2` itself as the confirmation sample: freeze it, develop only on `ohr_dev_v1`, and run each
  pre-registered comparison on `ohr_dev_v2` once. Nobody inspects its outputs before that run.
- Keep `ohr_dev_v2` for development and confirm on the validation split, as study brief section 15.11 already
  proposes for the frozen method (one confirmation run after the freeze).

The first keeps validation untouched for the method freeze. The second gives more development data but uses
validation earlier. I recommend the first: the repair experiment in `docs/research/visual-repair-design.md` is
written that way.

## Decisions for the lead

1. Approve, change or reject the selection rule and the config above.
2. Approve the document-disjoint exclusion of `ohr_dev_v1`.
3. Decide the role of `ohr_dev_v2`: confirmation sample (recommended) or development sample.
4. Approve any later run on `ohr_dev_v2` separately, with its spending cap.
