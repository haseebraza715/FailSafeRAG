# Where a longer-document sample can come from

**Status: report for the research lead, written 2026-11-01. Nothing was selected or frozen, and nothing was
spent.** The counts come from `scripts/data/count_multipage_pools.py`, which reads only metadata: the locked QA file,
the fixed split, the asset audit, MinerU page inventories and the frozen selections of `ohr_dev_v1` and `ohr_dev_v2`.
It reads no answers, evidence text or model outputs. Eligibility is the rule of `scripts/data/build_pilot.py`: one
raw QA name, one verified source PDF and a clean reference text.

## Why a longer-document sample

On `ohr_dev_v2`, every gain of the visual repair came from single-page documents. On the 128 questions on multi-page
documents it gained none and lost 4. Those documents had almost no empty-OCR cases, so the sample cannot show
whether runtime page choice works, and it gives the retrieval repair little to fix. The diagnosis-based policy routes
semantic failures to `retry_retrieval`, which matters only when a document has more than one page.

## Answer to the first question: not enough train-only questions are left

| Pool | Questions | On multi-page documents | Multi-page documents | Questions per multi-page document | On documents of 6+ pages | Multi-page, evidence page with empty OCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Train-only documents outside `ohr_dev_v1` and `ohr_dev_v2` | 208 | **59** | 20 | 2.95 | 31 | 2 |
| D. All train-only documents (reference) | 692 | 206 | 66 | 3.12 | 85 | 3 |

Fifty-nine questions on 20 documents cannot detect a 5-point paired gain at any plausible disagreement rate.
McNemar's test needs 186 to 403 questions before any clustering correction (table below). Pool A is too small by a
factor of 3 to 7.

## A second finding: OHR-Bench has almost no multi-page questions with empty-OCR evidence

| Pool | Multi-page questions | Of these, evidence page with empty or missing MinerU text | On a document with any empty page |
| --- | ---: | ---: | ---: |
| A. Train-only, outside both pilots | 59 | 2 | 16 |
| B. Train questions on documents shared with validation or test | 4,320 | 65 | 1,990 |
| C. Validation questions | 968 | 19 | 445 |

Empty-OCR evidence pages sit almost entirely on single-page documents (handwriting and short news pages). So no
OHR-Bench sample can test "find the right page when its text is missing" with useful power. A longer-document sample
would test retrieval and diagnosis on documents whose text exists. That is still the case the diagnosis-based policy
has to win, because the gate already handles empty-text pages.

## Other sources inside OHR-Bench

The fixed split assigns questions, not documents, to train, validation and test. Most long documents carry questions
from more than one split.

| Option | Multi-page questions | Documents | Cost to the study |
| --- | ---: | ---: | --- |
| E. Train questions on documents shared only with validation | 364 | 81 | Validation documents are seen during development (their train questions, not their validation questions) |
| F. Train questions on documents with no test question (pool A plus option E, outside both pilots) | **423** | 101 | As E. Test documents stay unseen |
| B. Train questions on documents shared with validation or test | 4,320 | 398 | Test documents are seen during development. Not recommended |

**Option F is the only OHR-Bench option that keeps test documents unseen and has a few hundred multi-page
questions.** Its price is that validation is no longer document-disjoint from development: the method is developed
on other questions about the same documents. The validation questions themselves stay unused, so validation remains
free of tuning on its own outcomes, but a validation result would then be optimistic for retrieval on those
documents. Confirmation could move to a held-out part of option F instead (see below).

## What option F can detect

Questions on the same document are correlated. Option F averages 4.19 questions per multi-page document. The
intraclass correlation of EM within documents was 0.086 on `ohr_dev_v2` and 0.40 on the v1 baseline of `ohr_dev_v1`.

Questions needed to detect a 5-point paired EM gain (McNemar, alpha 0.05, power 0.80):

| Disagreement rate between the two policies | No clustering | ICC 0.086 (design effect 1.27) | ICC 0.40 (design effect 2.28) |
| ---: | ---: | ---: | ---: |
| 6% | 186 | 238 | 424 |
| 8.9% (gated vs no recovery on `ohr_dev_v2`) | 278 | 354 | 631 |
| 10% | 312 | 398 | 710 |
| 12.9% (gold-page image vs no recovery on `ohr_dev_v1`) | 403 | 514 | 917 |

Gain detectable with all 423 questions of option F:

| Disagreement rate | No clustering | Design effect 1.27 | Design effect 2.28 |
| ---: | ---: | ---: | ---: |
| 6% | 3.4 points | 3.8 | 5.1 |
| 10% | 4.3 | 4.9 | 6.5 |
| 12.9% | 4.9 | 5.6 | 7.4 |

So option F detects a 5-point gain if the policies disagree on about 10% of questions or fewer and clustering is
mild. It cannot be split into a development half and a confirmation half that each have that power.

## Sources outside OHR-Bench

- **MP-DocVQA** (multi-page DocVQA, up to 20 pages per document, with its own OCR). The study brief lists it as later
  work (section 2). It would give thousands of multi-page questions. Whether its splits share documents was not
  checked here. It would need its own noisy-text input (the dataset's OCR, not MinerU), its own audit and a lead decision
  to bring it forward.
- **The validation split of OHR-Bench** has 968 multi-page questions. Using it for development would spend the
  split that brief section 15.11 keeps for one confirmation run after the method freeze. Not recommended.

## Recommendation

1. Use option F as the **longer-document development sample**, document-disjoint from `ohr_dev_v1` and `ohr_dev_v2`
   and stratified by document type, with all 423 multi-page questions. Record that validation documents are seen in
   development.
2. Keep `ohr_dev_v2` as the confirmation sample for the single-page and empty-text case. Run the main comparison
   (FAAR against the gated visual repair) on it, as pre-registered.
3. For a confirmation on long documents, bring MP-DocVQA forward as a lead decision, or accept that the long-document
   result stays development evidence until the OHR-Bench test run.

## Decisions for the lead

1. Choose a longer-document source: option F (recommended), option B, MP-DocVQA, or none.
2. If option F: approve recording validation documents as seen during development, and say whether the sample
   takes all 423 multi-page questions or a stratified subset.
3. Say whether the main comparison confirms on `ohr_dev_v2` only, or also needs a long-document confirmation.
