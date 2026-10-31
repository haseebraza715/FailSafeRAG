# Pre-registration draft: diagnosis-based repair (FAAR) against gated visual repair

**Status: draft for the research lead, written 2026-11-01. The lead decided the open points of section 9 on
2026-11-01 (analysis plan, amendment 3); section 11 records them and overrides the earlier sections where they
differ. Nothing has run.** It
follows the decision of 2026-11-01 (analysis plan, amendment 2, item 3): the bar is the gated visual repair, which
raised official EM by 8.5 points over no recovery on `ohr_dev_v2` at 0.0027 USD per question. FAAR has to beat it
at similar cost. Sections 9 and 10 list what must be decided and built first.

## 1. Two problems to settle before anything else

**1. Under the current gate, the two policies can differ on very few questions.** Both policies act on the same
flagged set (brief section 5). The gate `abstain-or-ocr-gap-v1` flagged 131 of the 414 `ohr_dev_v2` questions: 71
with no text evidence and 60 answered ones. On the 71, the only repair that can do anything is the image, because
there is no text to correct and nothing for retrieval to find. If FAAR also shows the image there, FAAR and the
gated visual repair give the same answers on those 71, and they can differ only on the 60. On those 60, the gated
visual repair gained 9 and lost 1. To beat it by 5 EM points over all 414 questions, FAAR would need about 21 more
correct answers than the image gives, out of 60. On `ohr_dev_v1`, a perfect text correction (clean ground-truth
text) added 1.4 points. **So on `ohr_dev_v2` with this gate, the main comparison is very likely to come out null.**
That would be a test of almost nothing, not evidence that diagnosis does not help.

The comparison has power only on cases where text exists but is wrong or was not retrieved, and where the gate flags
those cases. That needs both a longer-document sample
([longer-document-sample-options.md](longer-document-sample-options.md)) and a gate that flags answered questions
with weak retrieval, such as the brief's reranker-threshold gate (a lead decision).

**2. The existing controller sends questions with no text to the wrong repair.** `diagnose_failure` returns
`semantic` when retrieval found nothing (`src/faar/quality.py:96-103`), and `semantic` maps to `retry_retrieval`
(`src/faar/graph.py:22-26`). A retry on a document with no text finds nothing again. As written, FAAR would lose the
27 gains the image made on no-text questions, by construction. The draft adds a `missing_text` diagnosis that routes
to the image (section 4). That rule was suggested by the `ohr_dev_v1` headroom runs (the image fixed 4 of 10
`ocr_lost` cases) and confirmed by the `ohr_dev_v2` aggregate result. Because it was seen on the confirmation
sample, the lead should approve it explicitly. Otherwise the confirmation for FAAR moves elsewhere (section 6).

## 2. Question

On the same questions, with the same gate, initial answer, answer model and budgets, does the repair chosen by
diagnosis give a higher official EM than showing the page image on every flagged question, at a similar cost?

## 3. Policies

All four share the retrieval, the initial answer (no recovery, `faar-answer-v2`), the answer model and its settings,
the gate and its flagged set, the image budget (3 pages, `retrieved-then-gaps-v1`, 150 DPI, JPEG quality 90,
`detail: high`) and the text budget (top 5 chunks).

| Policy | Flagged question gets |
| --- | --- |
| No recovery | the initial answer |
| Gated visual repair (the bar) | `invoke_vlm`: the initial text plus page images |
| Random repair | one of `retry_retrieval`, `correct_text`, `invoke_vlm`, drawn by `sha256(seed, question_id)` for each of 3 seeds fixed before the run |
| FAAR | the repair its diagnosis selects (section 4) |

A question the gate does not flag keeps its initial answer under every policy.

## 4. FAAR's diagnosis and repairs

The diagnosis reads only what brief section 4 allows: the question, the retrieved chunks with their scores, and the
per-page OCR status of the document.

| Diagnosis | Fires when (in this order) | Repair |
| --- | --- | --- |
| `missing_text` (new, needs approval) | no text evidence was retrieved, or the retrieved pages or the pages the page policy would show include a page whose MinerU text is empty or missing | `invoke_vlm` |
| `structural` | the top chunk shows at least `structural_threshold` layout signals (`quality.layout_signals`) | `invoke_vlm` |
| `word_level` | the top chunk's `weird_char_ratio` is above `weird_char_threshold` | `correct_text` |
| `semantic` | none of the above | `retry_retrieval` |

The thresholds (code defaults 1 and 0.10) are fixed on development data before the confirmation run and recorded in
the method record.

What each repair does in this experiment:

- **`invoke_vlm`**: exactly the request of `scripts/experiments/run_visual_repair.py`. Where the request is
  byte-identical to one already answered in the gated visual run on the same sample, that saved response is reused,
  so the two policies give the same answer wherever they choose the same repair.
- **`correct_text`**: each retrieved chunk is corrected (ByT5 or SymSpell, chosen on development data, with the
  existing acceptance rules), then the answer model answers again from the corrected chunks with the same prompt. No
  ground-truth text is a correction target.
- **`retry_retrieval`**: a second retrieval over the same document with an expanded query, then the answer model
  answers again. The current expansion appends the first 24 words of the retrieved text (`recovery.py:544-549`),
  which can repeat the first error. A better rule may be developed on development data only.

## 5. Outcomes, contrasts and decision rules

- **Primary:** official EM over all questions; `execution_failed` and `no_evidence` count as 0.
- **Primary contrast:** EM(FAAR) minus EM(gated visual repair), paired by question. 95% bootstrap interval that
  resamples documents (10,000 resamples, seed 20261031), and McNemar's exact two-sided test.
- **Secondary contrasts:** FAAR minus random (pooled over the 3 seeds, and per seed), FAAR minus no recovery. Holm's
  method over the primary and these two at family-wise alpha 0.05 (plan section 5).
- **Decision rule:** "FAAR beat the gated visual repair" only if the Holm-adjusted McNemar p is below 0.05 and the
  document-level interval lies above 0.
- **Cost rule:** "at similar cost" only if FAAR's total cost per question, measured plus reserved, is at most 1.10
  times the gated visual repair's. Every cost component of brief section 5 is reported per policy. If FAAR's cost is
  higher than that, the result is reported as "higher EM at higher cost", whatever the EM.
- **Also reported:** F1, ANLS, NFKC EM; repair and damage rates on the flagged set with exact intervals; diagnosis
  counts; each repair's results on the cases routed to it; the per-signal agreement with human labels (Q4) when the
  labels exist.
- **Execution:** the resend rule of amendment 1 and the failure rule of amendment 2 apply.

## 6. Samples

| Stage | Sample | Use |
| --- | --- | --- |
| Development | `ohr_dev_v1` and the longer-document sample, if approved | Build the repairs, fix the thresholds, choose the correction model and the retry rule, run the all-repairs diagnostic of brief section 6 |
| Confirmation | To be decided (section 9) | One run of every policy, then the comparison script, committed before the run |

Candidates for the confirmation:

- **`ohr_dev_v2`**, reusing its no-recovery and gated visual runs. Cheap, but underpowered under the current gate
  (section 1). Its aggregate results are already known, and they shaped the `missing_text` rule.
- **A held-out part of the longer-document sample**, where text repairs and retrieval matter. Its power depends on the
  split (the options report gives the numbers).
- **OHR-Bench validation**, after the method freeze, as brief section 15.11 already plans.

## 7. Power

The primary contrast compares two active policies, so they agree on most questions. The disagreement rate is unknown
until the all-repairs diagnostic runs on development data. At 6% disagreement, a 5-point gain needs 186 questions
without clustering and about 240 with mild clustering; at 10%, 312 and about 400. The comparison script reports the
observed rate and the detectable gain next to the result.

## 8. Cost estimate (not authorised)

On `ohr_dev_v2`, with reuse of the gated visual responses, FAAR adds one text request per flagged question routed to
`correct_text` or `retry_retrieval`, at most 131 requests at about 0.0016 USD each, so at most 0.21 USD. Random
repair adds up to 3 × 131 text requests, at most about 0.63 USD, plus any image requests not covered by reuse (none,
since the gated run answered every flagged question). Total at most about 0.85 USD. A longer-document confirmation
needs its own no-recovery and gated visual runs first, and is costed when its sample is chosen.

## 9. Decisions for the lead

1. The gate for the main comparison: keep `abstain-or-ocr-gap-v1` (the measured bar, but little room to differ), or
   add a retrieval-confidence condition chosen on development data, with the bar rerun under that gate.
2. Approve the `missing_text` diagnosis and its route to `invoke_vlm`, knowing it was confirmed on `ohr_dev_v2`.
3. The confirmation sample (section 6).
4. The cost tolerance of 1.10 for "similar cost".
5. The correction model for `correct_text` (ByT5 or SymSpell) and whether `retry_retrieval` may get a new expansion
   rule during development.
6. Approval of each paid run and its cap.

## 10. What has to be built

1. A diagnosis step that writes, per flagged question, the signals, their values and the chosen repair, reading only
   runtime inputs. Tests from a failure list, as for the visual repair runner.
2. `correct_text` and `retry_retrieval` runners in the style of `run_visual_repair.py`: same ledger, cap, resend
   rule and boundary checks, and the same answer prompt.
3. The random-repair assignment and the reuse of byte-identical image responses.
4. A comparison script, committed before the confirmation run.

## 11. Decisions of 2026-11-01

1. **Gate:** `abstain-or-ocr-gap-v1` plus a retrieval-confidence condition, chosen on development data only. The
   gated visual bar is rerun under the same gate.
2. **`missing_text`:** approved, routing to `invoke_vlm`. The method record says that `ohr_dev_v2` results informed
   it. `ohr_dev_v2` is not the confirmation sample for this comparison.
3. **Development samples:** `ohr_dev_v1` and `ohr_long_dev_v1` (423 questions, 101 multi-page documents). Only
   global settings are tuned.
4. **Confirmation:** one run after the method freeze on the 835 validation questions on the 259 multi-page documents
   outside `ohr_long_dev_v1`. No validation question is read before it. At 6% disagreement a 5-point gain needs 186
   questions without clustering; at 10%, 312; at 12.9%, 403. With 3.22 questions per document the design effect is
   1.19 to 1.89 for the intraclass correlations seen so far (0.086 and 0.40). Then a 5-point gain needs 480 to 761
   questions at 12.9% disagreement, so 835 questions cover every disagreement rate observed so far.
5. **Cost tolerance** 1.10. The correction model and any new retry expansion are chosen on development data.


## 12. Development runs on `ohr_long_dev_v1`: plan and cost (2026-11-01, nothing authorised)

Built and tested offline: the gate and diagnosis (`src/faar/diagnosis.py`, `scripts/experiments/diagnose_run.py`),
`retry_retrieval` with two expansion rules and `correct_text` with SymSpell (`src/faar/text_repair.py`,
`scripts/experiments/run_text_repair.py`). Both runners use the ledger, cap and resend rule of
`run_visual_repair.py`. ByT5 is not available: it needs a model download of about 1 GB that has not been authorised.

Offline facts on `ohr_long_dev_v1`, from a dry run and a retrieval replay:

- The no-recovery run sends all 423 questions (every document has text). Prompts average 6,962 characters, against
  5,941 on `ohr_dev_v2`.
- The page policy would attach 3 images to 357 questions and 2 to 66 (2.84 on average), against 1.55 on `ohr_dev_v2`.
- On `ohr_dev_v1`, the current expansion rule `append-anchor-words-v0` sends exactly the same chunks again for 38 of
  63 questions. `exclude-seen-v1` finds new chunks for 35 of 63; the rest are documents with no unseen chunk.

Unit costs, measured on the `ohr_dev_v2` runs: a text request about 1,516 input and 15 output tokens (0.0016 USD),
scaled to 0.00185 USD for the longer prompts here; one page image about 2,570 input tokens (0.0026 USD), so a
repair request with 2.84 images costs about 0.0093 USD.

| Step | Requests | Expected USD |
| --- | ---: | ---: |
| 1. No-recovery run (`faar-answer-v2`) on all 423 | 423 | 0.78 |
| 2. Choose the gate's confidence condition (rule below), diagnose | 0 | 0 |
| 3. `retry_retrieval`, `exclude-seen-v1`, on the flagged set | about 150 to 215 | 0.28 to 0.40 |
| 4. `retry_retrieval`, `append-anchor-words-v0`, on the flagged set | about 215 | 0.40 |
| 5. `correct_text`, SymSpell, on the flagged set | about 180 to 215 | 0.33 to 0.40 |
| 6. Visual repair on the flagged set (needed for the all-repairs diagnostic and the bar) | about 215 | 2.00 |
| Total | | about 3.8 to 4.0 |

The flagged set is not known until step 1 runs. The estimate assumes about 215 of 423 flagged: 110 questions on
documents with an empty page, about 35 abstentions and about 70 from the confidence condition.

**This does not fit the 2.29 USD left.** Steps 1 to 5 fit (about 1.9 USD, cap 2.25). Step 6 needs about 2 USD more.
Dropping step 4 saves 0.40 USD but leaves the current expansion rule untested against the new one.

**Proposed rule for choosing the confidence condition, fixed before step 1.** On `ohr_long_dev_v1` only, for each
feature in `CONFIDENCE_FEATURES` and each threshold at the observed feature values, compute how well "feature below
threshold" separates questions whose no-recovery answer is wrong (official EM 0) from those it got right: true
positive rate minus false positive rate (Youden's J). Choose the feature and threshold with the highest J, subject
to the confidence condition flagging at most 30% of the questions that the base gate does not already flag. Ties go
to the lower flag rate. This uses development gold answers only to choose one global threshold, which the brief
allows on development data. It never uses validation.

The confirmation on the 835 validation questions is costed later. At these unit costs it is roughly 1.6 USD for the
no-recovery run, 3 to 4 USD for the gated visual bar, and about 2 USD for FAAR's text repairs and the random policy,
so about 7 to 8 USD.
