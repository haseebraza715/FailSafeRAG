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

## 13. Practice rounds and round 1 decisions (2026-11-01, before step 1 runs)

The research lead approved the practice stage on 2026-11-01: at most 3 practice rounds on `ohr_long_dev_v1`, a
spending cap of 10 USD for the stage counting reserved cost, then the freeze. Rounds 2 and 3 rerun only what
changed. After round 3, tuning stops whatever the result. The study lead fixed the following before step 1 of round 1.

1. **Steps.** Round 1 runs steps 1, 2, 3, 5 and 6 of section 12. Step 4 (`append-anchor-words-v0`) is skipped: it
   resent the same chunks for 38 of 63 questions on `ohr_dev_v1`. `correct_text` uses SymSpell
   (`symspell-en-82765`); ByT5 is not downloaded. Steps 3, 5 and 6 run on the flagged set only.
2. **Confidence condition (Youden's J), exact form.** Population: the questions the base gate
   (`abstain-or-ocr-gap-v1`) does not flag and whose feature value is not null, since only there can the condition
   change the flagged set. Positive: no-recovery official EM 0. For each feature in `CONFIDENCE_FEATURES` and each
   distinct observed value `t` in that population, the condition flags `feature < t`. Candidates that flag more than
   `floor(0.30 × N)` of the `N` questions in the population are dropped. The highest J wins; ties go to the fewer
   flagged questions, then to the earlier feature in `CONFIDENCE_FEATURES`, then to the lower `t`. If no candidate
   has J above 0, the gate stays the base gate. Gold answers of `ohr_long_dev_v1` (train split) are read only for
   this choice; the script refuses any pilot that is not train data.
3. **Diagnosis thresholds in round 1:** the code defaults, `structural_threshold` 1 and `weird_char_threshold` 0.10.
4. **Random repair:** seeds 20261101, 20261102 and 20261103. Each flagged question gets
   `REPAIRS[int(sha256(f"{seed}:{question_id}"), 16) % 3]` with `REPAIRS = (retry_retrieval, correct_text,
   invoke_vlm)`. A drawn repair that skips the question keeps the initial answer.
5. **Per-run ceilings in round 1 (USD, reserved cost):** no-recovery 1.20, `retry_retrieval` 0.70, `correct_text`
   0.70, visual repair 3.50. The stage cap of 10 USD binds over all rounds.
6. **What a round measures.** On development data, every policy of section 3 is assembled from the same runs
   (FAAR, the gated visual bar, random and no recovery), with the all-repairs diagnostic of brief section 6 and the
   costs of brief section 5. These are development results, never a confirmation.

## 14. Frozen method for the confirmation (2026-11-01, before any validation run)

The research lead decided to keep the routing of section 4 and freeze after practice round 2
([faar-practice-rounds.md](faar-practice-rounds.md)). Frozen at code commit `e8c024d`:

| Item | Frozen value |
| --- | --- |
| Answer model | `gpt-6.1-terra`, temperature 0, `reasoning_effort` none, 128 output tokens, store false; provider config sha256 `35ea0799e1931fe0167224b5b05483f8881c91b3d1d0f3055c72910906b3ae6e` |
| Prompt | `faar-answer-v2` (the image repair adds its recorded image note) |
| Retrieval | `multilingual-v1` tokenizer, `cjk-weighted-words-v1` chunks, top 5 (text budget) |
| Gate | `abstain-or-ocr-gap-v1` plus `query_token_coverage < 0.6923076923076923` (chosen in round 1 by `youden-j-cap30-v1`) |
| Diagnosis | section 4 order and routes; `structural_threshold` 1, `weird_char_threshold` 0.10 |
| `retry_retrieval` | `keep-top3-add-next2-v1` |
| `correct_text` | SymSpell `symspell-en-82765` |
| `invoke_vlm` | 3 pages, `retrieved-then-gaps-v1`, 150 DPI, JPEG quality 90, long side 2048, `detail: high` |
| Random repair | seeds 20261101, 20261102, 20261103, section 13 item 4 |
| Execution | resend rule `one-resend-connection-lost-v1`; failure rule of amendment 2 |
| Comparison | `scripts/analysis/faar_confirmation/compare_confirmation.py`, committed before the run, with the outcomes, contrasts, decision and cost rules of section 5 |

**Confirmation run.** Once, on `ohr_val_multipage_v1` (835 validation questions on 259 documents, frozen as
`2026-11-01-ohr-val-multipage-v1-selection`): the no-recovery run, the diagnosis, then `invoke_vlm`,
`retry_retrieval` and `correct_text` on the flagged set, then the comparison script. Spending cap 10 USD counting
reserved cost. A run that breaks amendment 2 is reported as invalid; it is not rerun with changed settings. The
report adds the oracle against the gated repair as descriptive headroom only, not as a claim.

**Amendment to section 14 (2026-11-01, before any validation run).** The final review of the frozen method found that
a no-recovery run with a question still blocked after the amendment 1 resend stopped diagnosis and repair preparation,
so the run could never reach the amendment 2 check. The fix (`ba0f920`, reviewed with no findings) lets such
terminal failures through and changes nothing for runs without them (development results reproduce exactly). It
uses no outcome. The frozen code commit is now `b075dbd`; every other frozen value is unchanged. Registry record
`2026-11-01-faar-frozen-method-v2` supersedes `-v1`.

**Second amendment to section 14 (2026-11-01, before the image repair run).** The research lead raised the
confirmation spending cap from 10 to 12 USD counting reserved cost, with a 7.50 USD ceiling on the `invoke_vlm` run.
The no-recovery, diagnosis and text repair runs had spent 3.581498 USD with no reserved cost, and the image run's
expected 6.5 USD would have left no margin under 10 USD. No validation outcome had been read. Nothing else changes.

## 15. Final test plan (2026-11-01, before any test run)

The research lead approved this plan. It may still overrule it before the test runs.

**Primary sample.** Every eligible OHR-Bench test question on a multi-page document: 946 questions on 327
documents, counted from metadata only. No test document is in `ohr_dev_v1`, `ohr_dev_v2` or `ohr_long_dev_v1`.
252 of the documents also carry questions in `ohr_val_multipage_v1`; the validation sample was used once, for the
confirmation, and never for tuning, and the report states this overlap.

**Secondary sample, reported separately.** The 330 single-page test questions. They were left out of the
confirmation because on a page with empty text FAAR and the gated repair send the same image, so the two policies
tend to give the same answers there. The secondary result shows whether the primary conclusion holds on that part of
the test split. It does not enter the Holm family of the primary result, and it cannot change the primary decision.

**Method.** The frozen method of section 14 at code commit `b075dbd`, unchanged: the same answer model, prompt,
retrieval, gate, diagnosis, repairs, random seeds, resend rule and failure rule.

**Comparison.** The confirmation comparison script with only the pilot and run ids changed, once per sample, each
committed before its sample's first test run. Outcomes, contrasts, the decision rule and the cost rule are those of
section 5. The oracle against the gated repair is reported as descriptive headroom only.

**Order and cost.** Each sample is built once as its own frozen pilot, then gets the no-recovery run, the diagnosis,
the three repairs on the flagged set and its comparison. The primary sample runs first. Spending cap 15 USD for the
whole test stage, counting reserved cost. The confirmation flagged 77% of its questions (647 of 835), more than the
55% assumed in the first estimate, so the primary sample is now expected to cost about 11.5 USD and the secondary
about 3 USD. The secondary sample starts only if its dry-run estimate fits under the remaining cap; if it does not,
the research lead decides whether to raise the cap. A run stopped by its cap or invalid under amendment 2 is reported
as such and is not rerun with changed settings.
