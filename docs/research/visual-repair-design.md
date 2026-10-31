# Design: the first repair experiment, visual repair with a runtime page policy

**Status: proposal for the research lead, written 2026-10-31. Nothing here is built or run.** The experiment runs
only after the lead approves this design, the analysis plan
([repair-analysis-plan.md](repair-analysis-plan.md)) and the sample
([ohr-dev-v2-proposal.md](ohr-dev-v2-proposal.md)).

## Why the visual repair comes first

The headroom runs on `ohr_dev_v1` bound each repair from above, because each uses an evaluation-only input. With the
`faar-answer-v2` prompt
([v2 comparison report](../../results/analysis/2026-10-31-ohr-dev-v1-terra-answer-v2/report.md)):

| Condition against the v2 no-recovery baseline (EM 0.314) | EM difference, 95% interval by document | McNemar gained / lost |
| --- | --- | --- |
| Baseline text plus gold page image(s), all 70 questions | +7.1 [0.0, +17.5] | 7 / 2 |
| The same, on the 67 questions both runs answered | +9.0 [+1.4, +19.6] | 7 / 1 |
| Clean ground-truth text, same retrieval | +1.4 [-7.0, +10.5] | 5 / 4 |
| All MinerU text of the gold page(s) | +1.4 [-3.5, +5.9] | 2 / 1 |

The image condition is the only one whose interval reaches above zero on the questions both runs answered, and it
made 4 of the 10 cases where MinerU lost the evidence exact. Text correction cannot restore text that MinerU never
produced, and retrieval has little to fix on a sample of mostly single-page documents. These are development
results on 70 questions. No contrast passed McNemar's test at 0.05.

The headroom run chose pages by the gold evidence labels. A repair may not do that (study brief sections 4 and 7). The
experiment below measures what a runtime page choice keeps of that bound.

## What the repair does

For one question, the repair sends one request to the answer model. The request holds:

- the system message of `faar-answer-v2` followed by `IMAGE_NOTE` from `scripts/analysis/ohr_dev_v1_failures/headroom.py`;
- the user message of the question's no-recovery request, unchanged: the question, the same retrieved MinerU text
  blocks and the v2 closing line. A question with no text evidence gets `NO_TEXT_USER_PREFIX` and the v2 closing line;
- after the text, each selected page image, preceded by the line `Image of page N:`.

The reply is parsed by `faar.answer_prompt.parse_reply`, as for the initial answer. The repaired answer replaces the
initial answer. Nothing combines or votes between the two.

### Inputs the repair may read

| Input | Source |
| --- | --- |
| Question, document identity, page inventory, per-page OCR status | `results/pilots/<pilot_id>/runtime_manifest.json` |
| Retrieved evidence and the exact initial user message | the no-recovery run's `requests.jsonl` |
| Page images | rendered from the locked PDF archive (`data/ohr_bench_raw/pdfs.zip`), checked against the hash in the runtime manifest |

The repair never opens `evaluation_manifest.json`, the clean reference text or any score. A test asserts that the
repair code path reads none of them, in the way `assert_runtime_boundary` guards the runtime manifest.

## Page selection: `retrieved-then-gaps-v1`

The policy ranks the pages of the question's document and takes the first `B`:

1. Pages of the retrieved chunks, in retrieval rank order, each page once.
2. Pages whose MinerU text is empty or missing (`ocr_status` other than `ok`), in page order.
3. All other pages, in page order.

The first rule covers the pages the text answer drew on. The second covers the pages that text retrieval cannot
see, because a page with no text produces no chunk (study brief section 7). The third fills the budget on short
documents, so a document of `B` pages or fewer is shown whole. A question with no retrieved text starts at the second
rule.

**Proposed budget: `B` = 3 images per request**, rendered at 150 DPI, JPEG quality 90, long side at most 2048 px,
`detail` set to `high`. These are the render settings of the headroom runs. The image budget is a lead decision
(study brief section 14).

**Offline check on `ohr_dev_v1`.** The policy was fixed first and then scored once against the gold evidence pages.
The gold labels scored the policy. They did not choose any page.

| Budget `B` | Questions whose every evidence page is shown | On multi-page documents | Mean images per question |
| ---: | ---: | ---: | ---: |
| 1 | 57 of 70 | 6 of 19 | 1.00 |
| 2 | 65 of 70 | 14 of 19 | 1.27 |
| 3 | 68 of 70 | 17 of 19 | 1.46 |
| 5 | 69 of 70 | 18 of 19 | 1.71 |

All 7 questions with no text evidence get their evidence page at `B` = 3. The check needs no model call. Its known
gap: on a long document, an evidence page with partial OCR text that retrieval missed is shown only if the budget
reaches it in page order.

## Which questions get the repair

The run sends the repair request for **every** question, not only for flagged ones. Each question then has an
initial answer and a repaired answer from the same evidence, which gives two policies without further calls:

- **Visual repair on every question** (the primary policy): the repaired answer for all questions.
- **Gated visual repair** (secondary): the repaired answer where the gate flags the question, the initial answer
  elsewhere. The gate, `abstain-or-ocr-gap-v1`, flags a question when the initial outcome is `no_evidence`, when the
  initial answer is `NO_ANSWER`, or when any page of the document has empty or missing MinerU text. The gate has no
  threshold to tune. It is not the study brief's gate, whose threshold is a lead decision. It is a declared, fixed
  rule for this experiment only.

Comparing diagnosis-based recovery (FAAR) with gate-triggered visual recovery is the study's main question. It
needs the other repairs and the diagnosis, and it is not part of this first experiment.

## Pre-registered comparison

The full analysis rules are in [repair-analysis-plan.md](repair-analysis-plan.md), section 10. In short:

- **Sample.** `ohr_dev_v2` (285 questions, if approved), all questions, as a confirmation sample: nobody inspects its
  outputs before the comparison is computed.
- **Shared initial answer.** A no-recovery run on `ohr_dev_v2` with `faar-answer-v2`, run first and frozen. The
  repair reuses its retrieval and user messages.
- **Primary outcome and contrast.** Official EM, visual repair on every question minus no recovery, paired by
  question. 95% interval from a bootstrap that resamples documents (10,000 resamples, seed 20261031). McNemar's exact
  two-sided test at alpha 0.05.
- **Decision rule.** The repair is reported as raising EM only if McNemar's p is below 0.05 and the document-level
  interval excludes 0. Otherwise the result is reported as it is, with its interval.
- **Secondary.** F1, ANLS and NFKC EM (FAAR-defined); the gated policy against no recovery; repair and damage rates
  with counts (study brief section 8); cost per question; images per request; execution failures.
- **Subgroups, descriptive only.** Questions whose evidence page has empty MinerU text, Han-script questions,
  questions on multi-page documents, and document type.

**Power.** With 285 questions and a discordant rate near the 12.9% seen with the gold image on `ohr_dev_v1`, the test
detects a paired EM gain of about 6 points (7 with a design effect of 1.45). The headroom bound was 7 to 9 points with
perfect page choice. A real gain below 6 points will probably not reach significance, and the report says so in
advance.

## Fixed before the run

The prompt (`faar-answer-v2` and `IMAGE_NOTE`), the page policy and `B`, the render settings, the model and its
settings, the gate rule, the analysis script and its seeds, and the spending cap. The run's identity records them. A
change after the run starts makes a new run, and the earlier one stays in the record.

## Unknown outcomes and the SSL drops

Image requests dropped with `SSLV3_ALERT_BAD_RECORD_MAC` in all four image runs on 2026-10-31: 5, 2, 3 and 1 of 70
requests ended with an unknown outcome, 11 of 280 in all. None of the 394 text-only requests in the six text runs
dropped. The cause is not known. It could be the network path or the request size. An unknown outcome is never
resent (study brief section 15.7), so each drop becomes an `execution_failed` question that scores 0. At the
observed rate, about 4% of the repair requests would fail. Before the repair run, a small engineering check should find the cause or show that the rate stays
low. The maximum execution-failure rate is a lead decision (study brief section 15.7). The primary analysis counts
failures as 0. A second view on the questions both runs answered is reported next to it.

## What has to be built

1. A repair runner that reads only the inputs listed above, renders the selected pages, and sends requests through
   `faar.live_runner`'s provider with the same ledger, cost cap, retry rules and unknown-outcome handling as the
   no-recovery run. `headroom.py` shows the request shape. It reads evaluation files by design, so it cannot be reused
   as is.
2. Tests written from a failure list first: a gold field or the evaluation manifest reaches the repair; the policy
   order or budget differs from this page; a page is rendered from a PDF whose hash differs from the manifest; an
   unknown outcome is resent.
3. A comparison script in the style of `compare_v2.py`, fixed before the run.

## Cost

| Step | Requests | Expected cost (USD) | Basis |
| --- | ---: | ---: | --- |
| No-recovery run on `ohr_dev_v2` | about 245 | about 0.40 | 0.0015 per request on `ohr_dev_v1` |
| Visual repair on `ohr_dev_v2` | 285 | about 1.40 | 0.0041 per request with 1.3 images on `ohr_dev_v1`, scaled to about 1.6 images |
| Debug run of the repair on `ohr_dev_v1` | 70 | about 0.30 | as above |

Image tokens were priced at the recorded text rates, because the image price of `gpt-6.1-terra` was not checked in
documentation. The total is about 2.10 USD. The upper bound that the cost cap reserves per request is several times
higher, so the cap for each run is set from the expected cost plus one request's bound.

## Decisions for the lead

1. Approve, change or reject the page policy `retrieved-then-gaps-v1` and the budget `B` = 3.
2. Approve the gate rule `abstain-or-ocr-gap-v1` as a secondary policy for this experiment.
3. Approve `faar-answer-v2` as the prompt that every policy shares, or name another.
4. Set the maximum execution-failure rate, and decide whether an engineering check of the SSL drops comes first.
5. Approve building the repair runner, then each paid run with its cap.
