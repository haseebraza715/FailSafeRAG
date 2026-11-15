# FAAR research log

This log records everything the FAAR study has tried, in date order, with what happened and what was decided. It is
for the supervisors and for anyone who continues the work. It keeps failed, interrupted and negative attempts in. It
is a record, not a report: the [FAAR study report](../reports/faar-study-report.md) gives the finished result, and
this log shows how the study got there.

[experiments/registry.jsonl](../../experiments/registry.jsonl) is authoritative for run ids, statuses, code commits,
costs and output hashes. If this log and the registry disagree, the registry wins and the log has an error to fix.
Dates below are UTC dates taken from the registry's attempt times. Commit times in `git log` are local time (+0100),
so a commit can fall on the next calendar day.

## How to add an entry

Add one dated entry per run or decision, on the day it happens. Put new entries at the end of "Entries", add one row
to the timeline, and answer five questions in the entry:

1. What was tried?
2. Why?
3. What happened? Give numbers with the file that holds them.
4. What was decided?
5. Which run ids or files back it?

Keep measured results, interpretation and plans visibly apart, as the stage reports do. Record an unknown fact as
unknown. Register a run in the registry first (see [experiments/README.md](../../experiments/README.md)), then
write the entry. Do not edit old entries to change what they said. Add a correction entry that says what changed and
why. Entries that cite a result must cite a registered run.

Spend figures are measured USD unless the entry says "reserved". Reserved cost is the worst-case charge for requests
whose outcome was unknown after a network drop.

## Timeline

| Date | What was tried | Outcome |
| --- | --- | --- |
| 2026-08-03 to 2026-08-08 | Prototype and smoke runs, ArXivQA preparation | Mock-backend engineering evidence only. ArXivQA was deferred |
| 2026-09-28 | Locked the OHR-Bench PDF archive, audited the assets, froze `ohr_dev_v1` | 1117 of 1117 QA documents verified. 30 documents and 70 questions frozen |
| 2026-09-28 to 2026-09-30 | Offline and fake-provider engineering checks | Data flow, scoring, ledger and records work. No answer-quality evidence |
| 2026-09-30 | First live check (gpt-4o) | Abandoned at the first request: invalid API key. Nothing billed |
| 2026-10-31 | Live check (gpt-4o) and baseline (gpt-4o) on `ohr_dev_v1` | Check passed. Baseline ran but was never scored: the model changed to `gpt-6.1-terra` |
| 2026-10-31 | `gpt-6.1-terra` baseline on `ohr_dev_v1`, then failure coding of the 55 non-exact cases | EM 0.214. About half the failures were correct answers that EM scores 0 |
| 2026-10-31 | Headroom runs: clean text, gold page text, gold page image, gold page image only | Image gave the largest gain. Two image runs were interrupted by network drops |
| 2026-10-31 | `faar-answer-v2` prompt | EM 0.214 to 0.314 on `ohr_dev_v1`, tuned on those questions |
| 2026-10-31 | TLS bad-record-MAC check | Large uploads drop about 2.3% of the time. Cannot be fixed in code. Resend rule written |
| 2026-10-31 | Froze `ohr_dev_v2` (414 questions), tested the prompt there | v2 prompt no better than v1 (0.287 against 0.290) |
| 2026-10-31 | Visual repair debug run on `ohr_dev_v1` | EM 0.386 with the runtime page choice |
| 2026-10-31 | Visual repair on `ohr_dev_v2` | +8.5 points over no recovery. The gated image repair matched it at half the cost and became the bar |
| 2026-11-01 | Pre-registration, longer-document sample choice, amendments 2 and 3 | Gate, `missing_text` diagnosis and sample decided. `ohr_long_dev_v1` frozen |
| 2026-11-01 | Practice round 1 | FAAR 0.307 against bar 0.397. The retry rule made answers worse than none |
| 2026-11-01 | Practice round 2 | New retry rule. FAAR 0.343 against bar 0.397 |
| 2026-11-01 | Method freeze, then the terminal-baseline fix and re-freeze | Frozen at code commit `b075dbd` |
| 2026-11-01 | Confirmation on 835 validation questions | FAAR 0.283 against bar 0.327. Rule not met |
| 2026-11-01 | Final test, primary (946) and secondary (330) samples | Primary: FAAR 0.266 against bar 0.315. Rule not met. Secondary: -1.8 points, not significant |
| 2026-11-01 | Independent rescoring | Reported as matching for every policy |
| 2026-11-15 | Exploratory offline analysis of the saved runs | Repairs help mainly when the first answer abstained. A two-step rule scored above the bar. Post hoc, unconfirmed |

## Entries

### 2026-08-03 to 2026-08-08: prototype work before the study

**What was tried.** Three early runs, all before the OHR-Bench protocol: a one-document smoke of the B0 baseline
(`2026-08-03-b0-one-doc-smoke`), a five-profile prototype comparison on 40 examples with a mock visual backend
(`proto-phase3-mock-40`), and staging of 500 ArXivQA questions (`2026-08-08-arxivqa-full-paper-prep`).

**What happened.** The prototype comparison tied every profile at EM 0.1250 and F1 0.2036 on 40 examples. The
ArXivQA PDFs, page images and OCR files are not in this checkout, and all 500 evidence-page mappings are unresolved.

**Decided.** The runs are engineering evidence only. ArXivQA was deferred from the first study. Older numbers from
these runs are in `docs/history/` and are not baselines.

**Backed by.** The three run ids above in the registry.

### 2026-09-28: data audit and the frozen `ohr_dev_v1` sample

**What was tried.** Obtain and lock the official OHR-Bench PDF archive (`2026-09-28-ohr-pdf-archive`), audit every
asset (`2026-09-28-ohr-asset-audit`), and freeze a small development sample with no model, OCR or retrieval call
(`2026-09-28-ohr-dev-v1-selection`).

**Why.** The study needed a locked input set and a development sample that no later decision could bend.

**What happened.** From [data_audit.md](../reports/data_audit.md): the archive is 1516951813 bytes and its SHA-256
matches the Hugging Face checksum. Each of the 1117 QA documents maps to exactly one archive PDF, and their pages
cover 8400 pages in total. The audit exited 0 with no integrity failures.

The sample, from [pilot_readiness.md](../reports/pilot_readiness.md): seed 42, 30 train-exclusive documents
stratified by document type, with all their 70 train questions. 23 of the 30 documents are single pages. The sample
has no chart questions. The registry records that a first freeze, a few minutes old and unused, was replaced after a
naming rule changed only the descriptive section of the record. The selected documents, question ids and manifest
digests did not change.

**Decided.** `ohr_dev_v1` is a development pilot for mechanism work, not a benchmark. `config/datasets/ohr_split.json`
and `OHR-Bench/data/qas_v2.json` stay locked.

**Backed by.** The three run ids above, `docs/reports/data_audit.md`, `docs/reports/pilot_readiness.md`.

### 2026-09-28 to 2026-09-30: offline and fake-provider engineering checks

**What was tried.** Four offline runs of the data flow on `ohr_dev_v1` (`2026-09-28-ohr-dev-v1-offline-engineering`,
`-r2`, `-r3`, `-r4`), six runs of the answer-model path with an in-process fake provider
(`2026-09-29-ohr-dev-v1-fake-provider-r1` to `-r4`, `2026-09-30-ohr-dev-v1-fake-provider-r5` and `-r6`), and two
rehearsals of an 8-question live check with the fake provider (`2026-09-30-ohr-dev-v1-live-check-subset-fake-r1`
and `-r2`).

**Why.** To check runtime inputs, retrieval, one terminal status per question, the safety ledger, export and the
join to scoring before any money was spent.

**What happened.** Every offline run reached 70 questions: 63 answered, 7 `no_evidence`, 0 `execution_failed`. EM was
4 of 70 in `-r2` and `-r3`. The checks found real problems that later runs fixed. In `-r3`, 19 questions had no
retrieval token because the tokenizer indexed only `[a-z0-9%$]`, so Han-script text could not be searched. The
multilingual retrieval change in `-r4` brought that to 0, and 14 answers changed, all on Han-script questions
(F1 0.0960 to 0.1142). The fake-provider runs ended with 63 answered at 0.221480 simulated USD. Their scores are of
fictional answers and measure nothing. Request and response ids changed between runs because the run identity changes
with the code. A refactor to remove dead code (`b7241a0`) changed every `src/faar` file and with it the identity.

**Decided.** These are `engineering_check` runs. They are not baselines and carry no claim about answer quality.

**Backed by.** The registry records above.

### 2026-09-30: the first live check was abandoned, and 2026-10-31: its successor

**What was tried.** The first live check with a real provider (`2026-09-30-ohr-dev-v1-live-check-r1`): 8 questions to
`gpt-4o-2024-11-20` under a 0.15 USD ceiling. It measured transport, accounting and record correctness, not answers.

**What happened.** The first request was rejected with HTTP 401 `invalid_api_key`. The runner stopped. Measured cost
was 0.000000 USD, with 0.019465 USD reserved for the rejected attempt at its bound. The run was then abandoned: the
deslop refactor had changed the run identity, so the runner refused to resume the directory.

The successor, `2026-10-31-ohr-dev-v1-live-check-r2`, ran on 2026-10-31 on code whose `src/` and `scripts/` are
identical to the approved `b7241a0`, with the same configuration and ceiling. It answered 6 questions, with 2
`no_evidence`, 0 failures, at 0.019360 USD. The returned model matched on every request.

**Decided.** The abandoned record stays as written. The live check passed and opened the way to a first baseline.

**Backed by.** `2026-09-30-ohr-dev-v1-live-check-r1` (abandoned), `2026-10-31-ohr-dev-v1-live-check-r2`.

### 2026-10-31: the gpt-4o baseline that was not scored, and the switch to gpt-6.1-terra

**What was tried.** The first real-model baseline, no recovery on all of `ohr_dev_v1`, with `gpt-4o-2024-11-20`
(`2026-10-31-ohr-dev-v1-baseline-r1`).

**What happened.** It completed: 63 answered, 7 `no_evidence`, 0 failures, 0.222571 USD. It was never scored. The research lead
chose `gpt-6.1-terra` as the answer model on the same day, because `gpt-4o-2024-11-20` shuts down on 2027-01-14 and
the run could not be repeated after that date. The lead also asked for no gpt-4o comparison. A live check of the new
model followed (`2026-10-31-ohr-dev-v1-live-check-terra-r1`): 6 answered, 0 failures, 0.007823 USD, every returned
model `gpt-6.1-terra`.

**Decided.** The answer model is `gpt-6.1-terra` with temperature 0, `reasoning_effort` none and 128 output tokens.
`gpt-6.1-terra` is an alias with no dated snapshot, so a rerun may not reproduce its answers. Each response records
the returned model string. The gpt-4o directory stays unscored and can be scored later.

**Backed by.** The two run ids above. One registry detail: the first snapshots of the terra live check copied
fields from the gpt-4o baseline record by mistake. The snapshot was corrected on 2026-10-31 (kind
`engineering_check`, 8-question subset). The run itself, its outputs and hashes did not change.

### 2026-10-31: the terra baseline on `ohr_dev_v1` and its failure analysis

**What was tried.** The no-recovery baseline on all 70 questions with `gpt-6.1-terra`
(`2026-10-31-ohr-dev-v1-baseline-terra-r1`), then one agent coder labelled the 55 non-exact cases under a codebook
([failure report](../../results/analysis/2026-10-31-ohr-dev-v1-terra-failures/report.md)).

**What happened.** EM 0.214, F1 0.468, ANLS 0.380, at 0.085587 USD. Primary cause of the 55 non-exact cases, as coded:

| Primary cause | Cases |
| --- | ---: |
| Answer correct, EM cannot see it (wording, units, list format, notation, Unicode) | 27 |
| Gold answer or question questionable | 7 |
| OCR lost the evidence | 10 |
| OCR damaged the evidence | 3 |
| Retrieval missed the page (1) or the passage (1) | 2 |
| Evidence in the prompt, model wrong or abstained | 6 |

**Decided.** Treat the coding as a working assumption: one AI coder, no second labeller, 70 questions from 30
documents. The failure analysis shaped three things: a prompt change for answer format, the headroom runs, and the
plan to test an image repair first.

**Backed by.** `results/analysis/2026-10-31-ohr-dev-v1-terra-failures/`. That report still shows the two image
headroom conditions as pending, because the image runs were interrupted when it was written. The answer-format
report below has their final rows.

### 2026-10-31: headroom runs on `ohr_dev_v1`

**What was tried.** Four diagnostic runs that each give the answer model an evaluation-only input, so each bounds one
repair from above and none is a method:

- clean ground-truth text with the baseline's retrieval (`headroom-clean-text`)
- all MinerU text of the gold evidence pages (`headroom-gold-page-text`)
- the baseline text plus the gold page images (`headroom-gold-page-image`)
- the gold page images only (`headroom-gold-page-image-only`)

**What happened (first prompt, `faar-answer-draft-v1`).** Clean text and gold page text completed. The two image runs
were interrupted: 5 of 70 and 2 of 70 requests ended with `connection_lost` (SSL bad record MAC) and an unknown
outcome. The runner does not resend an unknown outcome, because a resend could be charged twice and `store=false`
means no lookup. Over all 70 questions, with blocked questions scored 0, EM was 0.271 for clean text, 0.229 for gold
page text, 0.243 for the image and 0.271 for the image only, against 0.214 for the baseline.

**Decided.** No contrast could be called a result: 70 questions cannot detect a 5-point gain. The interrupted runs
stay in the registry as `interrupted`. The two drops are the first sign of the network problem below.

**Backed by.** The four `2026-10-31-ohr-dev-v1-headroom-*-terra-r1` runs,
[answer-format report](../../results/analysis/2026-10-31-ohr-dev-v1-terra-answer-v2/report.md), table 1.

### 2026-10-31: the `faar-answer-v2` prompt on `ohr_dev_v1`

**What was tried.** A new answer template, `faar-answer-v2`, written after reading the coded failures, to fix the
format cases. The baseline and all four headroom runs were repeated with it (five `answer-v2` runs).

**What happened.** The baseline rose from EM 0.214 to 0.314 (McNemar 8 gained, 1 lost, p 0.039, document interval
[-1.5, +22.8]). All the gain came from `scoring_artifact` cases: 8 of 27 became exact. 19 of the 27 stay at 0 because
the reply and the gold differ in where the span starts or ends, and no prompt can know that. Under v2 the headroom
conditions scored: clean text 0.329, gold page text 0.329, gold page and image 0.386, image only 0.357. The image added
+7.1 points over all 70 questions ([0.0, +17.5], 7 gained, 2 lost) and +9.0 points on the 67 both runs answered
([+1.4, +19.6]). The two image runs were interrupted again (3 and 1 unknown outcomes). The first one stopped after 3
consecutive drops on its first 4 requests and was resumed once for the unsent questions.

**Decided.** The image is the only evidence repair that clearly moved EM, so it is built first. The v2 gain is
optimistic because v2 was written after reading these cases. That question went to a test on new data (see the
`ohr_dev_v2` entries).

**Backed by.** The five `2026-10-31-ohr-dev-v1-*-answer-v2-terra-r1` runs, the
[answer-format report](../../results/analysis/2026-10-31-ohr-dev-v1-terra-answer-v2/report.md), and
[visual-repair-design.md](visual-repair-design.md).

### 2026-10-31: the TLS bad-record-MAC drops, the resend rule and the failure rule

**What was tried.** An engineering check, with no model call and nothing billed, of why 11 of 280 image requests
ended in `SSLV3_ALERT_BAD_RECORD_MAC` and none of 394 text requests did. It sent the same kind of body through the
runner's client, through a fresh connection each time, and through curl on a different TLS library.

**What happened.** Large bodies dropped 9 of 400 times (2.3%) and small ones 0 of 200. Fresh connections dropped too,
and so did curl, so connection reuse and the client library are not the cause. Every request used HTTP/1.1. The
report's interpretation is that bytes of the upload are corrupted on the local network path, below TLS.

**Decided.** Amendment 1 of [repair-analysis-plan.md](repair-analysis-plan.md), the resend rule
`one-resend-connection-lost-v1`: a `connection_lost` attempt with an unknown outcome may be sent once more, the first
attempt stays counted at its upper bound, and a second unknown outcome blocks the question. Shrinking the images was
not chosen because it would change the pre-registered image settings. Amendment 2, dated 2026-11-01, added the
failure rule: a run is invalid if more than 2% of its questions stay blocked after the resend, or if two compared
conditions differ in blocked rate by more than 1 percentage point.

**Backed by.** `2026-10-31-tls-bad-record-mac-check` and
[the check report](../../results/engineering/2026-10-31-tls-bad-record-mac-check/report.md). A change of network
or a wired connection might remove the drops. That was not tested.

### 2026-10-31: `ohr_dev_v2` frozen and the prompt test

**What was tried.** A larger development pilot, because 70 questions cannot show a 5-point difference. The lead raised
the proposed size from 138 documents to 201. The selection excluded the documents of `ohr_dev_v1`
(`2026-10-31-ohr-dev-v2-selection`). Then two no-recovery runs on it, one per prompt, with the same retrieval.

**What happened.** `ohr_dev_v2` has 414 questions on 201 documents, all train-only. 128 of its questions are on
multi-page documents and 56 have an evidence page with empty MinerU text. 208 train-only questions were left unused.
v2 against v1 on the new questions: EM 0.287 against 0.290, a difference of -0.2 points
([-3.3, +3.2], 19 gained, 20 lost). Costs were 0.545853 USD for v2 and 0.521591 USD for v1.

**Decided.** The +10 points on `ohr_dev_v1` came from tuning on those questions. Amendment 2 froze `faar-answer-v2`
for every shared run anyway, because every repair run already used it, and the documents say plainly that it is no
better than v1.

**Backed by.** `2026-10-31-ohr-dev-v2-baseline-answer-v1-terra-r1`, `2026-10-31-ohr-dev-v2-baseline-answer-v2-terra-r1`,
[ohr-dev-v2-proposal.md](ohr-dev-v2-proposal.md), the
[`ohr_dev_v2` report](../../results/analysis/2026-10-31-ohr-dev-v2-confirmation/report.md).

### 2026-10-31: the visual repair debug run on `ohr_dev_v1`

**What was tried.** The visual repair runner (`invoke_vlm`) on every `ohr_dev_v1` question, as a debug run on
development data before the run that counts. It picks up to 3 page images at runtime with the page policy
`retrieved-then-gaps-v1`, with no gold label.

**What happened.** 70 answered after 2 `connection_lost` drops, both resent once and saved. EM 0.386, the same as the
gold-page image bound of 0.386. The gated version (`abstain-or-ocr-gap-v1`) scored 0.400. Contrast against no
recovery: +7.1 points for repair on every question ([0.0, +17.5], 7 gained, 2 lost), and +8.6 for the gated repair
([+2.7, +17.2], 6 gained, 0 lost). Cost was 0.344581 USD plus 0.045303 USD reserved.

**Decided.** The runner and the resend rule work. The debug result is development evidence only.

**Backed by.** `2026-10-31-ohr-dev-v1-visual-repair-answer-v2-terra-r1` and the
[debug report](../../results/analysis/2026-10-31-ohr-dev-v1-visual-repair-debug/report.md).

### 2026-10-31 and 2026-11-01: the gated image repair becomes the bar

**What was tried.** The pre-registered first experiment of the analysis plan: visual repair on every one of the 414
`ohr_dev_v2` questions, with the gated repair computed from the same answers.

**What happened.** 414 answered after 6 `connection_lost` drops, all resent once, none blocked. EM rose from 0.287 to
0.372, +8.5 points ([+5.1, +12.3], 42 gained, 7 lost, p 3.6e-07), so the pre-registered rule was met. The gated repair
gave the same EM, +8.5 ([+5.6, +11.8], 36 gained, 1 lost), at 0.00269 USD per question against 0.00545 for repair on
every question. Measured cost was 2.147518 USD plus 0.108866 USD reserved.

Where the gain sat, from the interpretation note: all 42 gains were on single-page documents. On the 128 multi-page
questions the repair gained 0 and lost 4. 27 of the 42 gains were questions with no text evidence at all. So the sample
shows that the image helps when MinerU text is empty. It cannot show that the page policy chooses the right page.

**Decided.** Amendment 2 (2026-11-01): the gated visual repair is the comparator for a diagnosis-based policy. It is
"the bar". A longer-document sample comes before that comparison.

**Backed by.** `2026-10-31-ohr-dev-v2-visual-repair-answer-v2-terra-r1`,
[the `ohr_dev_v2` report and its interpretation](../../results/analysis/2026-10-31-ohr-dev-v2-confirmation/interpretation.md),
[repair-analysis-plan.md](repair-analysis-plan.md) section 13.

### 2026-11-01: the pre-registration and its two starting problems

**What was tried.** A draft pre-registration of FAAR against the gated image repair
([faar-vs-gated-preregistration.md](faar-vs-gated-preregistration.md)). It named two problems before anything ran.

1. **Little room to differ under the old gate.** `abstain-or-ocr-gap-v1` flagged 131 of 414 `ohr_dev_v2` questions: 71
   with no text evidence and 60 answered ones. On the 71, only the image can help, so FAAR and the bar give the same
   answers there. They could differ only on the 60. The draft judged that the comparison would very likely be null,
   a test of almost nothing.
2. **No-text questions routed to retry.** The existing controller returned `semantic` when retrieval found nothing,
   and `semantic` maps to `retry_retrieval`. A retry on a document with no text finds nothing again. By construction
   FAAR would lose the 27 gains the image made on no-text questions.

**Decided (lead, 2026-11-01, amendment 3).** A retrieval-confidence condition is added to the gate, chosen on
development data only, and the bar is rerun under the same gate. A new `missing_text` diagnosis routes to the image
repair. It was approved knowing that `ohr_dev_v2` results informed it, so `ohr_dev_v2` is not the confirmation sample.
The cost tolerance for "similar cost" is 1.10 times the bar.

**Backed by.** The pre-registration sections 1, 9 and 11, and [repair-analysis-plan.md](repair-analysis-plan.md)
section 14.

### 2026-11-01: the longer-document sample

**What was tried.** An options report on where a longer-document sample could come from
([longer-document-sample-options.md](longer-document-sample-options.md)). Counts came from metadata only.

**What happened.** Train-only questions on multi-page documents outside both pilots number 59 on 20 documents, too few.
Empty-OCR evidence pages sit almost entirely on single-page documents, so no OHR-Bench sample can test "find the right
page when its text is missing" with useful power. Option F is the only option that keeps test documents unseen: all
423 train questions on 101 multi-page documents with no test question. Its price is that 81 of its documents also
carry validation questions, so validation is no longer document-disjoint from development.

**Decided.** Option F became `ohr_long_dev_v1` (`2026-11-01-ohr-long-dev-v1-selection`). The 133 validation questions on
its documents are excluded from both manifests and were not read. Only global settings may be tuned on it. The
confirmation sample is the 835 validation questions on 259 multi-page documents outside it. MP-DocVQA was not brought
forward.

**Backed by.** The options report, pre-registration section 11 and `2026-11-01-ohr-long-dev-v1-selection`,
`2026-11-01-ohr-val-multipage-v1-selection`.

### 2026-11-01: practice round 1

**What was tried.** Steps 1 to 3, 5 and 6 of pre-registration section 12 on `ohr_long_dev_v1`: the no-recovery run,
the Youden's J choice of the gate's confidence condition, `retry_retrieval` with `exclude-seen-v1`, `correct_text`
with SymSpell, and the image repair. The cost estimate first showed that step 6 did not fit the 2.29 USD then
left. The lead approved a practice stage of at most 3 rounds with a 10 USD cap. `append-anchor-words-v0` was skipped
because it resent the same chunks for 38 of 63 questions on `ohr_dev_v1`.

**What happened.** The chosen gate condition was `query_token_coverage < 0.6923076923076923` (J 0.116). The gate
flagged 228 of 423 questions. Official EM: no recovery 0.310, bar 0.397, FAAR 0.307. FAAR minus bar -0.090 (interval
[-0.132, -0.051]), and FAAR minus no recovery -0.002. The retry rule made things worse: `retry_retrieval` on every
flagged question scored 0.284, below no recovery, and on its own route it scored 0.075 against 0.124 for the initial
answer. The image scored highest on every route. 14 image questions had connection drops and were resent. Round 1
spent 3.282597 USD measured, 3.695924 USD counting reserved cost.

**Decided.** `exclude-seen-v1` drops all 5 initial chunks, which may explain the loss. Round 2 changes only the retry
rule.

**Backed by.** The `2026-11-01-ohr-long-dev-v1-*` runs of round 1,
[faar-practice-rounds.md](faar-practice-rounds.md), `results/analysis/2026-11-01-ohr-long-dev-v1-round-1/`.

### 2026-11-01: practice round 2

**What was tried.** Only `retry_retrieval`, with `keep-top3-add-next2-v1`: keep the initial top 3 chunks and add the
2 best chunks the first retrieval did not send.

**What happened.** FAAR rose from 0.307 to 0.343, above no recovery (0.310) and 0.054 below the bar (0.397) (interval
[-0.092, -0.022]). Retry on its own route rose from 0.075 to 0.168 EM. The best repair per question in hindsight
scored 0.409. Round 2 cost 0.342294 USD, so the stage total was 3.624891 USD measured, 4.038218 USD with reserved cost.

**Decided (lead, 2026-11-01).** Keep the routing as pre-registered and freeze the method. No round 3. One approved
setting was not tested: a higher `weird_char_threshold` that would move the 44 `word_level` cases to retry. It would
change about 5 answers, not the conclusion.

**Backed by.** `2026-11-01-ohr-long-dev-v1-r2-retry-keep-top3-terra`,
`results/analysis/2026-11-01-ohr-long-dev-v1-round-2/`, [faar-practice-rounds.md](faar-practice-rounds.md).

### 2026-11-01: the method freeze and the terminal-baseline fix

**What was tried.** Freeze every method choice for the one confirmation run. The first freeze record
(`2026-11-01-faar-frozen-method-v1`) used code commit `e8c024d`. A final review then found that a no-recovery run with
a question still blocked after the resend stopped diagnosis, so the run could never reach the amendment 2 check.

**What happened.** The fix (`ba0f920`, reviewed with no findings) lets such terminal failures through and changes
nothing for runs without them. Development results reproduce exactly. It uses no outcome.

**Decided.** The frozen code commit is now `b075dbd`, and `2026-11-01-faar-frozen-method-v2` supersedes `-v1`.
Every other frozen value is unchanged: model, prompt, retrieval, gate, diagnosis thresholds, repairs, random seeds
20261101 to 20261103, resend rule and comparison script. The registry notes one limit: `ohr_long_dev_v1` shares 81
documents with validation, so validation retrieval may be optimistic.

**Backed by.** Pre-registration section 14 and the two freeze records.

### 2026-11-01: the confirmation on `ohr_val_multipage_v1`

**What was tried.** One run of every policy on 835 validation questions on 259 multi-page documents, with the frozen
method. The gate flagged 647 of 835.

**What happened.** Official EM: no recovery 0.250, bar 0.327, FAAR 0.283. FAAR minus bar -0.044 ([-0.068, -0.021], Holm
p 0.0003), so the rule was not met and the bar was significantly better. FAAR minus no recovery +0.032, and minus random
repair -0.002. FAAR cost 0.35 times the bar. Hindsight best per question: 0.369.

Before the image run, the lead raised the confirmation cap from 10 to 12 USD counting reserved cost, with a 7.50 USD
ceiling on the image run. The first three runs had spent 3.581498 USD, and the image run's expected 6.5 USD would have
left no margin under 10. No validation outcome had been read.

The image run took 5 invocations. Invocations 1 to 3 each stopped after 3 consecutive TLS drops and were resumed with
the same settings. There were 25 unknown outcomes on 22 questions, 22 resends and 3 blocked questions (0.46%), under both
amendment 2 limits. A free probe before resuming dropped 4 of 80 large bodies. Total spend was 9.243314 USD measured,
10.038311 USD counting reserved cost.

**Decided.** The result is reported as it came. The run is not repeated. Final-test planning went ahead.

**Backed by.** The `2026-11-01-ohr-val-multipage-v1-*` runs, [faar-confirmation.md](faar-confirmation.md),
`results/analysis/2026-11-01-ohr-val-multipage-v1-confirmation/`, pre-registration section 14.

### 2026-11-01: the final test

**What was tried.** The plan of pre-registration section 15, run once. The primary sample is all 946 eligible test
questions on 327 multi-page documents. The secondary sample is 330 single-page test questions, reported separately and
outside the Holm family. Both samples were built at 11:57 UTC and frozen. The comparison script was committed before
the first test run.

**What happened (primary).** The gate flagged 706 questions, and the diagnosis sent 541 to retry, 136 to text correction
and 29 to the image repair. Every request completed. Official EM: no recovery 0.236, bar 0.315, FAAR 0.266, random
0.274 to 0.278, retry on every flagged question 0.265, hindsight best 0.346. FAAR minus bar -4.9 points
([-7.3, -2.6], 24 gained, 70 lost, Holm p 6.6e-6). FAAR minus no recovery +3.1. FAAR minus random -0.9 (Holm p 0.43).
FAAR cost 0.39 times the bar. Per route, the image scored highest on all three routes. SymSpell correction never beat
the initial answer on its own route.

**What happened (secondary).** No recovery 0.294, bar 0.345, FAAR 0.327. FAAR minus bar -1.8 points
([-4.3, +0.6], 5 gained, 11 lost, Holm p 0.42), not significant. 23 of the 330 questions had no text in the noisy input.

**Cost.** The primary sample cost 10.255497 USD, the stage 11.937165 USD of the 15 USD cap, with no reserved cost.
The secondary sample started because its expected cost (about 2.9 USD) fit under the 4.744503 USD left.

**Decided.** The pre-registered rule is not met. FAAR does not beat the gated image repair. The study ends, and the
runs are not repeated with changed settings.

**Backed by.** The `2026-11-01-ohr-test-*` runs, [faar-final-test.md](faar-final-test.md),
`results/analysis/2026-11-01-ohr-test-multipage-v1-final-test/` and `...-singlepage-v1-final-test/`.

### 2026-11-01: independent rescoring

**What was tried.** A second reviewer, who did not write the analysis code, rescored from the raw model replies with an
independent scorer.

**What happened.** The study report says every policy score matched and no question scored differently, for the
confirmation and both test samples, and that the practice-round numbers were recomputed. The stage documents record
less: the practice-round report records the recomputation for rounds 1 and 2, and the final-test document records the
rescoring of the primary sample only (24 gained and 70 lost for FAAR against the bar). No checked file records the date of
the rescoring. The study report that states it was committed on 2026-11-01.

**Decided.** Nothing changed. The scorer's own provenance is in `docs/reports/ohr-scorer-provenance.md`.

**Backed by.** [faar-study-report.md](../reports/faar-study-report.md), [faar-final-test.md](faar-final-test.md).

### 2026-11-15: the exploratory router analysis

**What was tried.** An offline analysis of the saved runs, with no model call. It asked which flagged questions need the
image, whether runtime signals can predict that, what two simple fixed rules would have scored, and what the gate's
flagged questions gain or lose from repair. Features are runtime-only (68, none from the evaluation manifest). Models
were fit on the practice and validation questions and applied once to test. Intervals resample documents (seed
20261115).

**Why.** The final test left a gap: the best repair per question in hindsight scored 0.346 EM at 0.00244 USD, above the
bar at lower cost. The supervisors wanted to know how much of that gap a router could reach.

**What happened.** Details are in the [analysis report](../../results/analysis/2026-11-15-router-signal-exploration/report.md)
and are summarised in "Observations and findings" below. In short: a signal exists but is modest and mostly the
abstention flag. A learned router saves cost and does not gain accuracy. Two fixed rules written down before their test
numbers were computed did better: image only when the first answer abstained, and retry retrieval first with the image
only if the retry still abstains. The second scored 0.3383 EM on test against 0.3150 for the bar.

**Decided.** Nothing about the study result changes. The rules were found after the test results were seen, and they
are unconfirmed. The run is registered as `2026-11-15-router-signal-exploration` (kind `development_pilot`). A new
section in the study report states this. Any claim needs a fresh pre-registered run on documents not used so far.

**Backed by.** `2026-11-15-router-signal-exploration`, `results/analysis/2026-11-15-router-signal-exploration/`
(`analyse.py`, `report.md`, `results.json`, committed at `47ac998`).

## Observations and findings

### Measured facts

Each fact below comes from a registered run and the file named.

**The study result.**

- On 946 multi-page test questions, FAAR scored EM 0.266, the gated image repair 0.315, no recovery 0.236. FAAR minus
  the bar was -4.9 points, interval [-7.3, -2.6], Holm p 6.6e-6. The same direction held in the practice rounds
  (-5.4 points in round 2), in the confirmation (-4.4) and, not significantly, in the secondary sample (-1.8).
  ([faar-study-report.md](../reports/faar-study-report.md))
- FAAR beat no recovery by +3.1 points on the primary test sample and cost 0.39 times the bar per question. It scored
  the same as always retrying retrieval (0.266 and 0.265) and slightly below a random choice of repair. (same file)
- On the primary test sample, the image repair scored highest on each of the three routes: 0.231 against 0.177 for retry
  on the 541 retry questions, 0.338 against 0.250 on the 136 text-correction questions, and 0.448 against 0.241 on the 29
  image questions. (same file, "Why FAAR lost")

**Development.**

- The failure coding put 27 of 55 non-exact `ohr_dev_v1` cases down to correct answers that EM scores 0. OCR was the
  primary cause in 13 of 55. ([failure report](../../results/analysis/2026-10-31-ohr-dev-v1-terra-failures/report.md))
- The v2 prompt raised `ohr_dev_v1` EM from 0.214 to 0.314 and left `ohr_dev_v2` unchanged (0.290 to 0.287).
- On `ohr_dev_v2` the visual repair added +8.5 points. All 42 gains were on single-page documents, 27 of them on
  questions with no text evidence. On multi-page documents it gained 0 and lost 4.
- The retry rule `exclude-seen-v1` scored 0.075 on its own route against 0.124 for the initial answer. `keep-top3-add-next2-v1`
  scored 0.168. ([faar-practice-rounds.md](faar-practice-rounds.md))

**Network and cost.**

- Image requests with large bodies drop about 2.3% of the time on this network path. No text-only request dropped in the first check (394 requests) or in the `ohr_dev_v2` runs (686). The drops are corrupted uploads, below TLS, and cannot be fixed in the runner.
- Total spend of the FAAR stages: 24.805370 USD measured and 26.013694 USD counting reserved cost, of the 40 USD budget
  ([faar-final-test.md](faar-final-test.md)). The earlier `ohr_dev_v1` and `ohr_dev_v2` pilots are recorded per run in
  the registry and are not in that total.

**Exploratory analysis of 2026-11-15** ([report](../../results/analysis/2026-11-15-router-signal-exploration/report.md)).

- Among flagged questions on test, 27 were answered correctly only by a text repair, 71 only by the image, 113 by both
  and 495 by neither. On validation the counts are 35, 57, 118 and 437, and on practice 5, 28, 48 and 147.
- 252 of 327 test documents also carry validation questions. No question id is shared. 0 are in the practice sample.
- 68 of the 71 image-only questions on test had an initial answer that abstained. The abstention flag alone has AUC
  0.665 on test for "image needed".
- A logistic router (fallback retry) at about 40% of flagged questions sent to the image scored EM 0.3214 at 0.00538 USD
  on test, against 0.3150 at 0.00831 for the bar. Its gain over random at the same share was +0.0344
  [+0.0204, +0.0498]. The best operating point on test, 0.3224, sent 53% of flagged questions to the image. No point
  came near the hindsight best of 0.346 at 0.00244 USD.
- Image only when the first answer abstained: 0.3161 EM at 0.00612 USD on test (bar 0.3150 at 0.00831), a difference
  of +0.0011 [-0.0064, +0.0085].
- Retry retrieval first, then image if the retry abstains: 0.3383 EM at 0.00687 USD on test, +0.0233
  [+0.0111, +0.0350] against the bar. On practice and validation pooled it was +0.0167 [+0.0033, +0.0305]. On the 75
  test documents not in validation it was +0.0248 [-0.0091, +0.0597].
- The gate's 239 test questions flagged without an abstention had 108 right at first. Repairs fixed 6 of the wrong ones,
  and at least one repair broke 14 right ones. The 467 abstained questions had 1 right at first and 98 fixed by some
  repair (35 + 23 + 17 + 23).

### Interpretation

These are readings of the measured facts. They are not results.

- With these repairs, diagnosing the failure type does not help. The image is the strongest repair on every type the
  diagnosis recognises, so sending some questions to cheaper text repairs costs accuracy. FAAR's real advantage is
  cost, at about 0.4 times the bar.
- Most of the help from any repair goes to questions where the first answer abstained. That points to "did the model
  abstain" as the useful signal and away from OCR defect type. The document-level OCR-gap and low-coverage gate reasons
  flag questions that repairs rarely fix.
- A learned router on the available features is a cost saver and not an accuracy gain. It holds roughly the bar's EM at
  about two thirds of the cost, mostly through the abstention flag.
- The cascade scored above the bar because the retry's own abstention is a stronger signal than anything known before a
  repair runs. Whether that holds on new documents is open.
- On the `ohr_dev_v2` sample the image's gain was mostly "show the page when the OCR text is empty". It was not evidence
  of good page choice, and `ohr_dev_v1` and `ohr_dev_v2` were too short to test page choice.

### Open and unconfirmed

- **The two-step rule** (retry retrieval, then image only if the retry still abstains) scored above the bar on test
  and on practice plus validation. On test it cost 0.00687 USD per question against 0.00831 for the bar. It was found
  after the test results were seen, so it is unconfirmed.
- **The abstention-only gate** (repair only when the first answer abstained) matched the bar's EM at about three
  quarters of its cost on test. It is also post hoc. On the 75 new test documents it scored 0.3636 against 0.3802, a
  difference of -0.0165 [-0.0504, +0.0161].
- **Why a fresh run is needed.** Both rules came from data the study had already used, and their test numbers were read
  after the test result. The learned routers were trained on validation questions, and 252 of 327 test documents also
  carry validation questions, so the test AUCs are not document-independent. The 75 new documents hold 65 flagged
  questions and only 6 image-needed ones, too few to say anything. The cross-validated AUC (0.689 logistic, 0.704
  boosting) is the document-independent figure. A confirmation would need a rule and a decision rule written down
  first, and documents that none of the practice, validation or test samples used.
- **A source of such documents** is not settled. The 208 train-only questions left after `ohr_dev_v2` hold 59
  multi-page questions on 20 documents, too few to detect a 5-point difference by the options report. This is a point
  for the research lead.
- **Not tested:** whether the result holds with another answer model (the model is an undated alias), another OCR
  system or another benchmark. A query-rewriting retry and a learned correction model were not tried. The human
  agreement check of the diagnosis was not run.
- **Coding caveat.** The failure coding on `ohr_dev_v1` is one AI coder's judgement on 70 questions. No second labeller
  has coded the cases.

## What was not done

- **Cluster runs.** The study used the OpenAI API only. The cluster calibration on one 108-page document
  (`faar-ohr-108-calibration`) is still `planned` in the registry and was never submitted. The B0 to B4 paper runs of
  the earlier plan were not run. The runbook and templates remain unused.
- **The label study.** The plan to have two people label diagnosis types and report Cohen's kappa (at least 0.65 in the
  earlier plan) was not run. So the study never tested whether the diagnosis agrees with human observations. `docs/README.md` says the
  label study was not run.
- **ByT5 text correction.** It was never used. It needs a download of about 1 GB that was not authorised, so
  `correct_text` used SymSpell (`symspell-en-82765`) throughout.
- **Practice round 3.** The lead froze the method after round 2. A higher `weird_char_threshold` was approved but not
  tested.
- **`append-anchor-words-v0` retry rule.** Step 4 of the plan was skipped after it resent the same chunks for 38 of 63
  questions on `ohr_dev_v1`.
- **gpt-4o scoring.** The 2026-10-31 gpt-4o baseline directory was never scored.
- **A query-rewriting retry, a learned correction model and a learned router.** Not built as methods. The router
  appears only in the offline analysis of 2026-11-15.
- **MP-DocVQA and other benchmarks.** Not brought forward as a longer-document source.
- **The unused train questions.** The 208 train-only questions left after `ohr_dev_v2` were not used by any run.
- **ArXivQA.** The 500-question ArXivQA staging was deferred. Its PDFs are not in this checkout.
