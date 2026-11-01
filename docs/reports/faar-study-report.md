# FAAR study report: diagnosis-based repair against a simple image repair

Haseeb Raza, 2026-11-01. This report covers the whole FAAR study, from the first real-model pilot to the final test
on held-out OHR-Bench questions. Every number in it comes from a registered run in
[experiments/registry.jsonl](../../experiments/registry.jsonl) and an analysis folder under `results/analysis/`. The
detailed stage reports are linked in each section.

## Summary

FAAR does not beat the simple baseline. On 946 held-out multi-page test questions, FAAR answered 26.6% of questions
exactly right. Sending page images for every flagged question answered 31.5%. The difference is -4.9 points, with a
95% document-bootstrap interval of [-7.3, -2.6] and a Holm-adjusted McNemar p of 6.6e-6. The pre-registered rule for
"FAAR beats the image repair" is not met, and the image repair is significantly better.

FAAR does help compared with no recovery (+3.1 points, interval [+1.6, +4.7]), and it costs 0.39 times as much per
question as the image repair. Its diagnosis adds nothing measurable. FAAR scores the same as always retrying
retrieval (26.5%) and slightly below a random choice of repair (27.5% averaged over three seeds).

The same pattern held in development, in the confirmation run on validation data and in the final test. On a
secondary sample of 330 single-page test questions, FAAR was 1.8 points below the image repair, which is not
significant (Holm p 0.42).

## The question

OCR-backed question answering fails in different ways. The OCR text can be empty, garbled at the word level, broken
in its layout, or present but not retrieved. FAAR (Failure-Aware Agentic Retrieval) diagnoses which of these happened
and picks a matching repair. The question was:

> On the same questions, with the same gate, initial answer and answer model, does the repair chosen by diagnosis
> give a higher exact-match score than showing the page images on every flagged question, at a similar cost?

## Data and setup

The benchmark is OHR-Bench, using its MinerU OCR output as the noisy input text. Clean ground-truth text, gold
answers and evidence-page labels were never used to make a runtime decision. They were used only for scoring and for
diagnostic reference conditions: the `ohr_dev_v1` headroom runs, which answered from clean text or gold evidence
pages, and the per-question best repair in hindsight, reported as headroom.

The answer model is `gpt-6.1-terra` through the OpenAI API, with temperature 0, `reasoning_effort` none and at most
128 output tokens. Every policy uses the same retrieval (top 5 chunks from the document), the same answer prompt
(`faar-answer-v2`) and the same initial answer.

Questions passed to a repair are chosen by a gate. The gate flags a question when the initial answer abstained or found no evidence, when
the document has a page with empty OCR text, or when fewer than 69.2% of the question's words appear in the
retrieved text. The last threshold was chosen on development data by Youden's J with a 30% flag cap.

The primary score is the official OHR-Bench exact match (EM) over all questions. A question with no usable evidence
or a failed request scores 0.

## The policies compared

Every policy shares the gate. A question the gate does not flag keeps its initial answer under every policy.

| Policy | What a flagged question gets |
| --- | --- |
| No recovery | The initial answer. |
| Gated image repair (the bar) | The initial text plus up to 3 page images (150 DPI, JPEG), answered again. |
| Random repair | One of the three repairs, drawn by a hash of the question id, for each of 3 fixed seeds. |
| FAAR | The repair its diagnosis selects, from the table below. |

FAAR's diagnosis reads only the question, the retrieved chunks with their scores and the per-page OCR status. It
checks the rules in this order:

| Diagnosis | Fires when | Repair |
| --- | --- | --- |
| `missing_text` | No text evidence was retrieved, or one of the up to 3 pages the image repair would show has empty or missing OCR text | Image repair |
| `structural` | The top chunk shows at least one layout-damage signal | Image repair |
| `word_level` | More than 10% of the top chunk's characters or tokens look like OCR noise | Text correction (SymSpell), then answer again |
| `semantic` | None of the above | Retry retrieval (keep the top 3 chunks, add the 2 best unseen ones), then answer again |

## How the study kept the result honest

The study ran in stages, and each stage used data that no earlier decision had seen.

1. Development pilots on train data (`ohr_dev_v1`, `ohr_dev_v2`) set the bar and the answer prompt.
2. Two practice rounds on 423 train questions (`ohr_long_dev_v1`) built and tuned the repairs. After round 2 the
   method was frozen at code commit `b075dbd`.
3. A confirmation ran once on 835 validation questions (`ohr_val_multipage_v1`).
4. The final test ran once on 946 multi-page test questions (`ohr_test_multipage_v1`), with 330 single-page test
   questions (`ohr_test_singlepage_v1`) as a separate secondary result.

The method, the samples, the decision rule and the comparison script were written down and committed before each
confirmation and test run, in [faar-vs-gated-preregistration.md](../research/faar-vs-gated-preregistration.md). No
score or answer was read before the comparison script ran. Nothing was rerun with changed settings after an outcome
was seen. Failed and negative runs stay in the registry.

The decision rule was set in advance. FAAR beats the bar only if the Holm-adjusted McNemar p is below 0.05 and the
95% document-bootstrap interval (10,000 resamples, seed 20261031) lies above 0.

A second reviewer, who did not write the analysis code, rescored the confirmation and both test samples from the raw
model replies with an independent scorer, and recomputed the practice-round numbers. Every policy score matched, with
no question scored differently. The stage reports record this for the practice rounds and the primary test sample.

## Development: setting the bar

On `ohr_dev_v2` (414 train questions on 201 documents), showing page images raised EM from 0.287 to 0.372, a gain of
8.5 points (interval [+5.1, +12.3], McNemar 42 gained and 7 lost). Most of the gain came from pages whose OCR text was
empty. Applying the image repair only to questions the base gate `abstain-or-ocr-gap-v1` flagged, without the coverage
condition, gave the same +8.5 points at half the cost (0.0027
against 0.0055 USD per question). That gated image repair became the bar FAAR had to beat. The answer prompt change
tested on the same sample made no difference (-0.2 points), so `faar-answer-v2` was kept.

The full report is
[results/analysis/2026-10-31-ohr-dev-v2-confirmation/report.md](../../results/analysis/2026-10-31-ohr-dev-v2-confirmation/report.md).

## Practice rounds

The practice rounds ran on `ohr_long_dev_v1`, 423 train questions on 101 multi-page documents that share no
document with the test questions. Details are in [faar-practice-rounds.md](../research/faar-practice-rounds.md).

| Policy | Round 1 EM | Round 2 EM |
| --- | ---: | ---: |
| No recovery | 0.310 | 0.310 |
| Gated image repair (the bar) | 0.397 | 0.397 |
| FAAR | 0.307 | 0.343 |
| Best repair per question in hindsight | 0.416 | 0.409 |

In round 1, the retry repair threw away all 5 initial chunks and made answers worse than no recovery. Round 2
changed only the retry rule, to keep the top 3 chunks and add the 2 best unseen ones. That lifted FAAR from 0.307
to 0.343, above no recovery, but still 5.4 points below the bar. The image repair scored highest or tied for highest on
every diagnosis route in both rounds. The method was frozen after round 2 without a third round.

## Confirmation on validation data

The confirmation ran once on 835 validation questions from 259 multi-page documents. The gate flagged 647 of them.
Details are in [faar-confirmation.md](../research/faar-confirmation.md).

| Policy | EM | Cost per question (USD) |
| --- | ---: | ---: |
| No recovery | 0.250 | 0.00169 |
| Gated image repair (the bar) | 0.327 | 0.00942 |
| FAAR | 0.283 | 0.00333 |
| Random repair, 3 seeds | 0.281 to 0.287 | 0.0050 to 0.0052 |
| Retry retrieval on every flagged question | 0.287 | 0.00301 |

FAAR minus the bar was -4.4 points (interval [-6.8, -2.1], Holm p 0.0003, 27 gained and 64 lost). FAAR minus no
recovery was +3.2 points (interval [+1.7, +4.8]). FAAR minus random repair was -0.2 points (Holm p 1.0). The rule
was not met.

## Final test

### Primary sample: 946 multi-page test questions

The primary sample has 946 test questions on 327 multi-page documents, none of them used in development. The gate
flagged 706. The diagnosis sent 541 to retry retrieval, 136 to text correction and 29 to the image repair. Every
request completed. No question was blocked. The full report is
[results/analysis/2026-11-01-ohr-test-multipage-v1-final-test/report.md](../../results/analysis/2026-11-01-ohr-test-multipage-v1-final-test/report.md).

| Policy | EM | F1 | Cost per question (USD) |
| --- | ---: | ---: | ---: |
| No recovery | 0.236 | 0.337 | 0.00171 |
| Gated image repair (the bar) | 0.315 | 0.436 | 0.00831 |
| FAAR | 0.266 | 0.381 | 0.00320 |
| Random repair, 3 seeds | 0.274 to 0.278 | | 0.0045 to 0.0049 |
| Retry retrieval on every flagged question | 0.265 | 0.379 | 0.00298 |
| Best repair per question in hindsight (headroom only) | 0.346 | 0.445 | 0.00244 |

| Contrast | EM difference | Document interval | Gained / lost | Holm p |
| --- | ---: | --- | --- | ---: |
| FAAR minus the bar | -4.9 | [-7.3, -2.6] | 24 / 70 | 6.6e-6 |
| FAAR minus no recovery | +3.1 | [+1.6, +4.7] | 34 / 5 | 6.6e-6 |
| FAAR minus random repair (pooled) | -0.9 | [-2.3, +0.6] | | 0.43 |

F1 and ANLS give the same direction. FAAR minus the bar is -5.5 F1 points and -5.2 ANLS points. FAAR costs 0.39
times the bar per question, which meets the cost rule. The cost advantage does not offset the accuracy loss, because
the rule asks for higher EM at similar or lower cost.

### Secondary sample: 330 single-page test questions

This sample was fixed in advance as a separate result outside the primary test family. It cannot change the primary
decision. The full report is
[results/analysis/2026-11-01-ohr-test-singlepage-v1-final-test/report.md](../../results/analysis/2026-11-01-ohr-test-singlepage-v1-final-test/report.md).

| Policy | EM | Cost per question (USD) |
| --- | ---: | ---: |
| No recovery | 0.294 | 0.00155 |
| Gated image repair | 0.345 | 0.00407 |
| FAAR | 0.327 | 0.00240 |

FAAR minus the image repair was -1.8 points (interval [-4.3, +0.6], 5 gained and 11 lost, McNemar p 0.21, Holm p
0.42 within this sample). The
difference is not significant. FAAR minus no recovery was +3.3 points (interval [+1.4, +5.5], Holm p 0.010).

## Why FAAR lost

FAAR can beat the image repair only if, on some diagnosis route, a text repair answers more questions correctly than
the image does. That never happened. The table shows the EM of each repair on the questions each route received in
the primary test sample.

| Route chosen by FAAR (questions) | Initial answer | Retry | Text correction | Image |
| --- | ---: | ---: | ---: | ---: |
| Retry retrieval (541) | 0.131 | 0.177 | 0.124 | 0.231 |
| Text correction (136) | 0.228 | 0.250 | 0.213 | 0.338 |
| Image repair (29) | 0.241 | 0.241 | 0.207 | 0.448 |

The table shows why:

- The image repair is best on every route, including the routes where the diagnosis said the text was the problem.
- SymSpell text correction never beat the initial answer on its own route. That suggests word-level OCR noise was
  not what kept the model from answering on these documents.
- Retry retrieval helps on its route (0.131 to 0.177), but the image helps more (0.231). Most flagged questions (541
  of 706) went to retry.

The per-question scores show how much a better router could gain. Of the 706 flagged test questions, 27 were
answered correctly by a text repair but not by the image, and 71 by the image but not by either text repair. The
best repair per question in hindsight reaches 0.346 EM at 0.00244 USD per question, against 0.315 and 0.00831 for
the bar. A router that chose well could therefore score higher at lower cost. FAAR's rule-based diagnosis does not
find those cases. Its choices do no better than random.

## Limitations

- On long documents the gate flags most questions: 706 of 946 in the primary test (75%). A document with one empty
  OCR page triggers the gate for every question on it. 412 of the 706 flagged test questions carry that reason. The
  base gate was designed on the shorter `ohr_dev_v1` and `ohr_dev_v2` documents. With the coverage condition it
  flagged 54% of the practice-round questions, against 75% on the test.
- The initial answer abstained on 467 of the 946 test questions. Much of the comparison happens on questions the
  baseline did not try to answer.
- One answer model, one OCR system (MinerU) and one benchmark were tested. The result may differ with a stronger OCR
  system, a different model or other document types.
- The text repairs are simple: SymSpell spelling correction and a single retrieval retry. A learned correction model
  or a query-rewriting retry was not tested.
- 252 of the 327 primary test documents also carry validation questions used in the confirmation. The test questions
  themselves are distinct, and no method decision used validation outcomes.
- The headroom figure uses gold answers to pick the best repair per question. It shows what a perfect router could
  reach. It is not a result any method achieved.

## What the result means

Interpretation: with these repairs, diagnosing the failure type does not help. Sending page images to the answer
model is the strongest repair on every failure type the diagnosis recognises, so routing some questions to cheaper
text repairs costs accuracy. FAAR's real advantage is cost, at about 0.4 times the image repair. A study that values
cost could report FAAR as a cheaper option that keeps about 40% of the image repair's gain over no recovery (+3.1 of
+7.9 points).

The headroom suggests a different research question for future work. A learned router could predict, from runtime
signals, whether the image repair is needed for a question. That is a new question and would need its own
development and test data.

## Cost

| Stage | Measured USD | Counting reserved cost | Cap |
| --- | ---: | ---: | ---: |
| Practice rounds | 3.62 | 4.04 | 10 |
| Confirmation | 9.24 | 10.04 | 12 |
| Final test | 11.94 | 11.94 | 15 |
| Total | 24.81 | 26.01 | 40 |

Reserved cost is the worst-case charge for requests whose outcome was unknown after a network drop. Earlier pilots
on `ohr_dev_v1` and `ohr_dev_v2` are recorded separately in the registry.

## Where to find everything

| What | Where |
| --- | --- |
| Pre-registration, frozen method and test plan | [docs/research/faar-vs-gated-preregistration.md](../research/faar-vs-gated-preregistration.md) |
| Practice rounds | [docs/research/faar-practice-rounds.md](../research/faar-practice-rounds.md) |
| Confirmation | [docs/research/faar-confirmation.md](../research/faar-confirmation.md) |
| Final test | [docs/research/faar-final-test.md](../research/faar-final-test.md) |
| Analysis reports, per-question scores and statistics | `results/analysis/2026-11-01-*/` |
| Raw model replies and run records | `results/development/2026-11-01-*/` |
| Every run with its settings, code commit and output hashes | [experiments/registry.jsonl](../../experiments/registry.jsonl) |
| Comparison scripts | `scripts/analysis/faar_confirmation/compare_confirmation.py`, `scripts/analysis/faar_test/compare_test.py` |
