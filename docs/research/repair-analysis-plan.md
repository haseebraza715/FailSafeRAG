# Analysis plan for the repair experiments

**Status: approved by the research lead on 2026-10-31 for the first experiment (section 10), with the changes in
amendment 1 (section 12), made before any run on `ohr_dev_v2`.** Section 11 lists what changed from the first draft.
The rest of the plan (the FAAR comparison, sections 1 to 9) still needs the decisions listed in section 9.
Once approved, the plan is frozen. A later change is a numbered amendment with its date and reason, made before the
run it affects. Results already produced stay in the record. Approval of this plan does not settle any open
"Lead decision" in the [study brief](study-brief.md); section 9 lists the ones it depends on.

The plan rests on two analyses of `ohr_dev_v1`: the failure analysis of the gpt-6.1-terra baseline
(`results/analysis/2026-10-31-ohr-dev-v1-terra-failures/report.md`), which coded the 55 non-exact cases under
codebook v2, and the answer-format comparison
(`results/analysis/2026-10-31-ohr-dev-v1-terra-answer-v2/report.md`), which reran the baseline and the four headroom
conditions with the prompt `faar-answer-v2`. Four findings shape the plan. All come from one agent coder and one
70-question development sample, so they are working assumptions, not results:

1. About half of the non-exact cases (27 of 55) are correct answers that official EM scores 0, because of units,
   wording, list format, LaTeX notation or a fullwidth character. Seven more rest on a questionable gold answer.
2. The prompt `faar-answer-v2` raised EM from 0.214 to 0.314 on the same evidence (McNemar 8 gained, 1 lost), by
   fixing 8 of those 27 cases. The other 19 differ from the gold in span boundaries that no prompt can know. They
   still dilute every comparison. v2 was written after reading these cases, so its gain is optimistic.
3. OCR is the primary cause in 13 of 55 cases. Ten are pages where MinerU lost the text (mostly handwriting). With
   v2, the gold page image added 7.1 EM points over all 70 questions and 9.0 on the 67 that both runs answered;
   clean text and the gold page's MinerU text added 1.4 each. Only the image interval reached above zero.
4. Retrieval is the primary cause in 2 cases (one page-level, one passage-level), because 23 of 30 pilot documents
   are single pages.

## 1. Questions

- **Q1 (primary).** On the same questions, does diagnosis-selected repair (FAAR) give a higher official EM than
  gate-triggered visual recovery?
- **Q2.** Does FAAR give a higher EM than random repair, and than no recovery?
- **Q3.** On the flagged set, how often does each policy turn a wrong answer right (repair rate) and a right answer
  wrong (damage rate)?
- **Q4.** Does the controller's diagnosis agree with human observations, signal by signal (brief section 3)?

Q1 to Q3 follow the brief's two links: Q4 tests the diagnosis link and Q1 to Q3 the repair link. Agreement between
the diagnosis and the policy that uses it proves neither.

## 2. Data and stages

| Stage | Data | Purpose | May inform method changes |
| --- | --- | --- | --- |
| Development | `ohr_dev_v1` (70 questions) and, if the lead approves one, a longer-document development sample | Build and debug the repairs, the gate and the diagnosis; run the all-repairs diagnostic (brief section 6) | Yes, with each change logged as a new registered run |
| Confirmation | `ohr_dev_v2` ([proposal and freeze record](ohr-dev-v2-proposal.md)): 414 train-only questions on 201 documents, document-disjoint from `ohr_dev_v1`, frozen before any run | Test each pre-registered contrast once, starting with section 10 | No |
| Test | OHR-Bench test split, only after the lead freezes the method | One final run | No |

The confirmation sample is selected by a rule written down before selection and frozen with its own `pilot_id`.
No question in it is inspected before the confirmation run. The train split limits its size: only 343 documents
(692 questions) have no question in validation or test, and `ohr_dev_v1` uses 70 of those questions. After
`ohr_dev_v2`, 208 train-only questions remain, and no run uses them.

## 3. Fixed before any repair run

1. **The answer prompt and its output format.** The development step is done: `faar-answer-v2` was written,
   recorded as the no-recovery run `2026-10-31-ohr-dev-v1-baseline-answer-v2-terra-r1`, and compared with v1
   (finding 2). It becomes the shared prompt only when the lead approves it. All policies then share it, and it is
   never changed after a repair result is seen.
2. The answer model and its settings, the retrieval settings, the gate and its threshold (chosen on development data
   only), the diagnosis rules, the repair implementations, the image budget and page-selection policy, the random
   seeds, and the cost-matching rule. Each is a brief "Lead decision" or "Proposed default"; this plan uses whatever
   the lead fixes.
3. This analysis plan.

All of these go into one registered method record that every confirmation and test run cites (brief section 9).

## 4. Outcomes

- **Primary:** official OHR-Bench EM over all eligible questions. `execution_failed` and `no_evidence` count as 0
  (brief section 8).
- **Secondary, declared now:** official F1; ANLS (threshold 0.5, FAAR-defined); NFKC EM (FAAR-defined: official EM
  after Unicode NFKC normalisation of both strings, so a fullwidth `～` matches `~`); repair rate and damage rate on the
  flagged set with counts (brief section 8 denominators); cost per policy and per question (brief section 5).
- **Not an outcome:** an agent's or the experimenter's correctness judgement. If the lead wants a human-judged
  accuracy, two people label it blind to the policy, with agreement reported, and it stays secondary.

## 5. Analysis

- **Unit and pairing.** Every policy answers the same questions from the same initial retrieval and initial answer.
  All comparisons are paired by question.
- **Primary contrast.** EM(FAAR) minus EM(gate-triggered visual). Report the difference with a 95% paired bootstrap
  interval that resamples documents (10,000 resamples, seed 20261031), and McNemar's exact two-sided test on EM.
  Report the question-level bootstrap next to it for reference.
- **Secondary contrasts.** FAAR minus random (pooled over the declared seeds, and per seed), FAAR minus no recovery.
  The three contrasts (primary and two secondary) are adjusted with Holm's method at family-wise alpha 0.05.
- **Flagged-set view.** Repairs and damage per policy on the shared flagged set, with counts and exact binomial
  intervals.
- **Diagnosis agreement (Q4).** Two people label the flagged cases with the codebook's text-defect and
  evidence-impact fields, blind to the controller and to each other. Report Cohen's kappa between the two people per
  field and per label, then each controller signal against the adjudicated human label. Mixed and uncertain cases are
  separate rows.
- **Descriptive subgroups (no tests).** Han-script questions, empty-OCR evidence pages, multi-page evidence, multi-page
  documents, evidence type, domain.
- **Exclusions.** None. Every eligible question stays in every denominator.

## 6. Sample size

Detecting a 5-point paired EM gain with McNemar's test (alpha 0.05, power 0.80) needs about 190 questions if the
two policies disagree on 6% of questions, about 310 at 10% and about 470 at 15%. Questions sharing a document inflate
this by a design effect that is poorly known: the intraclass correlation of EM within documents was 0.40 on the v1
baseline and 0.02 on the v2 baseline, from the same 30 documents. `ohr_dev_v2` averages 2.06 questions per document,
which gives design effects between 1.0 and 1.42. With the v2 prompt, the gold page image disagreed with no recovery
on 12.9% of questions. At that rate a 5-point gain needs 403 questions without clustering and 584 with a design
effect of 1.42, and a 6-point gain needs 279 and 405. The 414 questions of `ohr_dev_v2` therefore detect a gain of
5 to 6 points. Each confirmation result is reported with its interval, and a null result is reported as underpowered
for gains below that size, not as evidence of no effect.

`ohr_dev_v1` (70 questions) cannot detect a 5-point difference at any plausible rate. Results on it are
development evidence only.

## 7. Decision rules

- FAAR is reported as better than gate-triggered visual recovery only if the Holm-adjusted McNemar p is below 0.05
  and the document-level interval excludes 0.
- A null or negative result is reported as such, with the interval. The experiment is not rerun with changed settings
  on the confirmation sample to look for a different result.
- If a policy runs out of budget, the comparison is reported as budget-limited with the counts (brief section 5).
- If more than the lead's maximum execution-failure rate fails, the run is `failed` and is reported, not replaced
  silently.

## 8. Reporting

One table per view (all questions; flagged set), each with EM, F1, ANLS, the paired difference and interval, the
McNemar counts and p-value, repair and damage counts, and cost components. Every number cites a registered run.
Development results and confirmation results are never pooled.

## 9. Decisions this plan needs from the lead

1. Approve or change this plan, including the primary contrast and the Holm family.
2. Approve the answer-format work in section 3 item 1 as a development step before any repair run.
3. Decide the confirmation sample: its source, its selection rule and the budget that sets its size.
4. The brief's open decisions that every repair run depends on: image budget and page selection, gate threshold,
   maximum execution-failure rate, spending cap, and the longer-document development sample.
5. Who the two human labellers are for Q4, and whether the agent coding of the 55 baseline failures is used only as
   development material.
6. Approve `faar-answer-v2` as the shared prompt (section 3 item 1).
7. Approve the first experiment in section 10, its sample and its spending cap.
8. Decide whether to adopt a normalised EM as a secondary score next to official EM (section 4). This plan proposes
   NFKC EM as FAAR-defined and secondary only. The primary score stays official EM unless the lead decides
   otherwise.

## 10. First experiment: visual repair on every question

The design is in [visual-repair-design.md](visual-repair-design.md). It runs before the full policy comparison,
because only the visual repair showed headroom (finding 3), and because it tests the repair link alone, without a
diagnosis.

- **Sample.** `ohr_dev_v2`, all 414 questions, as the confirmation sample.
- **Policies.** No recovery (the shared initial answer, `faar-answer-v2`). Visual repair on every question, with
  pages chosen by `retrieved-then-gaps-v1` and a budget of 3 images. Gated visual repair, computed from the same
  answers with the gate `abstain-or-ocr-gap-v1`.
- **Primary contrast.** Official EM, visual repair on every question minus no recovery. One primary contrast, so no
  multiplicity correction. Document-level paired bootstrap 95% interval (10,000 resamples, seed 20261031) and
  McNemar's exact two-sided test at alpha 0.05.
- **Decision rule.** "The visual repair raised EM" only if McNemar's p is below 0.05 and the document-level interval
  excludes 0. Otherwise the result is reported with its interval and the detectable gain from section 6.
- **Secondary, declared now.** F1, ANLS and NFKC EM for the same contrast. The gated policy minus no recovery, with
  the same interval and test, reported without a decision rule. Repair rate and damage rate with counts (brief
  section 8). Cost per question, images per request and execution failures per policy.
- **Execution failures.** `execution_failed` scores 0 in the primary view. A second view on the questions both runs
  answered is reported next to it.
- **Subgroups, descriptive only.** Empty-OCR evidence pages, Han-script questions, multi-page documents, document
  type.

## 11. Changes from the first draft

Made on 2026-10-31, before any repair run, after the answer-format runs and the codebook v2 recoding:

- The findings now cite codebook v2 (27 `scoring_artifact` cases instead of 28) and the v2 headroom runs.
- Section 2 names `ohr_dev_v2` as the proposed confirmation sample and records the size limit of the train split.
- Section 3 records that the answer-format development step is done and awaits approval.
- Section 6 uses the discordant rate observed with v2 and states the detectable gain at 285 questions.
- Section 4 adds NFKC EM as a secondary score.
- Section 9 adds decisions 6 to 8. Section 10 is new.

## 12. Amendment 1 (2026-10-31, before any run on `ohr_dev_v2`)

The lead approved `ohr_dev_v2` and the first experiment with these changes:

1. **Sample size.** `ohr_dev_v2` was frozen at 201 documents and 414 questions instead of 138 and 285, because a
   runtime page choice will probably gain 5 to 6 points and 285 questions detect that only about half the time.
   Sections 2, 6 and 10 now give the frozen numbers.
2. **The v2 prompt is tested on `ohr_dev_v2`.** v2 was written after reading the `ohr_dev_v1` failures, so its gain
   there is optimistic. Before the repair, two no-recovery runs on `ohr_dev_v2` share retrieval and differ only in
   the template: `faar-answer-draft-v1` and `faar-answer-v2`. Contrast: official EM, v2 minus v1, paired by
   question, with the document-level bootstrap interval (10,000 resamples, seed 20261031) and McNemar's exact
   two-sided test. Decision rule: "v2 raised EM on new questions" only if McNemar's p is below 0.05 and the interval
   excludes 0. Secondary: F1, ANLS, NFKC EM. This contrast is separate from the repair contrast and is not
   adjusted with it. The v2 run is the shared initial answer of the repair (section 10), whatever this test shows.
3. **Resend rule `one-resend-connection-lost-v1`.** An engineering check
   (`results/engineering/2026-10-31-tls-bad-record-mac-check/report.md`) found that the TLS "bad record MAC" drops
   come from corruption of large uploads on the local network path, not from the runner, the TLS library,
   connection reuse or HTTP/2, and cannot be fixed in code. Large bodies dropped 2.3% of the time and small ones
   never. For every run on `ohr_dev_v2`:
   - An attempt that ends with kind `connection_lost` and an unknown outcome may be sent once more. A question gets
     at most one such resend. A second unknown outcome blocks it.
   - No other unknown outcome (timeout, 5xx, redirect, a crash after dispatch) is resent.
   - The first attempt stays counted at its upper bound in the cost ledger and under the cap, whether or not the
     resend succeeds.
   - The repair runner applies the rule itself (`scripts/experiments/run_visual_repair.py`, recorded in the run
     identity). For the no-recovery runs, which use `run_pilot_live.py`, the same rule is applied by hand with
     `reconcile --resolution allow_new_attempt`, citing this amendment in the note.
   - Each run reports, per condition, the questions with an unknown outcome, the resends, and the questions still
     blocked. Blocked questions are `execution_failed` and score 0 in the primary view. The view on questions both
     runs answered is reported next to it.
4. **Order of runs.** v2 no-recovery and v1 no-recovery on `ohr_dev_v2`, a debug run of the repair on `ohr_dev_v1`
   (development data, against the existing v2 no-recovery run there), then the visual repair on `ohr_dev_v2`. The
   comparison script for `ohr_dev_v2` is committed before the first of these runs. Nobody reads the `ohr_dev_v2`
   answers before the script has computed the comparisons. The spending cap for the stage is 6.00 USD, counting reserved cost.

