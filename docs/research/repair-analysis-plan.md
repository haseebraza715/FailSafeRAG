# Analysis plan for the repair experiments

**Status: draft for the research lead's approval, written 2026-10-31. No repair run exists, and this plan authorises none.**
Once approved, the plan is frozen. A later change is a numbered amendment with its date and reason, made before the
run it affects. Results already produced stay in the record. Approval of this plan does not settle any open
"Lead decision" in the [study brief](study-brief.md); section 9 lists the ones it depends on.

The plan rests on the failure analysis of the gpt-6.1-terra baseline
(`results/analysis/2026-10-31-ohr-dev-v1-terra-failures/report.md`), which coded the 55 non-exact cases of
`ohr_dev_v1` and bounds each repair with diagnostic headroom runs. Three findings shape it. All three come from one
agent coder and one 70-question development sample, so they are working assumptions, not results:

1. About half of the non-exact cases (28 of 55) are correct answers that official EM scores 0, because of units,
   wording, list format, LaTeX notation or a fullwidth character. Seven more rest on a questionable gold answer.
   A repair cannot change these cases, and they dilute every comparison.
2. OCR is the primary cause in 13 of 55 cases. Ten are pages where MinerU lost the text (mostly handwriting and
   handwritten notes). Only a visual repair can reach them. Text correction cannot restore text that is absent.
3. Retrieval is the primary cause in 2 cases, because 23 of 30 pilot documents are single pages.

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
| Confirmation | A new train-split sample, document-disjoint from development, sized by section 6, frozen before any run | Test Q1 to Q3 once | No |
| Test | OHR-Bench test split, only after the lead freezes the method | One final run | No |

The confirmation sample is selected by a rule written down before selection (for example, a seeded draw stratified
by domain and evidence type) and frozen with its own `pilot_id`. No question in it is inspected before the
confirmation run.

## 3. Fixed before any repair run

1. **The answer prompt and its output format.** Finding 1 means the answer format, not the evidence, decides about
   half the non-exact cases. A prompt change (for example: "give the number without its unit", "give the formula only")
   is a method change. It is developed on development data, recorded as a new no-recovery baseline run, and frozen.
   All policies then share that prompt. It is never changed after a repair result is seen.
2. The answer model and its settings, the retrieval settings, the gate and its threshold (chosen on development data
   only), the diagnosis rules, the repair implementations, the image budget and page-selection policy, the random
   seeds, and the cost-matching rule. Each is a brief "Lead decision" or "Proposed default"; this plan uses whatever
   the lead fixes.
3. This analysis plan.

All of these go into one registered method record that every confirmation and test run cites (brief section 9).

## 4. Outcomes

- **Primary:** official OHR-Bench EM over all eligible questions. `execution_failed` and `no_evidence` count as 0
  (brief section 8).
- **Secondary, declared now:** official F1; ANLS (threshold 0.5, FAAR-defined); repair rate and damage rate on the
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

From the failure-analysis report (section 5): detecting a 5-point paired EM gain with McNemar's test (alpha 0.05,
power 0.80) needs about 190 questions if the two policies disagree on 6% of questions, about 310 at 10% and about
470 at 15%. Questions sharing a document inflate this by an estimated design effect of 1.5 (ICC 0.40 on the
baseline, about 2.3 questions per document): about 290, 480 and 720 questions. The confirmation sample size is set
from the discordant rate that the development all-repairs diagnostic observes between FAAR and gate-triggered visual
recovery, rounded up, before the sample is drawn. If the required size exceeds what the lead can fund, the plan
says so and the confirmation run is reported as underpowered, with its interval, rather than skipped.

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
