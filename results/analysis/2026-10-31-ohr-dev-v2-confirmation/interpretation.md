# ohr_dev_v2: what the numbers show

Written on 2026-10-31 after `compare.py` produced `report.md` and `results.json` (SHA-256 `10ab0608…`). The
comparison script and the decision rules were committed before the first run on `ohr_dev_v2` (commit `4bf0c48`,
amendment 1 of `docs/research/repair-analysis-plan.md`). Commit `83fd04a` changed only the report title and the
`pilot_id` label, after the v2 baseline ran and before any result was read. Measured results and interpretation are kept apart below.
All numbers are official EM over all 414 questions unless stated. No question failed in any run, so the
"served by both" view equals the primary view.

## Measured

**The v2 prompt did not raise EM on new questions.** faar-answer-v2 minus faar-answer-draft-v1, both without
recovery: −0.2 points, document interval [−3.3, +3.2], McNemar 19 gained and 20 lost, p = 1. The decision rule is not
met. F1, ANLS and NFKC EM also moved by about −1 point. On `ohr_dev_v1`, where v2 was written, the same contrast was
+10.0 points.

**The visual repair raised EM.** Visual repair on every question minus no recovery (v2): +8.5 points, from 0.287 to
0.372. The document interval is [+5.1, +12.3], and McNemar gives 42 gained, 7 lost, p = 3.6 × 10⁻⁷. The
pre-registered decision rule is met. F1 rose by 14.7 points, ANLS by 13.5 and NFKC EM by 8.5.

**The gated policy reached the same EM at half the cost.** Gated visual repair (abstain-or-ocr-gap-v1) minus no
recovery: +8.5 points [+5.6, +11.8], McNemar 36 gained and 1 lost. It flagged 131 questions: the 71 with no text
evidence and 60 answered ones that either abstained or sit on a document with an OCR gap. Cost per question was 0.00269 USD, against 0.00545 for
the repair on every question and 0.00132 for no recovery. This contrast has no decision rule.

**Where the gains are.** From `case_scores.csv`, a descriptive count that was not pre-registered:

| Questions | Gained | Lost |
| --- | ---: | ---: |
| Flagged, no text evidence, evidence page with empty MinerU text | 22 | 0 |
| Flagged, no text evidence, evidence page with MinerU text | 5 | 0 |
| Flagged, answered at first | 9 | 1 |
| Not flagged | 6 | 6 |

On the 56 questions whose evidence page has empty MinerU text, EM went from 0.000 to 0.393. On the other 358 it went
from 0.332 to 0.369.

**Execution.** The three repair-side runs had 6 TLS drops in 414 image requests on `ohr_dev_v2` (1.4%) and 2 in 70 on
`ohr_dev_v1`. Every drop was `connection_lost` and was resent once under `one-resend-connection-lost-v1`. Every resend
succeeded, and no question is blocked in any condition. The text-only baselines had no drop in 686 requests.

## Interpretation

- The +10 points of the v2 prompt on `ohr_dev_v1` came from tuning to those questions. On new questions the format
  rules fixed as many answers as they broke. The repair results do not depend on this: the repair and the no-recovery
  run both use v2, and the v1 run on `ohr_dev_v2` scores the same.
- The visual repair's gain is mostly a gain on questions that text could not answer at all. Twenty-seven of the 42
  gains are questions the no-recovery run never sent, because their pages produced no text. On questions whose evidence
  page has MinerU text, the images help a little (+3.7 points), and outside the gate's flagged
  set they helped as often as they hurt.
- So on this sample a cheap gate captures the whole effect. Showing the page image to every question doubles the
  cost and adds nothing net. This is the comparison the study's main question is about (diagnosis-selected repair
  against gate-triggered visual recovery), and it sets the bar a diagnosis-based policy has to clear: about +8.5
  points at about twice the no-recovery cost.
- The repair rate on the flagged set is 0.29 (36 of 124 wrong at first). Most flagged questions stay wrong after the
  repair, so there is room for a better repair on the same cases.
- On `ohr_dev_v1` the runtime page choice gave EM 0.386, the same as the gold-page image bound there. Page choice was
  not the limiting factor on these mostly single-page documents, and `ohr_dev_v2` (162 of 201 documents single-page)
  does not test it either. On multi-page documents the repair's EM fell from 0.312 to 0.281 (128 questions,
  descriptive).

## Still unverified

- Whether the 8 dropped attempts were billed. They stay reserved at their upper bound (0.154 USD in all).
- The image price of `gpt-6.1-terra`. Costs use provider-reported tokens at the recorded text rates.
- Whether the effect holds on longer documents, where page choice matters, and on the test split.
- `gpt-6.1-terra` is an undated alias, so a rerun may not reproduce these answers.
