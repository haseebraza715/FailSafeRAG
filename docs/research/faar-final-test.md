# FAAR final test on the OHR-Bench test split

The single final test of [faar-vs-gated-preregistration.md](faar-vs-gated-preregistration.md) (section 15), with
the method frozen at `b075dbd`. Each sample was built once as its own frozen pilot, and
`scripts/analysis/faar_test/compare_test.py` (committed at `1e4fd50`, before the first test run) ran once per sample,
before anyone scored a run or read an answer. Every number comes from the analysis folders named below.

## Runs

Both samples were built at 11:57 UTC on 2026-11-01 (`2026-11-01-ohr-test-{multipage,singlepage}-v1-selection`). No
request ended with an unknown outcome, so no resend was needed and no question was blocked.

| Run (prefix `2026-11-01-ohr-test-`) | What | Measured USD |
| --- | --- | ---: |
| `multipage-v1-baseline-answer-v2-terra` | no recovery, 946 answered | 1.615026 |
| `multipage-v1-diagnosis` | offline: 706 of 946 flagged; `semantic` 541, `word_level` 136, `missing_text` 19, `structural` 10 | 0 |
| `multipage-v1-retry-keep-top3-terra` | `retry_retrieval`, 694 answered, 12 skipped | 1.208255 |
| `multipage-v1-correct-symspell-terra` | `correct_text`, 704 answered, 2 skipped | 1.183747 |
| `multipage-v1-visual-terra` | `invoke_vlm`, 706 answered | 6.248469 |
| `singlepage-v1-baseline-answer-v2-terra` | no recovery, 307 answered, 23 with no text in the noisy input (`no_evidence`) | 0.510896 |
| `singlepage-v1-diagnosis` | offline: 198 of 330 flagged; `word_level` 112, `semantic` 60, `missing_text` 23, `structural` 3 | 0 |
| `singlepage-v1-retry-keep-top3-terra` | `retry_retrieval`, 94 answered, 104 skipped (nothing new to retrieve) | 0.182478 |
| `singlepage-v1-correct-symspell-terra` | `correct_text`, 102 answered, 96 skipped (correction changed nothing) | 0.155993 |
| `singlepage-v1-visual-terra` | `invoke_vlm`, 198 answered | 0.832301 |

Stage total: 11.937165 USD of the 15 USD cap, with no reserved cost. The primary sample cost 10.255497 USD. The
secondary sample started because its expected cost (about 2.9 USD) fit under the 4.744503 USD left. Its three repair
caps summed to 4.20 USD, so the stage could not exceed its cap even if every request had cost its upper bound.

## Primary result: 946 multi-page test questions

From `results/analysis/2026-11-01-ohr-test-multipage-v1-final-test/report.md`:

| Policy | Official EM | Total USD per question |
| --- | ---: | ---: |
| No recovery | 0.236 | 0.00171 |
| Gated visual repair (the bar) | 0.315 | 0.00831 |
| FAAR | 0.266 | 0.00320 |
| Random repair, 3 seeds | 0.274, 0.278, 0.274 | 0.0045 to 0.0049 |

- **FAAR minus the bar:** -0.049 EM, document interval [-0.073, -0.026], Holm p 6.6e-6. The pre-registered rule
  (FAAR better than the bar) is not met. The bar is significantly better.
- **FAAR minus no recovery:** +0.031 EM, interval [0.016, 0.047], Holm p 6.6e-6.
- **FAAR minus random repair (pooled):** -0.009 EM, interval [-0.023, 0.006], Holm p 0.43.
- **Cost:** FAAR costs 0.39 times the bar per question.
- **Headroom (descriptive, not a claim):** the best repair per question in hindsight, chosen with gold answers,
  reaches 0.346 EM at 0.00244 USD per question.

## Secondary result: 330 single-page test questions

Reported separately: it is outside the primary Holm family and cannot change the primary decision. From
`results/analysis/2026-11-01-ohr-test-singlepage-v1-final-test/report.md`:

| Policy | Official EM | Total USD per question |
| --- | ---: | ---: |
| No recovery | 0.294 | 0.00155 |
| Gated visual repair | 0.345 | 0.00407 |
| FAAR | 0.327 | 0.00240 |

FAAR minus the gated repair: -0.018 EM, document interval [-0.043, 0.006], McNemar p 0.21 (Holm p 0.42 within this
sample's family), so the difference is not significant. FAAR minus no recovery: +0.033 EM, interval [0.014, 0.055],
Holm p 0.010. Cost ratio 0.59.

## Across the study

The direction is the same in development, the confirmation and the final test. FAAR beats no recovery and costs
about 0.35 to 0.4 times the gated visual repair on multi-page documents, but it answers fewer questions correctly
than the gated repair. The image repair scored highest on every route in both test samples (route tables in the
reports), which is why routing flagged questions to text repairs cannot beat it here.

The monitoring session rescored the primary sample independently from the raw replies: same EM for all three
policies, no question different, FAAR against the bar 24 gained and 70 lost.

Total study spend: practice 3.624891 USD measured (4.038218 counting reserved cost), confirmation 9.243314 measured
(10.038311), final test 11.937165. That is 24.805370 USD measured and 26.013694 USD counting reserved cost, of the
40 USD budget.
