# FAAR confirmation on `ohr_val_multipage_v1`

The single confirmation run of [faar-vs-gated-preregistration.md](faar-vs-gated-preregistration.md) (section 14), with
the method frozen at `b075dbd`. Every number comes from
`results/analysis/2026-11-01-ohr-val-multipage-v1-confirmation/` (`report.md`, `results.json`, `case_scores.csv`),
which `scripts/analysis/faar_confirmation/compare_confirmation.py` wrote in one run before anyone read a validation
answer or score.

## Runs

| Run | What | Measured USD | Reserved USD |
| --- | --- | ---: | ---: |
| `2026-11-01-ohr-val-multipage-v1-baseline-answer-v2-terra` | no recovery, 835 questions, 0 unknown outcomes | 1.407072 | 0 |
| `2026-11-01-ohr-val-multipage-v1-diagnosis` | offline: 647 of 835 flagged; `semantic` 492, `word_level` 128, `missing_text` 17, `structural` 10 | 0 | 0 |
| `2026-11-01-ohr-val-multipage-v1-retry-keep-top3-terra` | `retry_retrieval`, 641 answered, 6 skipped | 1.103037 | 0 |
| `2026-11-01-ohr-val-multipage-v1-correct-symspell-terra` | `correct_text`, 645 answered, 2 skipped | 1.071389 | 0 |
| `2026-11-01-ohr-val-multipage-v1-visual-terra` | `invoke_vlm`, 644 answered, 25 unknown outcomes on 22 questions, 22 resends, 3 blocked | 5.661816 | 0.794997 |

The image run took 5 invocations. Invocations 1 to 3 each stopped after 3 consecutive TLS "bad record MAC" drops
and were resumed with the same settings; the last one sent the one remaining resend that amendment 1 allows. A
free, unauthenticated probe before resuming dropped 4 of 80 large bodies (2.3% in the October check). The 3 blocked
questions (0.46%) are under both amendment 2 limits, and the comparison reports itself as valid.

Stage total: 9.243314 USD measured, 10.038311 USD counting reserved cost, of the 12 USD cap.

## Result

| Policy | Official EM | Total USD per question |
| --- | ---: | ---: |
| No recovery | 0.250 | 0.00169 |
| Gated visual repair (the bar) | 0.327 | 0.00942 |
| FAAR | 0.283 | 0.00333 |
| Random repair, 3 seeds | 0.281, 0.285, 0.287 | 0.0050 to 0.0052 |

- **Primary, FAAR minus the bar:** -0.044 EM, document interval [-0.068, -0.021], Holm p 0.0003. The pre-registered
  rule (FAAR better than the bar) is not met. The bar is significantly better.
- **FAAR minus no recovery:** +0.032 EM, interval [0.017, 0.048], Holm p 0.0003.
- **FAAR minus random repair (pooled):** -0.002 EM, interval [-0.015, 0.012], Holm p 1.0.
- **Cost:** FAAR costs 0.35 times the bar per question.
- **Headroom (descriptive, not a claim):** picking the best repair per question in hindsight using gold answers
  gives 0.369 EM at 0.00249 USD per question.

The image repair scored highest on every route here too (route table in `report.md`), as it did in development.
That repeats the development finding: with these text repairs, routing by diagnosis cannot beat sending every
flagged question to the image repair.
