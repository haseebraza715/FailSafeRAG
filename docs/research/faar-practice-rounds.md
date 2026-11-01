# FAAR practice rounds on `ohr_long_dev_v1`

Development log for the practice stage of [faar-vs-gated-preregistration.md](faar-vs-gated-preregistration.md)
(sections 12 and 13). Every number here comes from a registered run in `experiments/registry.jsonl` and the
analysis under `results/analysis/`. These are development results on train data, not a confirmation.

## Round 1 (2026-11-01)

**What ran.** Code at `36087c9` (gate chooser, flagged subsets, policy comparison); the comparison ran at `caa96e4`. Runs:

| Run | What | Measured USD | Reserved USD |
| --- | --- | ---: | ---: |
| `2026-11-01-ohr-long-dev-v1-baseline-answer-v2-terra-r1` | no recovery, 423 questions, 0 unknown outcomes | 0.705775 | 0 |
| `2026-11-01-ohr-long-dev-v1-r1-gate-choice` | offline: Youden's J choice and diagnosis | 0 | 0 |
| `2026-11-01-ohr-long-dev-v1-r1-retry-exclude-seen-terra` | `retry_retrieval`, `exclude-seen-v1`, 201 sent, 27 skipped | 0.312075 | 0 |
| `2026-11-01-ohr-long-dev-v1-r1-correct-symspell-terra` | `correct_text`, SymSpell, 227 sent, 1 skipped | 0.377510 | 0 |
| `2026-11-01-ohr-long-dev-v1-r1-visual-terra` | `invoke_vlm`, 228 sent, 14 connection drops, 14 resends, 0 blocked | 1.887237 | 0.413327 |

Stage total after round 1: 3.282597 USD measured, 3.695924 USD counting reserved cost, of the 10 USD cap.

**Gate.** The chooser picked `query_token_coverage < 0.6923076923076923` (J 0.116, TPR 0.248, FPR 0.132; it flags
48 of the 243 questions the base gate passes, cap 72). The gate flags 228 of 423 questions. Diagnoses of the flagged
questions: `semantic` 161, `word_level` 44, `missing_text` 16, `structural` 7.

**Measured** (`results/analysis/2026-11-01-ohr-long-dev-v1-round-1/report.md`; an independent review recomputed
these numbers from the raw run files and they matched):

| Policy | Official EM | Total USD per question |
| --- | ---: | ---: |
| No recovery | 0.310 | 0.00167 |
| Gated visual repair (the bar) | 0.397 | 0.00711 |
| FAAR | 0.307 | 0.00287 |
| Random repair, 3 seeds | 0.324, 0.333, 0.338 | 0.0036 to 0.0040 |
| `retry_retrieval` on every flagged question | 0.284 | 0.00241 |
| `correct_text` on every flagged question | 0.303 | 0.00256 |
| Best repair per question in hindsight (diagnostic only) | 0.416 | 0.00252 |

FAAR minus the bar: -0.090 EM (38 fewer correct of 423), document interval [-0.132, -0.051], McNemar p 1.4e-8.
FAAR minus no recovery: -0.002, interval [-0.040, 0.037]. FAAR's cost is 0.40 times the bar's.

EM of each repair on the questions each route received:

| Route (n) | Initial | Retry | Correct | Image |
| --- | ---: | ---: | ---: | ---: |
| `retry_retrieval` (161) | 0.124 | 0.075 | 0.099 | 0.280 |
| `correct_text` (44) | 0.318 | 0.250 | 0.318 | 0.432 |
| `invoke_vlm` (23) | 0.217 | 0.217 | 0.261 | 0.522 |

**Reading (interpretation, not a measurement).** The image repair scored highest on every route, and neither text
repair beat the initial answer on any route. Diagnosis can only help if some route has a text repair that beats the
image there, so with these repairs FAAR cannot beat the bar on this sample. `exclude-seen-v1` drops all 5 initial
chunks, which may explain why retry scores below the initial answer.

**Changes for round 2.** A retry rule that keeps the initial top 3 chunks and adds the 2 best unseen ones
(`keep-top3-add-next2-v1`, allowed by amendment 3 item 5). Round 2 reruns only `retry_retrieval` and the comparison.
