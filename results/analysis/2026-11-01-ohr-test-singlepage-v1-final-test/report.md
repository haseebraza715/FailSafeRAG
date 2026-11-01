# ohr_test_singlepage_v1 final test policy comparison (secondary)

330 questions on 240 documents; 198 flagged.

## Secondary result

Secondary result (pre-registered, section 15): outside the primary Holm family; cannot change the primary decision
FAAR beat the gated visual repair only if the Holm-adjusted McNemar p is below 0.05 and the document-level interval lies above 0.
Rule met: False. Valid comparison: True.

## Policy scores and costs

| Policy | EM | F1 | ANLS | NFKC EM | Measured USD | Reserved USD | Total USD | Measured / question | Reserved / question | Total / question | Requests | Text | Image requests | Images | Input tokens | Output tokens | Usage records | Latency ms | Latency records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_recovery | 0.293939 | 0.524200 | 0.489214 | 0.300000 | 0.510896 | 0.000000 | 0.510896 | 0.001548 | 0.000000 | 0.001548 | 307 | 307 | 0 | 0 | 480926 | 5994 | 307 | 364204 | 307 |
| gated_visual | 0.345455 | 0.603599 | 0.549658 | 0.354545 | 1.343197 | 0.000000 | 1.343197 | 0.004070 | 0.000000 | 0.004070 | 505 | 307 | 198 | 198 | 1291642 | 10311 | 505 | 703047 | 505 |
| faar | 0.327273 | 0.573947 | 0.531762 | 0.333333 | 0.791271 | 0.000000 | 0.791271 | 0.002398 | 0.000000 | 0.002398 | 454 | 428 | 26 | 26 | 746356 | 8983 | 454 | 536508 | 454 |
| random_20261101 | 0.324242 | 0.561878 | 0.522680 | 0.330303 | 0.919629 | 0.000000 | 0.919629 | 0.002787 | 0.000000 | 0.002787 | 444 | 371 | 73 | 73 | 875509 | 8824 | 444 | 563011 | 444 |
| random_20261102 | 0.318182 | 0.557072 | 0.516278 | 0.321212 | 0.910366 | 0.000000 | 0.910366 | 0.002759 | 0.000000 | 0.002759 | 445 | 379 | 66 | 66 | 864106 | 9252 | 445 | 563494 | 445 |
| random_20261103 | 0.318182 | 0.549244 | 0.506592 | 0.324242 | 0.902898 | 0.000000 | 0.902898 | 0.002736 | 0.000000 | 0.002736 | 438 | 371 | 67 | 67 | 857933 | 8993 | 438 | 550616 | 438 |
| all_retry_retrieval | 0.306061 | 0.542967 | 0.505101 | 0.312121 | 0.693374 | 0.000000 | 0.693374 | 0.002101 | 0.000000 | 0.002101 | 401 | 401 | 0 | 0 | 650659 | 8543 | 401 | 474778 | 401 |
| all_correct_text | 0.296970 | 0.518293 | 0.484344 | 0.303030 | 0.666889 | 0.000000 | 0.666889 | 0.002021 | 0.000000 | 0.002021 | 409 | 409 | 0 | 0 | 626659 | 8046 | 409 | 480092 | 409 |
| oracle | 0.360606 | 0.585052 | 0.542288 | 0.363636 | 0.574805 | 0.000000 | 0.574805 | 0.001742 | 0.000000 | 0.001742 | 329 | 312 | 17 | 17 | 543335 | 6294 | 329 | 394453 | 329 |

Image settings: {"archive_sha256": "f9bc65f383172c4ea47940c47dfab01dd36c03a120bc0450d7a962917098c783", "detail": "high", "dpi": 150, "format": "JPEG", "long_side_max": 2048, "quality": 90}. Local processing time and one-time preparation cost: not recorded.

FAAR / gated visual total cost per question: 0.5890952704629329; similar cost.

## Contrasts

95% bootstrap intervals use 10,000 resamples and seed 20261031. Random pooled uses the mean score per question; its McNemar p-value is the maximum of the three seed p-values.

| Contrast | EM difference | Document interval | Question interval | McNemar p | Holm p |
| --- | ---: | --- | --- | ---: | ---: |
| primary | -0.018182 | [-0.042553, 0.005666] | [-0.042424, 0.006061] | 0.210114 | 0.42022705078125 |
| no_recovery | 0.033333 | [0.013966, 0.055385] | [0.012121, 0.054545] | 0.003418 | 0.01025390625 |
| random_20261101 | 0.003030 | [-0.017443, 0.023121] | [-0.015152, 0.024242] | 1.000000 | n/a |
| random_20261102 | 0.009091 | [-0.012699, 0.032544] | [-0.012121, 0.033333] | 0.607239 | n/a |
| random_20261103 | 0.009091 | [-0.013029, 0.033034] | [-0.012121, 0.033333] | 0.607239 | n/a |
| descriptive_gated_visual (descriptive, unadjusted) | 0.051515 | [0.024766, 0.080247] | [0.024242, 0.078788] | 0.000488 | n/a |
| descriptive_all_retry_retrieval (descriptive, unadjusted) | 0.012121 | [0.002933, 0.024691] | [0.003030, 0.024242] | 0.125000 | n/a |
| descriptive_all_correct_text (descriptive, unadjusted) | 0.003030 | [-0.006211, 0.014535] | [-0.006061, 0.015152] | 1.000000 | n/a |
| descriptive_oracle (descriptive, unadjusted) | 0.066667 | [0.041008, 0.094463] | [0.042424, 0.093939] | 0.000000 | n/a |
| random_pooled | 0.007071 | [-0.010945, 0.025237] | [-0.010101, 0.025253] | 1.000000 | 1.0 |

## Secondary score contrasts

| Contrast | Metric | Difference | Document interval | Question interval |
| --- | --- | ---: | --- | --- |
| primary | f1 | -0.029652 | [-0.056128, -0.005270] | [-0.054480, -0.005972] |
| primary | anls | -0.017897 | [-0.040709, 0.003679] | [-0.040986, 0.003354] |
| primary | nfkc_em | -0.021212 | [-0.046440, 0.002985] | [-0.045455, 0.003030] |
| no_recovery | f1 | 0.049748 | [0.025225, 0.076132] | [0.026724, 0.075035] |
| no_recovery | anls | 0.042547 | [0.019854, 0.067566] | [0.021071, 0.066366] |
| no_recovery | nfkc_em | 0.033333 | [0.013966, 0.055385] | [0.012121, 0.054545] |
| random_20261101 | f1 | 0.012069 | [-0.010281, 0.034564] | [-0.009749, 0.034648] |
| random_20261101 | anls | 0.009082 | [-0.010457, 0.028706] | [-0.010235, 0.029173] |
| random_20261101 | nfkc_em | 0.003030 | [-0.017443, 0.023121] | [-0.015152, 0.024242] |
| random_20261102 | f1 | 0.016875 | [-0.009148, 0.043248] | [-0.008048, 0.043223] |
| random_20261102 | anls | 0.015484 | [-0.007422, 0.038865] | [-0.007382, 0.038816] |
| random_20261102 | nfkc_em | 0.012121 | [-0.009259, 0.034485] | [-0.009091, 0.033333] |
| random_20261103 | f1 | 0.024703 | [-0.000598, 0.050482] | [0.000630, 0.049790] |
| random_20261103 | anls | 0.025169 | [0.002961, 0.048311] | [0.003718, 0.047963] |
| random_20261103 | nfkc_em | 0.009091 | [-0.013029, 0.033034] | [-0.012121, 0.033333] |
| descriptive_gated_visual | f1 | 0.079400 | [0.047737, 0.112930] | [0.048100, 0.111523] |
| descriptive_gated_visual | anls | 0.060444 | [0.030295, 0.092529] | [0.032495, 0.090970] |
| descriptive_gated_visual | nfkc_em | 0.054545 | [0.027108, 0.084592] | [0.027273, 0.084848] |
| descriptive_all_retry_retrieval | f1 | 0.018768 | [0.006019, 0.033953] | [0.006430, 0.033385] |
| descriptive_all_retry_retrieval | anls | 0.015887 | [0.003784, 0.030280] | [0.003514, 0.030579] |
| descriptive_all_retry_retrieval | nfkc_em | 0.012121 | [0.002933, 0.024691] | [0.003030, 0.024242] |
| descriptive_all_correct_text | f1 | -0.005906 | [-0.017267, 0.002825] | [-0.015393, 0.002087] |
| descriptive_all_correct_text | anls | -0.004870 | [-0.012810, 0.002288] | [-0.012825, 0.002354] |
| descriptive_all_correct_text | nfkc_em | 0.003030 | [-0.006211, 0.014535] | [-0.006061, 0.015152] |
| descriptive_oracle | f1 | 0.060852 | [0.036732, 0.087142] | [0.036532, 0.086869] |
| descriptive_oracle | anls | 0.053074 | [0.030763, 0.077458] | [0.030800, 0.077746] |
| descriptive_oracle | nfkc_em | 0.063636 | [0.038576, 0.091195] | [0.039394, 0.090909] |
| random_pooled | f1 | 0.017882 | [-0.001983, 0.038333] | [-0.001260, 0.037738] |
| random_pooled | anls | 0.016578 | [-0.000565, 0.034367] | [-0.000186, 0.034335] |
| random_pooled | nfkc_em | 0.008081 | [-0.009805, 0.026042] | [-0.009091, 0.026263] |

## Flagged questions: repair and damage

95% Clopper-Pearson intervals.

| Policy | Repaired / initially wrong | Rate [interval] | Damaged / initially right | Rate [interval] |
| --- | --- | --- | --- | --- |
| no_recovery | 0.0 / 154 | 0.0 [0.0, 0.023669148372542772] | 0 / 44 | 0.0 [0.0, 0.08041993976357587] |
| gated_visual | 20 / 154 | 0.12987012987012986 [0.08116628783198608, 0.19342644996192487] | 3 / 44 | 0.06818181818181818 [0.014287668426975389, 0.18656201007202233] |
| faar | 12 / 154 | 0.07792207792207792 [0.04091098735376322, 0.13215791655397827] | 1 / 44 | 0.022727272727272728 [0.0005752392133630829, 0.12024158220634734] |
| random_20261101 | 12.0 / 154 | 0.07792207792207792 [0.04091098735376322, 0.13215791655397827] | 2 / 44 | 0.045454545454545456 [0.005552951995370494, 0.15473157765898934] |
| random_20261102 | 8.0 / 154 | 0.05194805194805195 [0.022691322331244903, 0.09979076774374385] | 0 / 44 | 0.0 [0.0, 0.08041993976357587] |
| random_20261103 | 10.0 / 154 | 0.06493506493506493 [0.03157543125795952, 0.11617453295680899] | 2 / 44 | 0.045454545454545456 [0.005552951995370494, 0.15473157765898934] |
| all_retry_retrieval | 4.0 / 154 | 0.025974025974025976 [0.007121413829478186, 0.06516643280288] | 0 / 44 | 0.0 [0.0, 0.08041993976357587] |
| all_correct_text | 2.0 / 154 | 0.012987012987012988 [0.0015766753956051492, 0.0461244509556018] | 1 / 44 | 0.022727272727272728 [0.0005752392133630829, 0.12024158220634734] |
| oracle | 22.0 / 154 | 0.14285714285714285 [0.09174423389007916, 0.20826292895084628] | 0 / 44 | 0.0 [0.0, 0.08041993976357587] |

## Diagnoses and route scores

Diagnosis counts: {"missing_text": 23, "semantic": 60, "structural": 3, "word_level": 112}.

| Route | n | Initial EM | Retry EM | Correct EM | Visual EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| retry_retrieval | 60 | 0.2 | 0.25 | 0.21666666666666667 | 0.3333333333333333 |
| correct_text | 112 | 0.2857142857142857 | 0.29464285714285715 | 0.2857142857142857 | 0.29464285714285715 |
| invoke_vlm | 26 | 0.0 | 0.0 | 0.0 | 0.3076923076923077 |

Skip reasons: {"correct_text": {"correction_changed_nothing": 73, "no_text_evidence": 23}, "invoke_vlm": {}, "retry_retrieval": {"no_text_evidence": 23, "nothing_new_retrieved": 81}}.

## Execution

| Run | Questions | Unknown outcomes | Questions with unknown outcomes | Resends | Blocked | Blocked rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 330 | 0 | 0 | 0 | 0 | 0.000000 |
| retry_retrieval | 198 | 0 | 0 | 0 | 0 | 0.000000 |
| correct_text | 198 | 0 | 0 | 0 | 0 | 0.000000 |
| invoke_vlm | 198 | 0 | 0 | 0 | 0 | 0.000000 |

Rates over 2%: {}; compared rate differences over 1 point: {}.

## Disagreement and power

Disagreement rate: 0.048485. Detectable gain at 80% power: 0.03383743495554293.
n = (z(0.975)*sqrt(d) + z(0.8)*sqrt(d - delta^2))^2 / delta^2; d is disagreement rate

## Descriptive headroom (not a claim)

The oracle selects the best of the initial answer and repairs in hindsight using gold answers and is not deployable; this is descriptive only.

| Policy | EM | Total USD per question |
| --- | ---: | ---: |
| Oracle | 0.360606 | 0.001742 |
| Gated visual repair | 0.345455 | 0.004070 |
