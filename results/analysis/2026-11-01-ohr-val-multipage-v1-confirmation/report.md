# ohr_val_multipage_v1 confirmation policy comparison

835 questions on 259 documents; 647 flagged.

## Confirmation decision

Confirmation result (pre-registered, section 5)
FAAR beat the gated visual repair only if the Holm-adjusted McNemar p is below 0.05 and the document-level interval lies above 0.
Rule met: False. Valid comparison: True.

## Policy scores and costs

| Policy | EM | F1 | ANLS | NFKC EM | Measured USD | Reserved USD | Total USD | Measured / question | Reserved / question | Total / question | Requests | Text | Image requests | Images | Input tokens | Output tokens | Usage records | Latency ms | Latency records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_recovery | 0.250299 | 0.337751 | 0.297309 | 0.250299 | 1.407072 | 0.000000 | 1.407072 | 0.001685 | 0.000000 | 0.001685 | 835 | 835 | 0 | 0 | 1373492 | 6716 | 835 | 819106 | 835 |
| gated_visual | 0.326946 | 0.431905 | 0.382627 | 0.326946 | 7.068888 | 0.794997 | 7.863885 | 0.008466 | 0.000952 | 0.009418 | 1504 | 835 | 669 | 2000 | 7008873 | 12003 | 1479 | 1974645 | 1479 |
| faar | 0.282635 | 0.386415 | 0.340232 | 0.282635 | 2.685403 | 0.091050 | 2.776453 | 0.003216 | 0.000109 | 0.003325 | 1481 | 1452 | 29 | 87 | 2625023 | 12076 | 1478 | 1451133 | 1478 |
| random_20261101 | 0.285030 | 0.382794 | 0.340870 | 0.285030 | 4.036256 | 0.285862 | 4.322118 | 0.004834 | 0.000342 | 0.005176 | 1483 | 1253 | 230 | 687 | 3976751 | 11901 | 1474 | 1603940 | 1474 |
| random_20261102 | 0.287425 | 0.389352 | 0.348704 | 0.287425 | 3.933859 | 0.218901 | 4.152760 | 0.004711 | 0.000262 | 0.004973 | 1487 | 1276 | 211 | 629 | 3873909 | 11990 | 1480 | 1602465 | 1480 |
| random_20261103 | 0.281437 | 0.377812 | 0.337031 | 0.281437 | 3.879615 | 0.415623 | 4.295238 | 0.004646 | 0.000498 | 0.005144 | 1490 | 1280 | 210 | 629 | 3819345 | 12054 | 1477 | 1593479 | 1477 |
| all_retry_retrieval | 0.287425 | 0.393944 | 0.349568 | 0.287425 | 2.510109 | 0.000000 | 2.510109 | 0.003006 | 0.000000 | 0.003006 | 1476 | 1476 | 0 | 0 | 2449774 | 12067 | 1476 | 1425105 | 1476 |
| all_correct_text | 0.243114 | 0.330429 | 0.294685 | 0.243114 | 2.478461 | 0.000000 | 2.478461 | 0.002968 | 0.000000 | 0.002968 | 1480 | 1480 | 0 | 0 | 2419796 | 11733 | 1480 | 1435353 | 1480 |
| oracle | 0.368862 | 0.451474 | 0.405708 | 0.368862 | 1.990810 | 0.090830 | 2.081640 | 0.002384 | 0.000109 | 0.002493 | 937 | 877 | 60 | 179 | 1953945 | 7373 | 934 | 964094 | 934 |

Image settings: {"archive_sha256": "f9bc65f383172c4ea47940c47dfab01dd36c03a120bc0450d7a962917098c783", "detail": "high", "dpi": 150, "format": "JPEG", "long_side_max": 2048, "quality": 90}. Local processing time and one-time preparation cost: not recorded.

FAAR / gated visual total cost per question: 0.3530637846306247; similar cost.

## Contrasts

95% bootstrap intervals use 10,000 resamples and seed 20261031. Random pooled uses the mean score per question; its McNemar p-value is the maximum of the three seed p-values.

| Contrast | EM difference | Document interval | Question interval | McNemar p | Holm p |
| --- | ---: | --- | --- | ---: | ---: |
| primary | -0.044311 | [-0.068376, -0.021117] | [-0.065868, -0.021557] | 0.000132 | 0.0002954663547853897 |
| no_recovery | 0.032335 | [0.016911, 0.048045] | [0.016766, 0.049102] | 0.000098 | 0.0002954663547853897 |
| random_20261101 | -0.002395 | [-0.021202, 0.015913] | [-0.019192, 0.015569] | 0.893853 | n/a |
| random_20261102 | -0.004790 | [-0.023204, 0.013937] | [-0.023952, 0.014371] | 0.712231 | n/a |
| random_20261103 | 0.001198 | [-0.012009, 0.014541] | [-0.014371, 0.016766] | 1.000000 | n/a |
| descriptive_gated_visual (descriptive, unadjusted) | 0.076647 | [0.055831, 0.098660] | [0.057485, 0.097006] | 0.000000 | n/a |
| descriptive_all_retry_retrieval (descriptive, unadjusted) | 0.037126 | [0.021964, 0.053109] | [0.021557, 0.053892] | 0.000009 | n/a |
| descriptive_all_correct_text (descriptive, unadjusted) | -0.007186 | [-0.017119, 0.002162] | [-0.016766, 0.002395] | 0.210114 | n/a |
| descriptive_oracle (descriptive, unadjusted) | 0.118563 | [0.094470, 0.142687] | [0.097006, 0.141317] | 0.000000 | n/a |
| random_pooled | -0.001996 | [-0.015487, 0.011511] | [-0.015968, 0.012375] | 1.000000 | 1.0 |

## Secondary score contrasts

| Contrast | Metric | Difference | Document interval | Question interval |
| --- | --- | ---: | --- | --- |
| primary | f1 | -0.045489 | [-0.071911, -0.018220] | [-0.068608, -0.022107] |
| primary | anls | -0.042394 | [-0.065973, -0.018772] | [-0.064233, -0.020119] |
| primary | nfkc_em | -0.044311 | [-0.068376, -0.021117] | [-0.065868, -0.021557] |
| no_recovery | f1 | 0.048664 | [0.031718, 0.067129] | [0.032213, 0.065656] |
| no_recovery | anls | 0.042923 | [0.027037, 0.060105] | [0.027037, 0.060084] |
| no_recovery | nfkc_em | 0.032335 | [0.016911, 0.048045] | [0.016766, 0.049102] |
| random_20261101 | f1 | 0.003622 | [-0.016703, 0.023672] | [-0.014360, 0.022198] |
| random_20261101 | anls | -0.000638 | [-0.019627, 0.018254] | [-0.017065, 0.016556] |
| random_20261101 | nfkc_em | -0.002395 | [-0.021202, 0.015913] | [-0.019192, 0.015569] |
| random_20261102 | f1 | -0.002936 | [-0.022454, 0.017371] | [-0.021997, 0.016325] |
| random_20261102 | anls | -0.008472 | [-0.026416, 0.009498] | [-0.027076, 0.010172] |
| random_20261102 | nfkc_em | -0.004790 | [-0.023204, 0.013937] | [-0.023952, 0.014371] |
| random_20261103 | f1 | 0.008603 | [-0.006192, 0.023127] | [-0.007803, 0.024964] |
| random_20261103 | anls | 0.003201 | [-0.010268, 0.016135] | [-0.012101, 0.018810] |
| random_20261103 | nfkc_em | 0.001198 | [-0.012009, 0.014541] | [-0.014371, 0.016766] |
| descriptive_gated_visual | f1 | 0.094153 | [0.071185, 0.118538] | [0.073904, 0.115059] |
| descriptive_gated_visual | anls | 0.085317 | [0.062805, 0.108906] | [0.065680, 0.105463] |
| descriptive_gated_visual | nfkc_em | 0.076647 | [0.055831, 0.098660] | [0.057485, 0.097006] |
| descriptive_all_retry_retrieval | f1 | 0.056193 | [0.039004, 0.075198] | [0.038754, 0.073813] |
| descriptive_all_retry_retrieval | anls | 0.052259 | [0.036258, 0.069725] | [0.035520, 0.069601] |
| descriptive_all_retry_retrieval | nfkc_em | 0.037126 | [0.021964, 0.053109] | [0.021557, 0.053892] |
| descriptive_all_correct_text | f1 | -0.007323 | [-0.016429, 0.000930] | [-0.016010, 0.000963] |
| descriptive_all_correct_text | anls | -0.002624 | [-0.010615, 0.004649] | [-0.010162, 0.004778] |
| descriptive_all_correct_text | nfkc_em | -0.007186 | [-0.017119, 0.002162] | [-0.016766, 0.002395] |
| descriptive_oracle | f1 | 0.113723 | [0.090531, 0.137463] | [0.092983, 0.135599] |
| descriptive_oracle | anls | 0.108398 | [0.085992, 0.131083] | [0.088676, 0.129324] |
| descriptive_oracle | nfkc_em | 0.118563 | [0.094470, 0.142687] | [0.097006, 0.141317] |
| random_pooled | f1 | 0.003096 | [-0.011613, 0.018166] | [-0.011343, 0.017957] |
| random_pooled | anls | -0.001969 | [-0.015119, 0.011487] | [-0.015509, 0.012278] |
| random_pooled | nfkc_em | -0.001996 | [-0.015487, 0.011511] | [-0.015968, 0.012375] |

## Flagged questions: repair and damage

95% Clopper-Pearson intervals.

| Policy | Repaired / initially wrong | Rate [interval] | Damaged / initially right | Rate [interval] |
| --- | --- | --- | --- | --- |
| no_recovery | 0 / 536 | 0.0 [0.0, 0.006858609425452974] | 0 / 111 | 0.0 [0.0, 0.032686994023947695] |
| gated_visual | 70.0 / 536 | 0.13059701492537312 [0.1032400272335816, 0.16210183384871585] | 6 / 111 | 0.05405405405405406 [0.02009265930446559, 0.11394448770454148] |
| faar | 37.0 / 536 | 0.06902985074626866 [0.049065720989994316, 0.09389592480285498] | 10 / 111 | 0.09009009009009009 [0.04405095002535123, 0.15944240892804004] |
| random_20261101 | 36.0 / 536 | 0.06716417910447761 [0.047480858552432594, 0.091773886898879] | 7 / 111 | 0.06306306306306306 [0.025728201214302242, 0.1256231426964945] |
| random_20261102 | 42.0 / 536 | 0.07835820895522388 [0.05705487436257294, 0.10444342651655109] | 11 / 111 | 0.0990990990990991 [0.05051642089401935, 0.17039968195273836] |
| random_20261103 | 35.0 / 536 | 0.06529850746268656 [0.04590070583164953, 0.08964730568379665] | 9 / 111 | 0.08108108108108109 [0.037746404003455365, 0.1483416859623683] |
| all_retry_retrieval | 40 / 536 | 0.07462686567164178 [0.053846855669236846, 0.10023638746277344] | 9 / 111 | 0.08108108108108109 [0.037746404003455365, 0.1483416859623683] |
| all_correct_text | 5 / 536 | 0.009328358208955223 [0.00303563335151084, 0.021634137864139685] | 11 / 111 | 0.0990990990990991 [0.05051642089401935, 0.17039968195273836] |
| oracle | 99 / 536 | 0.18470149253731344 [0.1527306790829971, 0.22019401429407504] | 0 / 111 | 0.0 [0.0, 0.032686994023947695] |

## Diagnoses and route scores

Diagnosis counts: {"missing_text": 17, "semantic": 492, "structural": 10, "word_level": 128}.

| Route | n | Initial EM | Retry EM | Correct EM | Visual EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| retry_retrieval | 492 | 0.13008130081300814 | 0.18699186991869918 | 0.12601626016260162 | 0.23170731707317074 |
| correct_text | 128 | 0.2890625 | 0.328125 | 0.2734375 | 0.390625 |
| invoke_vlm | 27 | 0.37037037037037035 | 0.2962962962962963 | 0.2962962962962963 | 0.4074074074074074 |

Skip reasons: {"correct_text": {"correction_changed_nothing": 2}, "invoke_vlm": {}, "retry_retrieval": {"nothing_new_retrieved": 6}}.

## Execution

| Run | Questions | Unknown outcomes | Questions with unknown outcomes | Resends | Blocked | Blocked rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 835 | 0 | 0 | 0 | 0 | 0.000000 |
| retry_retrieval | 647 | 0 | 0 | 0 | 0 | 0.000000 |
| correct_text | 647 | 0 | 0 | 0 | 0 | 0.000000 |
| invoke_vlm | 647 | 25 | 22 | 22 | 3 | 0.004637 |

Rates over 2%: {}; compared rate differences over 1 point: {}.

## Disagreement and power

Disagreement rate: 0.108982. Detectable gain at 80% power: 0.03196132199379808.
n = (z(0.975)*sqrt(d) + z(0.8)*sqrt(d - delta^2))^2 / delta^2; d is disagreement rate

## Descriptive headroom (not a claim)

The oracle selects the best of the initial answer and repairs in hindsight using gold answers and is not deployable; this is descriptive only.

| Policy | EM | Total USD per question |
| --- | ---: | ---: |
| Oracle | 0.368862 | 0.002493 |
| Gated visual repair | 0.326946 | 0.009418 |
