# ohr_test_multipage_v1 final test policy comparison (primary)

946 questions on 327 documents; 706 flagged.

## Final test decision

Final test result (pre-registered, section 15; primary)
FAAR beat the gated visual repair only if the Holm-adjusted McNemar p is below 0.05 and the document-level interval lies above 0.
Rule met: False. Valid comparison: True.

## Policy scores and costs

| Policy | EM | F1 | ANLS | NFKC EM | Measured USD | Reserved USD | Total USD | Measured / question | Reserved / question | Total / question | Requests | Text | Image requests | Images | Input tokens | Output tokens | Usage records | Latency ms | Latency records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_recovery | 0.235729 | 0.336758 | 0.293937 | 0.235729 | 1.615026 | 0.000000 | 1.615026 | 0.001707 | 0.000000 | 0.001707 | 946 | 946 | 0 | 0 | 1574766 | 8052 | 946 | 923044 | 946 |
| gated_visual | 0.315011 | 0.435736 | 0.382214 | 0.315011 | 7.863495 | 0.000000 | 7.863495 | 0.008312 | 0.000000 | 0.008312 | 1652 | 946 | 706 | 2101 | 7793675 | 13964 | 1652 | 2151065 | 1652 |
| faar | 0.266385 | 0.380886 | 0.330471 | 0.266385 | 3.030497 | 0.000000 | 3.030497 | 0.003203 | 0.000000 | 0.003203 | 1647 | 1618 | 29 | 85 | 2962437 | 13612 | 1647 | 1592713 | 1647 |
| random_20261101 | 0.273784 | 0.387395 | 0.342132 | 0.273784 | 4.448000 | 0.000000 | 4.448000 | 0.004702 | 0.000000 | 0.004702 | 1648 | 1419 | 229 | 681 | 4379465 | 13707 | 1648 | 1770931 | 1648 |
| random_20261102 | 0.278013 | 0.392020 | 0.340185 | 0.278013 | 4.607478 | 0.000000 | 4.607478 | 0.004870 | 0.000000 | 0.004870 | 1649 | 1399 | 250 | 747 | 4538763 | 13743 | 1649 | 1780964 | 1649 |
| random_20261103 | 0.273784 | 0.385933 | 0.339912 | 0.273784 | 4.273404 | 0.000000 | 4.273404 | 0.004517 | 0.000000 | 0.004517 | 1644 | 1440 | 204 | 609 | 4205169 | 13647 | 1644 | 1738870 | 1644 |
| all_retry_retrieval | 0.265328 | 0.378884 | 0.332780 | 0.265328 | 2.823281 | 0.000000 | 2.823281 | 0.002984 | 0.000000 | 0.002984 | 1640 | 1640 | 0 | 0 | 2754856 | 13685 | 1640 | 1575686 | 1640 |
| all_correct_text | 0.228330 | 0.330647 | 0.285209 | 0.228330 | 2.798773 | 0.000000 | 2.798773 | 0.002959 | 0.000000 | 0.002959 | 1650 | 1650 | 0 | 0 | 2731943 | 13366 | 1650 | 1572800 | 1650 |
| oracle | 0.345666 | 0.444781 | 0.395614 | 0.345666 | 2.312112 | 0.000000 | 2.312112 | 0.002444 | 0.000000 | 0.002444 | 1050 | 979 | 71 | 210 | 2268462 | 8730 | 1050 | 1079622 | 1050 |

Image settings: {"archive_sha256": "f9bc65f383172c4ea47940c47dfab01dd36c03a120bc0450d7a962917098c783", "detail": "high", "dpi": 150, "format": "JPEG", "long_side_max": 2048, "quality": 90}. Local processing time and one-time preparation cost: not recorded.

FAAR / gated visual total cost per question: 0.38538804946146715; similar cost.

## Contrasts

95% bootstrap intervals use 10,000 resamples and seed 20261031. Random pooled uses the mean score per question; its McNemar p-value is the maximum of the three seed p-values.

| Contrast | EM difference | Document interval | Question interval | McNemar p | Holm p |
| --- | ---: | --- | --- | ---: | ---: |
| primary | -0.048626 | [-0.073327, -0.026058] | [-0.068710, -0.028541] | 0.000002 | 6.608279494650746e-06 |
| no_recovery | 0.030655 | [0.016048, 0.046919] | [0.017970, 0.044397] | 0.000002 | 6.608279494650746e-06 |
| random_20261101 | -0.007400 | [-0.024758, 0.011226] | [-0.023256, 0.008457] | 0.427043 | n/a |
| random_20261102 | -0.011628 | [-0.029928, 0.006303] | [-0.027484, 0.003171] | 0.177001 | n/a |
| random_20261103 | -0.007400 | [-0.022827, 0.007307] | [-0.022199, 0.007400] | 0.391603 | n/a |
| descriptive_gated_visual (descriptive, unadjusted) | 0.079281 | [0.055439, 0.106807] | [0.060254, 0.098309] | 0.000000 | n/a |
| descriptive_all_retry_retrieval (descriptive, unadjusted) | 0.029598 | [0.016741, 0.042604] | [0.016913, 0.042283] | 0.000004 | n/a |
| descriptive_all_correct_text (descriptive, unadjusted) | -0.007400 | [-0.013889, -0.001842] | [-0.013742, -0.002114] | 0.039062 | n/a |
| descriptive_oracle (descriptive, unadjusted) | 0.109937 | [0.085864, 0.136364] | [0.089852, 0.130021] | 0.000000 | n/a |
| random_pooled | -0.008809 | [-0.022658, 0.005556] | [-0.020789, 0.003524] | 0.427043 | 0.42704285650325524 |

## Secondary score contrasts

| Contrast | Metric | Difference | Document interval | Question interval |
| --- | --- | ---: | --- | --- |
| primary | f1 | -0.054850 | [-0.081260, -0.031083] | [-0.076140, -0.034131] |
| primary | anls | -0.051742 | [-0.076654, -0.028316] | [-0.072995, -0.030744] |
| primary | nfkc_em | -0.048626 | [-0.073327, -0.026058] | [-0.068710, -0.028541] |
| no_recovery | f1 | 0.044128 | [0.029371, 0.060691] | [0.030384, 0.058404] |
| no_recovery | anls | 0.036535 | [0.020990, 0.053606] | [0.022663, 0.051299] |
| no_recovery | nfkc_em | 0.030655 | [0.016048, 0.046919] | [0.017970, 0.044397] |
| random_20261101 | f1 | -0.006510 | [-0.025360, 0.013349] | [-0.023770, 0.010043] |
| random_20261101 | anls | -0.011661 | [-0.030347, 0.007982] | [-0.028392, 0.004842] |
| random_20261101 | nfkc_em | -0.007400 | [-0.024758, 0.011226] | [-0.023256, 0.008457] |
| random_20261102 | f1 | -0.011134 | [-0.030433, 0.007532] | [-0.027708, 0.004682] |
| random_20261102 | anls | -0.009713 | [-0.027218, 0.007132] | [-0.025337, 0.005645] |
| random_20261102 | nfkc_em | -0.011628 | [-0.029928, 0.006303] | [-0.027484, 0.003171] |
| random_20261103 | f1 | -0.005048 | [-0.021578, 0.010743] | [-0.020249, 0.010142] |
| random_20261103 | anls | -0.009441 | [-0.025362, 0.005653] | [-0.024125, 0.005172] |
| random_20261103 | nfkc_em | -0.007400 | [-0.022827, 0.007307] | [-0.022199, 0.007400] |
| descriptive_gated_visual | f1 | 0.098978 | [0.074565, 0.127157] | [0.079838, 0.118835] |
| descriptive_gated_visual | anls | 0.088277 | [0.064252, 0.115462] | [0.069049, 0.107913] |
| descriptive_gated_visual | nfkc_em | 0.079281 | [0.055439, 0.106807] | [0.060254, 0.098309] |
| descriptive_all_retry_retrieval | f1 | 0.042126 | [0.027877, 0.056336] | [0.028422, 0.056440] |
| descriptive_all_retry_retrieval | anls | 0.038843 | [0.024184, 0.053563] | [0.024735, 0.053948] |
| descriptive_all_retry_retrieval | nfkc_em | 0.029598 | [0.016741, 0.042604] | [0.016913, 0.042283] |
| descriptive_all_correct_text | f1 | -0.006110 | [-0.011896, -0.000712] | [-0.012228, -0.000491] |
| descriptive_all_correct_text | anls | -0.008728 | [-0.015920, -0.002386] | [-0.015365, -0.002721] |
| descriptive_all_correct_text | nfkc_em | -0.007400 | [-0.013889, -0.001842] | [-0.013742, -0.002114] |
| descriptive_oracle | f1 | 0.108024 | [0.084209, 0.134351] | [0.088141, 0.128461] |
| descriptive_oracle | anls | 0.101677 | [0.079431, 0.126376] | [0.082929, 0.121047] |
| descriptive_oracle | nfkc_em | 0.109937 | [0.085864, 0.136364] | [0.089852, 0.130021] |
| random_pooled | f1 | -0.007564 | [-0.022907, 0.007696] | [-0.020480, 0.005137] |
| random_pooled | anls | -0.010272 | [-0.024245, 0.004007] | [-0.022191, 0.001920] |
| random_pooled | nfkc_em | -0.008809 | [-0.022658, 0.005556] | [-0.020789, 0.003524] |

## Flagged questions: repair and damage

95% Clopper-Pearson intervals.

| Policy | Repaired / initially wrong | Rate [interval] | Damaged / initially right | Rate [interval] |
| --- | --- | --- | --- | --- |
| no_recovery | 0 / 597 | 0.0 [0.0, 0.0061599766294688305] | 0 / 109 | 0.0 [0.0, 0.03327666479850845] |
| gated_visual | 82 / 597 | 0.1373534338358459 [0.1107539380138255, 0.16759927698354335] | 7 / 109 | 0.06422018348623854 [0.026207450690140837, 0.12784822372635282] |
| faar | 34 / 597 | 0.05695142378559464 [0.03975926360240316, 0.07867770592350556] | 5 / 109 | 0.045871559633027525 [0.015059902327092457, 0.10380691431609146] |
| random_20261101 | 45 / 597 | 0.07537688442211055 [0.05550887247981101, 0.09956225005059244] | 9 / 109 | 0.08256880733944955 [0.0384519779009497, 0.15095990241631646] |
| random_20261102 | 46 / 597 | 0.07705192629815745 [0.05696126423425549, 0.10144087505814962] | 6 / 109 | 0.05504587155963303 [0.02046622377682347, 0.11596653140905414] |
| random_20261103 | 42 / 597 | 0.07035175879396985 [0.051170472146032694, 0.09390814069636742] | 6 / 109 | 0.05504587155963303 [0.02046622377682347, 0.11596653140905414] |
| all_retry_retrieval | 33 / 597 | 0.05527638190954774 [0.03835153974109456, 0.07675588793652674] | 5 / 109 | 0.045871559633027525 [0.015059902327092457, 0.10380691431609146] |
| all_correct_text | 1 / 597 | 0.0016750418760469012 [4.2407489360400586e-05, 0.009297080684615738] | 8 / 109 | 0.07339449541284404 [0.03221801247960134, 0.13950088856734233] |
| oracle | 104 / 597 | 0.17420435510887772 [0.14461573662949126, 0.20705751318704912] | 0 / 109 | 0.0 [0.0, 0.03327666479850845] |

## Diagnoses and route scores

Diagnosis counts: {"missing_text": 19, "semantic": 541, "structural": 10, "word_level": 136}.

| Route | n | Initial EM | Retry EM | Correct EM | Visual EM |
| --- | ---: | ---: | ---: | ---: | ---: |
| retry_retrieval | 541 | 0.13123844731977818 | 0.17744916820702403 | 0.12384473197781885 | 0.23105360443622922 |
| correct_text | 136 | 0.22794117647058823 | 0.25 | 0.21323529411764705 | 0.3382352941176471 |
| invoke_vlm | 29 | 0.2413793103448276 | 0.2413793103448276 | 0.20689655172413793 | 0.4482758620689655 |

Skip reasons: {"correct_text": {"correction_changed_nothing": 2}, "invoke_vlm": {}, "retry_retrieval": {"nothing_new_retrieved": 12}}.

## Execution

| Run | Questions | Unknown outcomes | Questions with unknown outcomes | Resends | Blocked | Blocked rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 946 | 0 | 0 | 0 | 0 | 0.000000 |
| retry_retrieval | 706 | 0 | 0 | 0 | 0 | 0.000000 |
| correct_text | 706 | 0 | 0 | 0 | 0 | 0.000000 |
| invoke_vlm | 706 | 0 | 0 | 0 | 0 | 0.000000 |

Rates over 2%: {}; compared rate differences over 1 point: {}.

## Disagreement and power

Disagreement rate: 0.099366. Detectable gain at 80% power: 0.02867710351891278.
n = (z(0.975)*sqrt(d) + z(0.8)*sqrt(d - delta^2))^2 / delta^2; d is disagreement rate

## Descriptive headroom (not a claim)

The oracle selects the best of the initial answer and repairs in hindsight using gold answers and is not deployable; this is descriptive only.

| Policy | EM | Total USD per question |
| --- | ---: | ---: |
| Oracle | 0.345666 | 0.002444 |
| Gated visual repair | 0.315011 | 0.008312 |
