# Router signal exploration (2026-11-15)

Offline. Features are runtime-only; EM columns are used as targets and for scoring. Intervals are 95% percentile bootstraps resampling documents (seed 20261115). Practice diagnosis file used: `results/development/2026-11-01-ohr-long-dev-v1-r1-diagnosis/diagnosis.jsonl` (the gate-choice folder holds no diagnosis jsonl; this one matches case_scores on flagged, gate reasons and diagnosis for 423/423).

## 1. Outcome classes among flagged questions

| sample | questions | docs | flagged | text_only | image_only | both | neither | initial right (flagged) |
|---|---|---|---|---|---|---|---|---|
| practice | 423 | 101 | 228 | 5 | 28 | 48 | 147 | 39 |
| validation | 835 | 259 | 647 | 35 | 57 | 118 | 437 | 111 |
| test | 946 | 327 | 706 | 27 | 71 | 113 | 495 | 109 |

Test documents also in validation: 252 of 327 (in practice: 0; shared question ids: 0).

## 2. Features

68 runtime features: top_fused_score, fused_margin, query_token_coverage, weird_char_ratio, layout_signal_count, layout_excess_whitespace, layout_bare_numbers, layout_length_variance, layout_pipe_heavy, layout_fraction_heavy, initial_abstained, gate_abstained, gate_document_ocr_gap, gate_low_query_token_coverage, n_gate_reasons, diag_semantic, diag_word_level, diag_structural, diag_missing_text, n_shown_pages, n_shown_pages_ocr_gap, doc_page_count, doc_log_page_count, doc_n_gap_pages, doc_share_pages_empty_ocr, doc_pdf_kb_per_page, doc_chunks, doc_chunks_per_page, retrieved_any_page_ocr_gap, retrieved_adjacent_to_gap_page, retrieved_near_gap_page_le3, n_retrieved_chunks, n_distinct_retrieved_pages, retrieved_page_span, top_page_rel_position, top_chunk_chars, mean_chunk_chars, total_evidence_chars, top_bm25_score, top_dense_score, fused_top_minus_last, mean_fused_score, query_retrieval_tokens, initial_input_tokens, initial_output_tokens, initial_answer_chars, initial_answer_words, initial_answer_has_digit, question_words, question_chars, question_has_digit, question_has_year, question_has_money_or_pct, question_how_many_much, q_starts_what, q_starts_how, q_starts_which, q_starts_who, q_starts_when, q_starts_where, q_starts_yesno, doc_type_academic, doc_type_administration, doc_type_finance, doc_type_law, doc_type_manual, doc_type_news, doc_type_textbook.

Constant on flagged training questions (dropped from models): layout_excess_whitespace, layout_bare_numbers, layout_length_variance, retrieved_any_page_ocr_gap.

`retrieved_any_page_ocr_gap` is 0 for every flagged question in all samples: a page without OCR text yields no chunk, so it cannot be retrieved. Gap pages only reach `shown_pages` (which then forces the missing_text diagnosis).

## 3. Univariate: text_only vs image_only

AUC > 0.5 means higher values go with image_only. Rate diff = rate in image_only minus rate in text_only (flags only). Sorted by |AUC-0.5| on practice+validation. No multiple-comparison correction; with this many features a few intervals exclude 0.5 by chance.

n: practice+validation 40 text_only / 85 image_only; test 27 / 71; test docs not in validation 4 / 6.

| feature | train AUC [CI] | train rate diff [CI] | test AUC [CI] | test rate diff [CI] | test new-docs AUC | CI excludes 0.5 on both, same side |
|---|---|---|---|---|---|---|
| fused_top_minus_last | 0.685 [0.572, 0.789] |  | 0.680 [0.513, 0.813] |  | 0.833 | yes |
| doc_n_gap_pages | 0.316 [0.194, 0.461] |  | 0.298 [0.183, 0.449] |  | 0.479 | yes |
| top_fused_score | 0.673 [0.565, 0.775] |  | 0.683 [0.535, 0.818] |  | 0.771 | yes |
| gate_document_ocr_gap | 0.327 [0.217, 0.444] | -0.346 [-0.567, -0.113] | 0.331 [0.216, 0.470] | -0.338 [-0.567, -0.060] | 0.500 | yes |
| top_bm25_score | 0.671 [0.567, 0.765] |  | 0.720 [0.586, 0.847] |  | 0.729 | yes |
| doc_page_count | 0.343 [0.231, 0.485] |  | 0.367 [0.250, 0.523] |  | 0.562 |  |
| doc_log_page_count | 0.343 [0.231, 0.485] |  | 0.367 [0.250, 0.523] |  | 0.562 |  |
| retrieved_page_span | 0.345 [0.242, 0.488] |  | 0.398 [0.273, 0.563] |  | 0.500 |  |
| doc_chunks | 0.348 [0.227, 0.503] |  | 0.419 [0.283, 0.607] |  | 0.583 |  |
| fused_margin | 0.648 [0.544, 0.752] |  | 0.582 [0.433, 0.715] |  | 0.708 |  |
| doc_share_pages_empty_ocr | 0.356 [0.221, 0.504] |  | 0.338 [0.212, 0.484] |  | 0.438 |  |
| mean_chunk_chars | 0.626 [0.493, 0.749] |  | 0.491 [0.329, 0.657] |  | 0.500 |  |
| total_evidence_chars | 0.617 [0.489, 0.743] |  | 0.484 [0.327, 0.651] |  | 0.417 |  |
| doc_type_manual | 0.396 [0.293, 0.511] | -0.207 [-0.415, 0.022] | 0.403 [0.308, 0.490] | -0.194 [-0.383, -0.020] | 0.375 |  |
| question_has_year | 0.593 [0.488, 0.690] | 0.187 [-0.025, 0.380] | 0.549 [0.421, 0.668] | 0.098 [-0.158, 0.336] | 0.458 |  |
| top_chunk_chars | 0.591 [0.449, 0.731] |  | 0.534 [0.386, 0.682] |  | 0.417 |  |
| n_gate_reasons | 0.410 [0.307, 0.521] |  | 0.347 [0.244, 0.471] |  | 0.521 |  |
| question_chars | 0.589 [0.479, 0.714] |  | 0.405 [0.274, 0.547] |  | 0.667 |  |
| initial_abstained | 0.585 [0.509, 0.671] | 0.171 [0.018, 0.342] | 0.571 [0.502, 0.666] | 0.143 [0.004, 0.332] | 0.542 | yes |
| gate_abstained | 0.585 [0.509, 0.671] | 0.171 [0.018, 0.342] | 0.571 [0.502, 0.666] | 0.143 [0.004, 0.332] | 0.542 | yes |
| n_distinct_retrieved_pages | 0.415 [0.300, 0.546] |  | 0.522 [0.383, 0.678] |  | 0.562 |  |
| doc_chunks_per_page | 0.581 [0.455, 0.714] |  | 0.589 [0.429, 0.760] |  | 0.583 |  |
| initial_answer_chars | 0.419 [0.333, 0.494] |  | 0.427 [0.332, 0.497] |  | 0.438 | yes |
| doc_type_administration | 0.581 [0.491, 0.670] | 0.162 [-0.017, 0.340] | 0.574 [0.455, 0.682] | 0.148 [-0.090, 0.365] | 0.500 |  |
| initial_answer_words | 0.420 [0.335, 0.495] |  | 0.427 [0.332, 0.497] |  | 0.438 | yes |
| query_token_coverage | 0.579 [0.465, 0.697] |  | 0.623 [0.498, 0.744] |  | 0.500 |  |
| retrieved_near_gap_page_le3 | 0.424 [0.313, 0.533] | -0.153 [-0.374, 0.065] | 0.395 [0.288, 0.516] | -0.210 [-0.423, 0.032] | 0.417 |  |
| question_words | 0.576 [0.470, 0.698] |  | 0.387 [0.264, 0.523] |  | 0.667 |  |
| top_page_rel_position | 0.563 [0.426, 0.691] |  | 0.560 [0.417, 0.698] |  | 0.500 |  |
| doc_type_finance | 0.557 [0.448, 0.660] | 0.115 [-0.105, 0.319] | 0.462 [0.357, 0.570] | -0.076 [-0.285, 0.140] | 0.500 |  |
| question_has_digit | 0.557 [0.443, 0.665] | 0.113 [-0.115, 0.330] | 0.548 [0.432, 0.666] | 0.096 [-0.137, 0.332] | 0.375 |  |
| gate_low_query_token_coverage | 0.451 [0.349, 0.542] | -0.099 [-0.301, 0.083] | 0.394 [0.289, 0.500] | -0.212 [-0.421, 0.000] | 0.500 |  |
| n_shown_pages | 0.453 [0.382, 0.500] |  | 0.479 [0.450, 0.500] |  | 0.417 |  |
| initial_answer_has_digit | 0.454 [0.381, 0.524] | -0.093 [-0.238, 0.048] | 0.477 [0.415, 0.524] | -0.046 [-0.169, 0.048] | 0.583 |  |
| query_retrieval_tokens | 0.545 [0.436, 0.676] |  | 0.371 [0.251, 0.508] |  | 0.667 |  |
| weird_char_ratio | 0.461 [0.353, 0.573] |  | 0.599 [0.468, 0.720] |  | 0.688 |  |
| doc_type_law | 0.465 [0.381, 0.539] | -0.071 [-0.239, 0.077] | 0.464 [0.368, 0.551] | -0.073 [-0.264, 0.103] | 0.667 |  |
| doc_pdf_kb_per_page | 0.531 [0.376, 0.660] |  | 0.625 [0.472, 0.760] |  | 0.375 |  |
| initial_output_tokens | 0.474 [0.397, 0.549] |  | 0.470 [0.391, 0.546] |  | 0.688 |  |
| q_starts_how | 0.474 [0.409, 0.519] | -0.051 [-0.182, 0.038] | 0.503 [0.449, 0.541] | 0.005 [-0.101, 0.082] | 0.375 |  |
| initial_input_tokens | 0.526 [0.402, 0.642] |  | 0.451 [0.297, 0.622] |  | 0.625 |  |
| diag_word_level | 0.476 [0.406, 0.542] | -0.049 [-0.188, 0.084] | 0.543 [0.472, 0.614] | 0.086 [-0.057, 0.228] | 0.583 |  |
| question_has_money_or_pct | 0.524 [0.442, 0.608] | 0.049 [-0.116, 0.216] | 0.571 [0.487, 0.660] | 0.142 [-0.025, 0.320] | 0.667 |  |
| q_starts_what | 0.476 [0.362, 0.578] | -0.047 [-0.275, 0.156] | 0.518 [0.415, 0.616] | 0.036 [-0.170, 0.233] | 0.375 |  |
| mean_fused_score | 0.480 [0.364, 0.609] |  | 0.567 [0.424, 0.715] |  | 0.458 |  |
| layout_signal_count | 0.518 [0.500, 0.537] | 0.035 [0.000, 0.075] | 0.535 [0.500, 0.596] | 0.070 [0.000, 0.192] | 0.500 |  |
| layout_fraction_heavy | 0.518 [0.500, 0.537] | 0.035 [0.000, 0.075] | 0.535 [0.500, 0.596] | 0.070 [0.000, 0.192] | 0.500 |  |
| diag_structural | 0.518 [0.500, 0.537] | 0.035 [0.000, 0.075] | 0.535 [0.500, 0.596] | 0.070 [0.000, 0.192] | 0.500 |  |
| diag_missing_text | 0.517 [0.466, 0.579] | 0.034 [-0.069, 0.158] | 0.489 [0.442, 0.520] | -0.023 [-0.115, 0.039] | 0.500 |  |
| n_shown_pages_ocr_gap | 0.517 [0.466, 0.579] |  | 0.488 [0.441, 0.520] |  | 0.500 |  |
| doc_type_academic | 0.485 [0.405, 0.550] | -0.031 [-0.190, 0.100] | 0.613 [0.506, 0.711] | 0.227 [0.012, 0.421] | 0.458 |  |
| doc_type_textbook | 0.511 [0.481, 0.543] | 0.022 [-0.038, 0.085] | 0.463 [0.405, 0.500] | -0.074 [-0.191, 0.000] | 0.500 |  |
| diag_semantic | 0.490 [0.413, 0.576] | -0.021 [-0.175, 0.152] | 0.433 [0.339, 0.527] | -0.134 [-0.322, 0.055] | 0.417 |  |
| retrieved_adjacent_to_gap_page | 0.490 [0.419, 0.550] | -0.019 [-0.162, 0.100] | 0.448 [0.343, 0.555] | -0.104 [-0.314, 0.109] | 0.583 |  |
| question_how_many_much | 0.492 [0.424, 0.540] | -0.016 [-0.151, 0.081] | 0.503 [0.464, 0.532] | 0.005 [-0.071, 0.063] | 0.458 |  |
| top_dense_score | 0.507 [0.417, 0.602] |  | 0.483 [0.374, 0.586] |  | 0.667 |  |
| n_retrieved_chunks | 0.494 [0.481, 0.500] |  | 0.493 [0.475, 0.500] |  | 0.417 |  |
| q_starts_when | 0.506 [0.500, 0.520] | 0.012 [0.000, 0.040] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| doc_type_news | 0.505 [0.468, 0.540] | 0.010 [-0.065, 0.080] | 0.521 [0.500, 0.556] | 0.042 [0.000, 0.111] | 0.500 |  |
| q_starts_which | 0.504 [0.452, 0.553] | 0.007 [-0.096, 0.105] | 0.515 [0.439, 0.581] | 0.030 [-0.123, 0.162] | 0.500 |  |
| q_starts_yesno | 0.496 [0.429, 0.556] | -0.007 [-0.141, 0.112] | 0.487 [0.421, 0.549] | -0.027 [-0.158, 0.098] | 0.583 |  |
| q_starts_who | 0.499 [0.466, 0.527] | -0.001 [-0.068, 0.054] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| layout_excess_whitespace | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| layout_bare_numbers | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| layout_length_variance | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| layout_pipe_heavy | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| retrieved_any_page_ocr_gap | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |
| q_starts_where | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 0.500 |  |

## 4. Routers

Training set: 875 flagged practice+validation questions on 277 documents; binary target 'image needed' (image right, both text repairs wrong) has 85 positives; test has 71 of 706. Text repair EM on flagged training questions: {"retry_retrieval": 0.2217, "correct_text": 0.1611}; fallback text repair = **retry_retrieval**. 5-fold document-grouped CV; models refit on all training rows and applied once to test.

| model | CV AUC [CI] | test AUC [CI] | test new-docs AUC [CI] |
|---|---|---|---|
| logreg (logreg_C0.01) | 0.689 [0.594, 0.773] | 0.787 [0.715, 0.851] | 0.678 [0.426, 0.895] |
| tree (tree_d3) | 0.541 [0.445, 0.632] | 0.622 [0.533, 0.708] | 0.499 [0.195, 0.809] |
| gboost (gboost_d2) | 0.704 [0.624, 0.772] | 0.792 [0.718, 0.853] | 0.559 [0.211, 0.917] |

New-docs subset: 65 flagged questions, 6 positives. All CV candidates: {"logreg_C0.01": 0.689, "logreg_C0.1": 0.672, "logreg_C1": 0.665, "tree_d3": 0.541, "gboost_d2": 0.704}.

The abstention flag alone: AUC 0.638 on training, 0.665 on test. Image-needed questions among abstained / not abstained: training 74 / 11; test 68 of 467 / 3 of 239.

Same target, only questions whose initial answer abstained (signal beyond the abstention flag):

| model | CV n / pos | CV AUC [CI] | test n / pos | test AUC [CI] |
|---|---|---|---|---|
| logreg | 543 / 74 | 0.672 [0.586, 0.743] | 467 / 68 | 0.727 [0.632, 0.809] |
| tree | 543 / 74 | 0.538 [0.445, 0.615] | 467 / 68 | 0.574 [0.480, 0.666] |
| gboost | 543 / 74 | 0.676 [0.604, 0.740] | 467 / 68 | 0.730 [0.643, 0.807] |

Model score separating text_only from image_only (the supervisor's question):

| model | CV n / pos | CV AUC [CI] | test n / pos | test AUC [CI] |
|---|---|---|---|---|
| logreg | 125 / 85 | 0.671 [0.528, 0.794] | 98 / 71 | 0.785 [0.635, 0.911] |
| tree | 125 / 85 | 0.562 [0.427, 0.693] | 98 / 71 | 0.671 [0.534, 0.795] |
| gboost | 125 / 85 | 0.696 [0.586, 0.799] | 98 / 71 | 0.799 [0.675, 0.896] |

Largest standardised logistic coefficients: initial_abstained +0.18, gate_abstained +0.18, gate_document_ocr_gap -0.15, doc_log_page_count -0.12, query_retrieval_tokens -0.11, query_token_coverage +0.10, fused_margin +0.09, q_starts_how -0.09, gate_low_query_token_coverage -0.08, question_words -0.08, doc_type_academic -0.08, initial_answer_chars -0.08.

Largest boosting importances: fused_top_minus_last 0.10, query_retrieval_tokens 0.09, top_chunk_chars 0.07, query_token_coverage 0.07, retrieved_page_span 0.06, weird_char_ratio 0.05, total_evidence_chars 0.05, top_page_rel_position 0.05, fused_margin 0.05, mean_fused_score 0.04, doc_chunks 0.04, gate_abstained 0.03.

Depth-3 tree:

```
|--- doc_share_pages_empty_ocr <= 0.01
|   |--- query_token_coverage <= 0.73
|   |   |--- weird_char_ratio <= 0.02
|   |   |   |--- class: 0
|   |   |--- weird_char_ratio >  0.02
|   |   |   |--- class: 0
|   |--- query_token_coverage >  0.73
|   |   |--- fused_top_minus_last <= 0.31
|   |   |   |--- class: 0
|   |   |--- fused_top_minus_last >  0.31
|   |   |   |--- class: 0
|--- doc_share_pages_empty_ocr >  0.01
|   |--- top_chunk_chars <= 1311.50
|   |   |--- retrieved_page_span <= 140.50
|   |   |   |--- class: 0
|   |   |--- retrieved_page_span >  140.50
|   |   |   |--- class: 0
|   |--- top_chunk_chars >  1311.50
|   |   |--- class: 0
```

### References

From the test report: no_recovery 0.236 EM at 0.00171 USD; faar 0.266 EM at 0.0032 USD; gated_image 0.315 EM at 0.00831 USD; hindsight_best 0.346 EM at 0.00244 USD.

Recomputed here from case_scores EM and per-question `measured_cost` (baseline cost for every question plus the cost of the repair used; skipped repairs cost 0). Cost field null (run status execution_failed, counted as 0 USD; the question keeps its scored EM): {"practice": {"retry_retrieval": 0, "correct_text": 0, "invoke_vlm": 0}, "validation": {"retry_retrieval": 0, "correct_text": 0, "invoke_vlm": 3}, "test": {"retry_retrieval": 0, "correct_text": 0, "invoke_vlm": 0}}.

**test**

| policy | EM | USD/question | flagged to image |
|---|---|---|---|
| no_recovery | 0.2357 | 0.00171 | 0.000 |
| all_retry_retrieval | 0.2653 | 0.00298 | 0.000 |
| all_correct_text | 0.2283 | 0.00296 | 0.000 |
| gated_image | 0.3150 | 0.00831 | 1.000 |
| faar_diagnosis_route | 0.2664 | 0.00320 | 0.041 |
| hindsight_best | 0.3457 | 0.00244 | 0.101 |

**test_docs_not_in_validation**

| policy | EM | USD/question | flagged to image |
|---|---|---|---|
| no_recovery | 0.2893 | 0.00172 | 0.000 |
| all_retry_retrieval | 0.3636 | 0.00259 | 0.000 |
| all_correct_text | 0.2727 | 0.00264 | 0.000 |
| gated_image | 0.3802 | 0.00647 | 1.000 |
| faar_diagnosis_route | 0.3471 | 0.00278 | 0.046 |
| hindsight_best | 0.4132 | 0.00230 | 0.092 |

`faar_diagnosis_route` applies the diagnosis route to every flagged question; the reported FAAR keeps the initial answer for a few skipped repairs, hence tiny differences.

### Binary router sweeps

Thresholds are quantiles of the training out-of-fold probabilities (target share of flagged questions sent to image). 'random same share' is the expected EM of sending the same share of flagged questions to image at random; the gain and its interval are the router's EM minus that. 'picked' marks the threshold with the best out-of-fold training EM (chosen without test).

**logreg|fallback=retry_retrieval|test**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2653 | 0.00298 | 0.2653 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.118 | 0.2875 | 0.00365 | 0.2712 | +0.0164 [+0.0024, +0.0321] |  |
| 0.2 | 0.248 | 0.3055 | 0.00435 | 0.2776 | +0.0279 [+0.0120, +0.0447] |  |
| 0.3 | 0.326 | 0.3129 | 0.00480 | 0.2815 | +0.0314 [+0.0162, +0.0476] |  |
| 0.4 | 0.435 | 0.3214 | 0.00538 | 0.2869 | +0.0344 [+0.0204, +0.0498] |  |
| 0.5 | 0.528 | 0.3224 | 0.00587 | 0.2916 | +0.0308 [+0.0189, +0.0437] |  |
| 0.6 | 0.610 | 0.3214 | 0.00627 | 0.2957 | +0.0257 [+0.0149, +0.0369] |  |
| 0.7 | 0.686 | 0.3171 | 0.00666 | 0.2994 | +0.0177 [+0.0080, +0.0272] |  |
| 0.8 | 0.806 | 0.3161 | 0.00729 | 0.3054 | +0.0107 [+0.0039, +0.0175] |  |
| 0.9 | 0.905 | 0.3161 | 0.00781 | 0.3103 | +0.0058 [+0.0018, +0.0098] |  |
| 1.0 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] | picked |

**logreg|fallback=retry_retrieval|test_docs_not_in_validation**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.3636 | 0.00259 | 0.3636 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.123 | 0.3802 | 0.00306 | 0.3657 | +0.0145 [-0.0021, +0.0352] |  |
| 0.2 | 0.246 | 0.3884 | 0.00353 | 0.3677 | +0.0207 [-0.0019, +0.0463] |  |
| 0.3 | 0.338 | 0.3884 | 0.00391 | 0.3692 | +0.0192 [-0.0031, +0.0442] |  |
| 0.4 | 0.431 | 0.3884 | 0.00426 | 0.3708 | +0.0177 [-0.0134, +0.0504] |  |
| 0.5 | 0.508 | 0.3884 | 0.00457 | 0.3720 | +0.0164 [-0.0118, +0.0425] |  |
| 0.6 | 0.615 | 0.3884 | 0.00495 | 0.3738 | +0.0146 [-0.0133, +0.0399] |  |
| 0.7 | 0.677 | 0.3802 | 0.00521 | 0.3748 | +0.0053 [-0.0180, +0.0268] |  |
| 0.8 | 0.800 | 0.3719 | 0.00572 | 0.3769 | -0.0050 [-0.0243, +0.0105] |  |
| 0.9 | 0.908 | 0.3802 | 0.00613 | 0.3786 | +0.0015 [-0.0021, +0.0053] |  |
| 1.0 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] | picked |

**logreg|fallback=retry_retrieval|train_out_of_fold**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.3052 | 0.00283 | 0.3052 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.101 | 0.3203 | 0.00333 | 0.3098 | +0.0105 [+0.0001, +0.0234] |  |
| 0.2 | 0.200 | 0.3283 | 0.00384 | 0.3143 | +0.0140 [+0.0018, +0.0281] |  |
| 0.3 | 0.301 | 0.3362 | 0.00433 | 0.3189 | +0.0174 [+0.0055, +0.0310] |  |
| 0.4 | 0.400 | 0.3386 | 0.00482 | 0.3234 | +0.0153 [+0.0033, +0.0272] |  |
| 0.5 | 0.501 | 0.3410 | 0.00531 | 0.3279 | +0.0131 [+0.0012, +0.0241] |  |
| 0.6 | 0.600 | 0.3410 | 0.00578 | 0.3324 | +0.0086 [-0.0015, +0.0184] |  |
| 0.7 | 0.699 | 0.3426 | 0.00625 | 0.3369 | +0.0057 [-0.0032, +0.0142] |  |
| 0.8 | 0.800 | 0.3474 | 0.00673 | 0.3415 | +0.0059 [-0.0011, +0.0125] |  |
| 0.9 | 0.899 | 0.3474 | 0.00721 | 0.3460 | +0.0014 [-0.0037, +0.0059] |  |
| 1.0 | 1.000 | 0.3506 | 0.00768 | 0.3506 | +0.0000 [+0.0000, +0.0000] | picked |

**logreg|fallback=none|test**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2357 | 0.00171 | 0.2357 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.118 | 0.2664 | 0.00251 | 0.2450 | +0.0213 [+0.0076, +0.0366] |  |
| 0.2 | 0.248 | 0.2886 | 0.00337 | 0.2554 | +0.0332 [+0.0185, +0.0507] |  |
| 0.3 | 0.326 | 0.2960 | 0.00392 | 0.2616 | +0.0344 [+0.0192, +0.0511] |  |
| 0.4 | 0.435 | 0.3076 | 0.00465 | 0.2702 | +0.0374 [+0.0238, +0.0530] |  |
| 0.5 | 0.528 | 0.3118 | 0.00526 | 0.2776 | +0.0342 [+0.0229, +0.0470] |  |
| 0.6 | 0.610 | 0.3140 | 0.00577 | 0.2841 | +0.0298 [+0.0197, +0.0401] |  |
| 0.7 | 0.686 | 0.3161 | 0.00625 | 0.2901 | +0.0260 [+0.0175, +0.0351] |  |
| 0.8 | 0.806 | 0.3161 | 0.00704 | 0.2996 | +0.0164 [+0.0104, +0.0227] |  |
| 0.9 | 0.905 | 0.3161 | 0.00769 | 0.3075 | +0.0086 [+0.0048, +0.0130] |  |
| 1.0 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] | picked |

**logreg|fallback=none|test_docs_not_in_validation**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2893 | 0.00172 | 0.2893 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.123 | 0.3223 | 0.00229 | 0.3004 | +0.0219 [-0.0043, +0.0526] |  |
| 0.2 | 0.246 | 0.3388 | 0.00284 | 0.3116 | +0.0272 [-0.0016, +0.0590] |  |
| 0.3 | 0.338 | 0.3471 | 0.00331 | 0.3200 | +0.0271 [-0.0032, +0.0579] |  |
| 0.4 | 0.431 | 0.3554 | 0.00374 | 0.3284 | +0.0270 [-0.0024, +0.0586] |  |
| 0.5 | 0.508 | 0.3636 | 0.00412 | 0.3354 | +0.0282 [-0.0008, +0.0571] |  |
| 0.6 | 0.615 | 0.3719 | 0.00461 | 0.3452 | +0.0267 [-0.0000, +0.0541] |  |
| 0.7 | 0.677 | 0.3802 | 0.00491 | 0.3508 | +0.0294 [+0.0049, +0.0517] |  |
| 0.8 | 0.800 | 0.3719 | 0.00554 | 0.3620 | +0.0099 [-0.0090, +0.0252] |  |
| 0.9 | 0.908 | 0.3802 | 0.00604 | 0.3718 | +0.0084 [+0.0038, +0.0131] |  |
| 1.0 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] | picked |

**logreg|fallback=none|train_out_of_fold**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2703 | 0.00168 | 0.2703 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.101 | 0.2909 | 0.00230 | 0.2783 | +0.0126 [+0.0016, +0.0256] |  |
| 0.2 | 0.200 | 0.3029 | 0.00291 | 0.2863 | +0.0165 [+0.0049, +0.0303] |  |
| 0.3 | 0.301 | 0.3164 | 0.00352 | 0.2944 | +0.0220 [+0.0103, +0.0356] |  |
| 0.4 | 0.400 | 0.3267 | 0.00412 | 0.3024 | +0.0243 [+0.0123, +0.0361] |  |
| 0.5 | 0.501 | 0.3323 | 0.00473 | 0.3105 | +0.0218 [+0.0109, +0.0320] |  |
| 0.6 | 0.600 | 0.3370 | 0.00531 | 0.3184 | +0.0186 [+0.0092, +0.0273] |  |
| 0.7 | 0.699 | 0.3402 | 0.00590 | 0.3264 | +0.0138 [+0.0060, +0.0207] |  |
| 0.8 | 0.800 | 0.3474 | 0.00650 | 0.3345 | +0.0129 [+0.0077, +0.0183] |  |
| 0.9 | 0.899 | 0.3474 | 0.00708 | 0.3425 | +0.0049 [+0.0004, +0.0085] |  |
| 1.0 | 1.000 | 0.3506 | 0.00768 | 0.3506 | +0.0000 [+0.0000, +0.0000] | picked |

**tree|fallback=retry_retrieval|test**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2653 | 0.00298 | 0.2653 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.133 | 0.2770 | 0.00369 | 0.2719 | +0.0050 [-0.0033, +0.0148] |  |
| 0.2 | 0.263 | 0.2907 | 0.00442 | 0.2784 | +0.0123 [+0.0022, +0.0228] |  |
| 0.3 | 0.263 | 0.2907 | 0.00442 | 0.2784 | +0.0123 [+0.0022, +0.0228] |  |
| 0.4 | 0.483 | 0.3076 | 0.00564 | 0.2893 | +0.0183 [+0.0051, +0.0323] |  |
| 0.5 | 0.483 | 0.3076 | 0.00564 | 0.2893 | +0.0183 [+0.0051, +0.0323] |  |
| 0.6 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 0.7 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 0.8 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 0.9 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 1.0 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] | picked |

**tree|fallback=retry_retrieval|test_docs_not_in_validation**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.3636 | 0.00259 | 0.3636 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.185 | 0.3719 | 0.00333 | 0.3667 | +0.0052 [-0.0067, +0.0195] |  |
| 0.2 | 0.200 | 0.3719 | 0.00338 | 0.3669 | +0.0050 [-0.0072, +0.0197] |  |
| 0.3 | 0.200 | 0.3719 | 0.00338 | 0.3669 | +0.0050 [-0.0072, +0.0197] |  |
| 0.4 | 0.615 | 0.3719 | 0.00498 | 0.3738 | -0.0019 [-0.0249, +0.0180] |  |
| 0.5 | 0.615 | 0.3719 | 0.00498 | 0.3738 | -0.0019 [-0.0249, +0.0180] |  |
| 0.6 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 0.7 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 0.8 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 0.9 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 1.0 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] | picked |

**tree|fallback=retry_retrieval|train_out_of_fold**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.3052 | 0.00283 | 0.3052 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.111 | 0.3156 | 0.00337 | 0.3103 | +0.0053 [-0.0010, +0.0130] |  |
| 0.2 | 0.203 | 0.3219 | 0.00383 | 0.3145 | +0.0075 [-0.0017, +0.0180] |  |
| 0.3 | 0.333 | 0.3291 | 0.00446 | 0.3203 | +0.0088 [-0.0029, +0.0225] |  |
| 0.4 | 0.435 | 0.3299 | 0.00496 | 0.3250 | +0.0049 [-0.0064, +0.0178] |  |
| 0.5 | 0.503 | 0.3299 | 0.00529 | 0.3280 | +0.0019 [-0.0093, +0.0149] |  |
| 0.6 | 0.663 | 0.3347 | 0.00600 | 0.3353 | -0.0006 [-0.0118, +0.0119] |  |
| 0.7 | 0.761 | 0.3355 | 0.00648 | 0.3397 | -0.0043 [-0.0156, +0.0070] |  |
| 0.8 | 0.882 | 0.3442 | 0.00709 | 0.3452 | -0.0010 [-0.0071, +0.0053] |  |
| 0.9 | 0.910 | 0.3450 | 0.00722 | 0.3465 | -0.0015 [-0.0067, +0.0041] |  |
| 1.0 | 1.000 | 0.3506 | 0.00768 | 0.3506 | +0.0000 [+0.0000, +0.0000] | picked |

**tree|fallback=none|test**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2357 | 0.00171 | 0.2357 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.133 | 0.2526 | 0.00260 | 0.2463 | +0.0064 [-0.0020, +0.0162] |  |
| 0.2 | 0.263 | 0.2706 | 0.00349 | 0.2566 | +0.0140 [+0.0038, +0.0251] |  |
| 0.3 | 0.263 | 0.2706 | 0.00349 | 0.2566 | +0.0140 [+0.0038, +0.0251] |  |
| 0.4 | 0.483 | 0.2928 | 0.00499 | 0.2740 | +0.0188 [+0.0051, +0.0334] |  |
| 0.5 | 0.483 | 0.2928 | 0.00499 | 0.2740 | +0.0188 [+0.0051, +0.0334] |  |
| 0.6 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 0.7 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 0.8 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 0.9 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] |  |
| 1.0 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] | picked |

**tree|fallback=none|test_docs_not_in_validation**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2893 | 0.00172 | 0.2893 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.185 | 0.3223 | 0.00264 | 0.3060 | +0.0163 [-0.0092, +0.0465] |  |
| 0.2 | 0.200 | 0.3306 | 0.00271 | 0.3074 | +0.0231 [-0.0048, +0.0555] |  |
| 0.3 | 0.200 | 0.3306 | 0.00271 | 0.3074 | +0.0231 [-0.0048, +0.0555] |  |
| 0.4 | 0.615 | 0.3471 | 0.00463 | 0.3452 | +0.0019 [-0.0280, +0.0281] |  |
| 0.5 | 0.615 | 0.3471 | 0.00463 | 0.3452 | +0.0019 [-0.0280, +0.0281] |  |
| 0.6 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 0.7 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 0.8 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 0.9 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] |  |
| 1.0 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] | picked |

**tree|fallback=none|train_out_of_fold**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2703 | 0.00168 | 0.2703 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.111 | 0.2846 | 0.00236 | 0.2792 | +0.0054 [-0.0006, +0.0131] |  |
| 0.2 | 0.203 | 0.2957 | 0.00292 | 0.2866 | +0.0091 [+0.0005, +0.0200] |  |
| 0.3 | 0.333 | 0.3092 | 0.00370 | 0.2970 | +0.0122 [+0.0011, +0.0256] |  |
| 0.4 | 0.435 | 0.3132 | 0.00431 | 0.3052 | +0.0080 [-0.0032, +0.0207] |  |
| 0.5 | 0.503 | 0.3164 | 0.00471 | 0.3106 | +0.0057 [-0.0049, +0.0184] |  |
| 0.6 | 0.663 | 0.3267 | 0.00562 | 0.3235 | +0.0032 [-0.0081, +0.0153] |  |
| 0.7 | 0.761 | 0.3323 | 0.00622 | 0.3314 | +0.0009 [-0.0104, +0.0119] |  |
| 0.8 | 0.882 | 0.3426 | 0.00696 | 0.3411 | +0.0015 [-0.0047, +0.0075] |  |
| 0.9 | 0.910 | 0.3434 | 0.00711 | 0.3433 | +0.0001 [-0.0054, +0.0052] |  |
| 1.0 | 1.000 | 0.3506 | 0.00768 | 0.3506 | +0.0000 [+0.0000, +0.0000] | picked |

**gboost|fallback=retry_retrieval|test**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2653 | 0.00298 | 0.2653 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.095 | 0.2907 | 0.00350 | 0.2700 | +0.0207 [+0.0078, +0.0349] |  |
| 0.2 | 0.188 | 0.3023 | 0.00401 | 0.2747 | +0.0276 [+0.0121, +0.0442] |  |
| 0.3 | 0.309 | 0.3129 | 0.00467 | 0.2807 | +0.0322 [+0.0157, +0.0495] |  |
| 0.4 | 0.425 | 0.3161 | 0.00529 | 0.2864 | +0.0296 [+0.0152, +0.0444] |  |
| 0.5 | 0.513 | 0.3192 | 0.00575 | 0.2908 | +0.0284 [+0.0163, +0.0416] |  |
| 0.6 | 0.622 | 0.3182 | 0.00632 | 0.2962 | +0.0220 [+0.0118, +0.0325] |  |
| 0.7 | 0.729 | 0.3182 | 0.00689 | 0.3016 | +0.0166 [+0.0087, +0.0240] |  |
| 0.8 | 0.824 | 0.3203 | 0.00739 | 0.3063 | +0.0140 [+0.0085, +0.0199] |  |
| 0.9 | 0.916 | 0.3182 | 0.00785 | 0.3109 | +0.0073 [+0.0041, +0.0109] |  |
| 1.0 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] | picked |

**gboost|fallback=retry_retrieval|test_docs_not_in_validation**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.3636 | 0.00259 | 0.3636 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.169 | 0.3884 | 0.00321 | 0.3664 | +0.0220 [-0.0002, +0.0483] |  |
| 0.2 | 0.215 | 0.3884 | 0.00338 | 0.3672 | +0.0212 [-0.0015, +0.0469] |  |
| 0.3 | 0.292 | 0.3884 | 0.00370 | 0.3685 | +0.0200 [-0.0023, +0.0451] |  |
| 0.4 | 0.400 | 0.3802 | 0.00413 | 0.3702 | +0.0099 [-0.0202, +0.0397] |  |
| 0.5 | 0.523 | 0.3802 | 0.00460 | 0.3723 | +0.0079 [-0.0231, +0.0385] |  |
| 0.6 | 0.631 | 0.3636 | 0.00503 | 0.3741 | -0.0104 [-0.0379, +0.0129] |  |
| 0.7 | 0.723 | 0.3554 | 0.00540 | 0.3756 | -0.0202 [-0.0460, +0.0010] |  |
| 0.8 | 0.846 | 0.3719 | 0.00589 | 0.3776 | -0.0057 [-0.0234, +0.0053] |  |
| 0.9 | 0.938 | 0.3802 | 0.00623 | 0.3791 | +0.0010 [-0.0014, +0.0035] |  |
| 1.0 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] | picked |

**gboost|fallback=retry_retrieval|train_out_of_fold**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.3052 | 0.00283 | 0.3052 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.101 | 0.3211 | 0.00331 | 0.3098 | +0.0113 [+0.0013, +0.0240] |  |
| 0.2 | 0.200 | 0.3315 | 0.00382 | 0.3143 | +0.0172 [+0.0067, +0.0305] |  |
| 0.3 | 0.301 | 0.3323 | 0.00432 | 0.3189 | +0.0134 [+0.0023, +0.0264] |  |
| 0.4 | 0.400 | 0.3402 | 0.00480 | 0.3234 | +0.0169 [+0.0057, +0.0292] |  |
| 0.5 | 0.501 | 0.3450 | 0.00530 | 0.3279 | +0.0171 [+0.0061, +0.0289] |  |
| 0.6 | 0.600 | 0.3426 | 0.00578 | 0.3324 | +0.0102 [+0.0002, +0.0209] |  |
| 0.7 | 0.699 | 0.3402 | 0.00626 | 0.3369 | +0.0033 [-0.0052, +0.0116] |  |
| 0.8 | 0.800 | 0.3450 | 0.00673 | 0.3415 | +0.0035 [-0.0040, +0.0102] |  |
| 0.9 | 0.899 | 0.3498 | 0.00719 | 0.3460 | +0.0038 [+0.0003, +0.0073] |  |
| 1.0 | 1.000 | 0.3506 | 0.00768 | 0.3506 | +0.0000 [+0.0000, +0.0000] | picked |

**gboost|fallback=none|test**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2357 | 0.00171 | 0.2357 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.095 | 0.2664 | 0.00234 | 0.2433 | +0.0231 [+0.0103, +0.0380] |  |
| 0.2 | 0.188 | 0.2812 | 0.00297 | 0.2507 | +0.0305 [+0.0150, +0.0470] |  |
| 0.3 | 0.309 | 0.2981 | 0.00379 | 0.2602 | +0.0379 [+0.0219, +0.0553] |  |
| 0.4 | 0.425 | 0.3066 | 0.00456 | 0.2694 | +0.0371 [+0.0230, +0.0514] |  |
| 0.5 | 0.513 | 0.3108 | 0.00513 | 0.2764 | +0.0344 [+0.0228, +0.0459] |  |
| 0.6 | 0.622 | 0.3140 | 0.00584 | 0.2850 | +0.0289 [+0.0181, +0.0390] |  |
| 0.7 | 0.729 | 0.3129 | 0.00654 | 0.2936 | +0.0193 [+0.0117, +0.0269] |  |
| 0.8 | 0.824 | 0.3150 | 0.00716 | 0.3011 | +0.0139 [+0.0079, +0.0196] |  |
| 0.9 | 0.916 | 0.3150 | 0.00774 | 0.3084 | +0.0066 [+0.0031, +0.0099] |  |
| 1.0 | 1.000 | 0.3150 | 0.00831 | 0.3150 | +0.0000 [+0.0000, +0.0000] | picked |

**gboost|fallback=none|test_docs_not_in_validation**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2893 | 0.00172 | 0.2893 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.169 | 0.3306 | 0.00248 | 0.3046 | +0.0259 [-0.0016, +0.0584] |  |
| 0.2 | 0.215 | 0.3306 | 0.00271 | 0.3088 | +0.0217 [-0.0060, +0.0534] |  |
| 0.3 | 0.292 | 0.3471 | 0.00311 | 0.3158 | +0.0313 [+0.0007, +0.0635] |  |
| 0.4 | 0.400 | 0.3554 | 0.00364 | 0.3256 | +0.0298 [-0.0000, +0.0597] |  |
| 0.5 | 0.523 | 0.3554 | 0.00419 | 0.3368 | +0.0186 [-0.0103, +0.0470] |  |
| 0.6 | 0.631 | 0.3554 | 0.00471 | 0.3466 | +0.0088 [-0.0204, +0.0364] |  |
| 0.7 | 0.723 | 0.3471 | 0.00517 | 0.3550 | -0.0079 [-0.0369, +0.0162] |  |
| 0.8 | 0.846 | 0.3636 | 0.00575 | 0.3662 | -0.0025 [-0.0257, +0.0149] |  |
| 0.9 | 0.938 | 0.3802 | 0.00618 | 0.3746 | +0.0056 [+0.0026, +0.0087] |  |
| 1.0 | 1.000 | 0.3802 | 0.00647 | 0.3802 | +0.0000 [+0.0000, +0.0000] | picked |

**gboost|fallback=none|train_out_of_fold**

| train share | test share to image | EM | USD/question | random same share EM | gain vs random [CI] | picked |
|---|---|---|---|---|---|---|
| 0.0 | 0.000 | 0.2703 | 0.00168 | 0.2703 | +0.0000 [+0.0000, +0.0000] |  |
| 0.1 | 0.101 | 0.2917 | 0.00228 | 0.2783 | +0.0134 [+0.0025, +0.0259] |  |
| 0.2 | 0.200 | 0.3060 | 0.00290 | 0.2863 | +0.0197 [+0.0088, +0.0326] |  |
| 0.3 | 0.301 | 0.3148 | 0.00352 | 0.2944 | +0.0204 [+0.0104, +0.0321] |  |
| 0.4 | 0.400 | 0.3283 | 0.00412 | 0.3024 | +0.0259 [+0.0157, +0.0370] |  |
| 0.5 | 0.501 | 0.3339 | 0.00473 | 0.3105 | +0.0234 [+0.0141, +0.0332] |  |
| 0.6 | 0.600 | 0.3355 | 0.00533 | 0.3184 | +0.0170 [+0.0086, +0.0253] |  |
| 0.7 | 0.699 | 0.3386 | 0.00591 | 0.3264 | +0.0122 [+0.0043, +0.0198] |  |
| 0.8 | 0.800 | 0.3442 | 0.00650 | 0.3345 | +0.0097 [+0.0035, +0.0152] |  |
| 0.9 | 0.899 | 0.3490 | 0.00708 | 0.3425 | +0.0065 [+0.0028, +0.0097] |  |
| 1.0 | 1.000 | 0.3506 | 0.00768 | 0.3506 | +0.0000 [+0.0000, +0.0000] | picked |

### Three-way choice (no repair / text repair / image)

One logistic model per action for P(EM=1); choose argmax of p minus lambda x mean incremental cost ({"none": 0.0, "retry_retrieval": 0.00165, "invoke_vlm": 0.00863} USD). Per-action AUC (CV / test): none 0.937 / 0.945; retry_retrieval 0.821 / 0.840; invoke_vlm 0.792 / 0.812.

**test**

| lambda (per USD) | EM | USD/question | to image | to text | no repair | train OOF EM (flagged) |
|---|---|---|---|---|---|---|
| 0 | 0.3192 | 0.00712 | 0.788 | 0.153 | 0.059 | 0.2811 |
| 2 | 0.3192 | 0.00666 | 0.704 | 0.212 | 0.084 | 0.2834 |
| 5 | 0.3192 | 0.00593 | 0.571 | 0.309 | 0.120 | 0.2789 |
| 10 | 0.3140 | 0.00491 | 0.391 | 0.459 | 0.150 | 0.2697 |
| 15 | 0.3002 | 0.00398 | 0.229 | 0.565 | 0.205 | 0.2663 |
| 20 | 0.2907 | 0.00338 | 0.139 | 0.574 | 0.288 | 0.2480 |
| 30 | 0.2727 | 0.00257 | 0.045 | 0.442 | 0.513 | 0.2240 |
| 50 | 0.2452 | 0.00197 | 0.003 | 0.193 | 0.805 | 0.1920 |
| 100 | 0.2378 | 0.00177 | 0.000 | 0.041 | 0.959 | 0.1771 |

**test_docs_not_in_validation**

| lambda (per USD) | EM | USD/question | to image | to text | no repair | train OOF EM (flagged) |
|---|---|---|---|---|---|---|
| 0 | 0.3884 | 0.00571 | 0.831 | 0.108 | 0.062 | 0.2811 |
| 2 | 0.3802 | 0.00536 | 0.738 | 0.138 | 0.123 | 0.2834 |
| 5 | 0.3884 | 0.00479 | 0.600 | 0.231 | 0.169 | 0.2789 |
| 10 | 0.3802 | 0.00407 | 0.431 | 0.385 | 0.185 | 0.2697 |
| 15 | 0.3802 | 0.00353 | 0.308 | 0.446 | 0.246 | 0.2663 |
| 20 | 0.3719 | 0.00319 | 0.215 | 0.538 | 0.246 | 0.2480 |
| 30 | 0.3554 | 0.00268 | 0.108 | 0.508 | 0.385 | 0.2240 |
| 50 | 0.3306 | 0.00198 | 0.000 | 0.277 | 0.723 | 0.1920 |
| 100 | 0.3058 | 0.00183 | 0.000 | 0.092 | 0.908 | 0.1771 |

### Fixed rules (not tuned)

The cascade uses the text repair's own abstention, which is a runtime signal but costs the text repair first.

**train_pooled**

| rule | EM | USD/question | flagged to image | flagged running text repair | EM minus gated image [CI] |
|---|---|---|---|---|---|
| image if initial abstained else no repair | 0.3434 | 0.00548 | 0.621 | 0.000 | -0.0072 [-0.0154, +0.0008] |
| image if initial abstained else retry_retrieval | 0.3426 | 0.00589 | 0.621 | 0.379 | -0.0079 [-0.0172, +0.0009] |
| retry_retrieval if initial abstained else no repair | 0.3060 | 0.00241 | 0.000 | 0.621 | -0.0445 [-0.0684, -0.0208] |
| cascade: retry_retrieval, then image if it abstains | 0.3672 | 0.00609 | 0.536 | 1.000 | +0.0167 [+0.0033, +0.0305] |
| cascade only if initial abstained: retry_retrieval, then image if it abstains | 0.3641 | 0.00543 | 0.488 | 0.621 | +0.0135 [+0.0000, +0.0277] |

**test**

| rule | EM | USD/question | flagged to image | flagged running text repair | EM minus gated image [CI] |
|---|---|---|---|---|---|
| image if initial abstained else no repair | 0.3161 | 0.00612 | 0.661 | 0.000 | +0.0011 [-0.0064, +0.0085] |
| image if initial abstained else retry_retrieval | 0.3140 | 0.00656 | 0.661 | 0.339 | -0.0011 [-0.0078, +0.0060] |
| retry_retrieval if initial abstained else no repair | 0.2674 | 0.00255 | 0.000 | 0.661 | -0.0476 [-0.0768, -0.0219] |
| cascade: retry_retrieval, then image if it abstains | 0.3383 | 0.00687 | 0.584 | 1.000 | +0.0233 [+0.0111, +0.0350] |
| cascade only if initial abstained: retry_retrieval, then image if it abstains | 0.3372 | 0.00626 | 0.554 | 0.661 | +0.0222 [+0.0093, +0.0352] |

**test_docs_not_in_validation**

| rule | EM | USD/question | flagged to image | flagged running text repair | EM minus gated image [CI] |
|---|---|---|---|---|---|
| image if initial abstained else no repair | 0.3636 | 0.00395 | 0.462 | 0.000 | -0.0165 [-0.0504, +0.0161] |
| image if initial abstained else retry_retrieval | 0.3802 | 0.00440 | 0.462 | 0.538 | +0.0000 [-0.0250, +0.0246] |
| retry_retrieval if initial abstained else no repair | 0.3471 | 0.00214 | 0.000 | 0.462 | -0.0331 [-0.0790, +0.0088] |
| cascade: retry_retrieval, then image if it abstains | 0.4050 | 0.00420 | 0.354 | 1.000 | +0.0248 [-0.0091, +0.0597] |
| cascade only if initial abstained: retry_retrieval, then image if it abstains | 0.3884 | 0.00334 | 0.246 | 0.462 | +0.0083 [-0.0360, +0.0486] |

## 5. Gate check

fixed = initial wrong and the repair right; broken = initial right and the repair wrong. 'any' = at least one of the three repairs.

**test**

| gate reasons | n | initial EM | initial right | fixed by any | broken by any | broken by all | fixed retry/correct/image | broken retry/correct/image |
|---|---|---|---|---|---|---|---|---|
| abstained | 127 | 0.008 | 1 | 35 | 0 | 0 | 8/0/33 | 0/0/0 |
| abstained+document_ocr_gap | 102 | 0.000 | 0 | 23 | 0 | 0 | 8/0/15 | 0/0/0 |
| abstained+document_ocr_gap+low_query_token_coverage | 119 | 0.000 | 0 | 17 | 0 | 0 | 8/0/9 | 0/0/0 |
| abstained+low_query_token_coverage | 119 | 0.000 | 0 | 23 | 0 | 0 | 6/0/19 | 0/0/0 |
| document_ocr_gap | 163 | 0.503 | 82 | 3 | 5 | 2 | 1/0/3 | 3/3/3 |
| document_ocr_gap+low_query_token_coverage | 28 | 0.357 | 10 | 0 | 2 | 0 | 0/0/0 | 0/1/2 |
| low_query_token_coverage | 48 | 0.333 | 16 | 3 | 7 | 0 | 2/1/3 | 2/4/2 |
| ALL FLAGGED | 706 | 0.154 | 109 | 104 | 14 | 2 | 33/1/82 | 5/8/7 |

**validation**

| gate reasons | n | initial EM | initial right | fixed by any | broken by any | broken by all | fixed retry/correct/image | broken retry/correct/image |
|---|---|---|---|---|---|---|---|---|
| abstained | 104 | 0.000 | 0 | 30 | 0 | 0 | 5/0/28 | 0/0/0 |
| abstained+document_ocr_gap | 99 | 0.000 | 0 | 18 | 0 | 0 | 9/1/11 | 0/0/0 |
| abstained+document_ocr_gap+low_query_token_coverage | 142 | 0.000 | 0 | 17 | 0 | 0 | 12/0/7 | 0/0/0 |
| abstained+low_query_token_coverage | 95 | 0.000 | 0 | 21 | 0 | 0 | 6/1/17 | 0/0/0 |
| document_ocr_gap | 149 | 0.584 | 87 | 8 | 16 | 0 | 5/2/4 | 7/8/4 |
| document_ocr_gap+low_query_token_coverage | 22 | 0.500 | 11 | 2 | 2 | 0 | 1/0/1 | 1/1/1 |
| low_query_token_coverage | 36 | 0.361 | 13 | 3 | 4 | 0 | 2/1/2 | 1/2/1 |
| ALL FLAGGED | 647 | 0.172 | 111 | 99 | 22 | 0 | 40/5/70 | 9/11/6 |

**practice**

| gate reasons | n | initial EM | initial right | fixed by any | broken by any | broken by all | fixed retry/correct/image | broken retry/correct/image |
|---|---|---|---|---|---|---|---|---|
| abstained | 30 | 0.000 | 0 | 15 | 0 | 0 | 7/0/11 | 0/0/0 |
| abstained+document_ocr_gap | 13 | 0.000 | 0 | 5 | 0 | 0 | 0/0/5 | 0/0/0 |
| abstained+document_ocr_gap+low_query_token_coverage | 20 | 0.000 | 0 | 4 | 0 | 0 | 2/0/4 | 0/0/0 |
| abstained+low_query_token_coverage | 40 | 0.000 | 0 | 10 | 0 | 0 | 4/0/9 | 0/0/0 |
| document_ocr_gap | 62 | 0.355 | 22 | 5 | 5 | 0 | 0/0/5 | 1/4/0 |
| document_ocr_gap+low_query_token_coverage | 15 | 0.200 | 3 | 0 | 0 | 0 | 0/0/0 | 0/0/0 |
| low_query_token_coverage | 48 | 0.292 | 14 | 3 | 0 | 0 | 1/1/3 | 0/0/0 |
| ALL FLAGGED | 228 | 0.171 | 39 | 42 | 5 | 0 | 14/1/37 | 1/4/0 |

Flagged questions diagnosed semantic, by where the OCR gap is:

| sample | flagged semantic | gap on a retrieved/shown page | gap only elsewhere in document | of those: a retrieved page is next to a gap page | no gap in document |
|---|---|---|---|---|---|
| test | 541 | 0 | 319 | 125 | 222 |
| validation | 492 | 0 | 324 | 106 | 168 |
| practice | 161 | 0 | 81 | 32 | 80 |

The zero is by construction: a gap on a shown page makes the diagnosis missing_text, never semantic.

## 6. Verdict

Measured:

- A signal exists but it is modest. For 'image needed' the document-grouped CV AUC is 0.689 (logistic) and 0.704 (boosting); test AUC is 0.787 and 0.792, but 252 of 327 test documents are also validation documents, and on the 75 new documents (65 flagged, 6 positives) the intervals are too wide to say anything (0.678 and 0.559). The CV figure is the document-independent one.
- Separating text_only from image_only: model score CV AUC 0.671 / 0.696; the best single features sit at 0.67-0.69 (score gap between the first and last retrieved chunk, top fused score, top BM25 score: higher goes with image_only; a document with OCR gaps goes with text_only). No layout, weird-character, question-form or doc-type feature holds on both samples.
- Much of the signal is the abstention flag: on test 68 of 71 image-needed questions had an abstained initial answer. Within abstained questions the models still rank above chance (CV 0.672 / 0.676).
- Learned router on test, sending the top ~40% (by training quantile) to image and the rest to retry_retrieval: 0.3214 EM at 0.00538 USD (vs random at the same image share +0.0344 [+0.0204, +0.0498]). Out-of-fold on training at the same share: 0.3386 EM at 0.00482 USD (vs random at the same image share +0.0153 [+0.0033, +0.0272]). No operating point comes near hindsight best (0.346 at 0.00244).
- Untuned rule 'image only if the initial answer abstained, otherwise no repair': 0.3161 EM at 0.00612 USD on test, i.e. gated-image EM at about three quarters of its cost.
- Untuned cascade 'retry_retrieval, then image if the retry abstains': 0.3383 EM at 0.00687 USD on test, +0.0233 [+0.0111, +0.0350] EM against gated image; on practice+validation +0.0167 [+0.0033, +0.0305]; on the 75 new test documents +0.0248 [-0.0091, +0.0597].
- Gate: the 239 test questions flagged without an abstention had 108 right initially; a repair fixed 6 and at least one repair broke 14. For document_ocr_gap alone: n=163, initial EM 0.503, fixed 3, broken 5.

Interpretation:

- A learned router is a cost saver, not an accuracy gain: it can roughly hold gated-image EM for about two thirds of the cost, and most of that comes from one flag (did the first answer abstain). The features on file do not identify the text_only cases well enough to approach the hindsight policy.
- The two cheap rules do as well as or better than the learned models. The cascade is the only policy here that beats gated image on EM, because the text repair's own abstention is a stronger signal than anything available before a repair is run.
- The document_ocr_gap and low-coverage gate reasons flag questions that repairs almost never fix; restricting the gate to abstentions would cut cost with no measurable EM loss on these samples.
- All of this is post hoc on samples already used for the study. The rules were written down before their test numbers were computed, but they were not pre-registered, and the test sample shares documents with validation. A fresh document-disjoint sample is needed before any of it is claimed.
