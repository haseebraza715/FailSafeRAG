# Failure codebook v1 (frozen 2026-10-31)

Scope: the 55 questions of `ohr_dev_v1` that scored official EM 0 in run
`2026-10-31-ohr-dev-v1-baseline-terra-r1` (gpt-6.1-terra, no recovery). The
15 exact matches are not coded.

Version 1 was drafted after a first read of all 55 cases, then fixed before
any case was coded with it. A later change makes v2 and a recoding; v1 and its
codes stay in the record.

**Who coded.** The first coding is by an AI agent (Claude Opus 5.5) working
from the page image, the MinerU text, the clean reference text, the prompt
sent, the raw reply, the gold answer and the evidence pages. It is not a human
annotation and not independent validation (study brief section 10). Agreement
with a human labeller measures agent-human agreement unless a person adopts or
replaces these codes first.

Each case gets four observations, kept apart as the study brief asks
(section 3), plus one primary cause for the breakdown.

## 1. Text defect (page level, before retrieval; several values allowed)

Compare the page image with the MinerU text of each gold evidence page. Do
not look at the question or the gold answer for this step.

| Code | Meaning |
| --- | --- |
| `word_corruption` | Characters or words are wrong: misread letters, lost spaces, garbled fractions or Han characters, invented text (for example LaTeX formulas on a handwritten page) |
| `lost_structure` | Text is present but its structure is lost or misleading: table cells flattened or misaligned, reading order broken, text moved to another page |
| `missing_content` | Text that the image shows is absent from MinerU: an empty page or a page with only part of its text |
| `adequate` | No defect that a reader would notice. Formatting differences (LaTeX spacing, `$` around numbers) do not count |
| `uncertain` | Cannot decide; explain in the note |

## 2. Evidence impact (case level, needs the question and gold answer)

Does the passage the question needs survive in the MinerU text of the evidence page(s)?

| Code | Meaning |
| --- | --- |
| `survives` | The needed passage is readable and correct |
| `damaged` | The passage is there but altered in a way that can change the answer (a wrong number, a garbled name) |
| `lost` | The passage is absent or unreadable |
| `uncertain` | Cannot decide |

## 3. Retrieval (from the run record, not a judgement)

| Field | Meaning |
| --- | --- |
| `page_hit` | A retrieved chunk comes from a gold evidence page (all pages of a multi-page case: `all`, some: `partial`, none: `none`, nothing sent: `not_sent`) |
| `passage_in_prompt` | The coder's judgement that the needed passage is inside the prompt: `yes`, `partial`, `no`, `not_sent`. Shown next to the measured OHR-Bench retrieval LCS, which is the quantitative version |

## 4. Answer and gold (needs the reply, gold answer and evidence)

| Code | Meaning |
| --- | --- |
| `answer_correct` | The reply states the gold fact, possibly with extra correct context or another format. EM is 0 only because of wording, units, list format, notation or Unicode |
| `answer_partial` | The reply has part of the gold content and nothing wrong |
| `answer_wrong` | The reply states something else |
| `abstained` | The model replied `NO_ANSWER`, or no request was sent (`no_evidence`) |

Gold flags (several allowed, empty when none):

| Flag | Meaning |
| --- | --- |
| `gold_questionable` | The gold answer is incomplete, over-inclusive, vague, or disagrees with the page image |
| `question_ambiguous` | The question admits more than one reasonable answer from the document |
| `gt_transcription_error` | The question or gold copies an error in OHR-Bench's clean reference text that the page image does not have |

## 5. Primary cause (one per case, decided in this order)

1. `scoring_artifact`: `answer_correct`. The question was answered; the official EM cannot see it.
2. `gold_or_question`: `gold_questionable` or `question_ambiguous`, and the reply is a defensible answer to the question as asked. The case cannot count as a model or OCR failure.
3. `ocr_lost`: evidence impact `lost`.
4. `ocr_damaged`: evidence impact `damaged`, and the damage plausibly caused the wrong or missing answer.
5. `retrieval_miss`: evidence `survives`, but `passage_in_prompt` is `no` or `partial` and that explains the reply.
6. `model_error`: the needed passage is in the prompt and intact, and the reply is wrong, partial or an abstention.

The order follows the pipeline after two checks that the case is a real
failure at all. A case with lost evidence and a questionable gold answer is
`ocr_lost` when the reply is an abstention, because no reply was given to
defend.

## 6. Repair a case points to (interpretation, not measured)

`text_repair` (better OCR text of the page would fix it), `retrieval_repair`
(a better chunk choice would fix it), `visual_repair` (reading the page image
would fix it), `answer_format` (scoring or answer-format change only), `none`
(gold or question problem, or a model reasoning error that no evidence repair
targets). Several allowed. The headroom runs measure what these guesses only
suggest.
