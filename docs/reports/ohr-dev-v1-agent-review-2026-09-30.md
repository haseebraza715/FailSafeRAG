# AGENT REVIEW: the 20 inspection cases of ohr_dev_v1 (2026-09-30)

## Corrections (2026-09-30, later)

This note lists what changed after the first version of this review. It changes no frozen question, gold answer, pilot selection or annotation.

- **Retrieval count.** The summary said the sent evidence holds the answer fully in 15 of the 17 sent cases. The table rows give 13 fully (4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 18, 19), 2 partly (7, 14) and 2 not at all (17, 20). The summary now says 13. `scripts/audits/case_review_counts.py` derives the counts from [the structured file](ohr-dev-v1-agent-review-2026-09-30.cases.json), and its `--check-markdown` option compares every row with the table below.
- **Structured data.** The same 20 rows are committed as `ohr-dev-v1-agent-review-2026-09-30.cases.json`. It carries the header "AGENT REVIEW" and marks itself as neither human annotation nor ground truth. Its notes paraphrase the uncertainty bullets and quote no document text beyond single words.
- **Wording.** Five interpretations were reworded, and each change is in the text below:
  - noisy OCR text that exists is not shown to be correct, and the `ok` status only says text exists;
  - a gold page among the retrieved pages does not show that the needed passage was retrieved (case 20);
  - evidence missing from the first prompt does not mean a case tests only abstention, because it also bears on later retrieval or OCR repair;
  - agent concerns about gold answers are provisional and not adjudicated corrections;
  - the 20 cases are a purposive diagnostic sample and give no estimate of dataset-wide error rates.

**This is an agent review.** Two model workers wrote it and the lead spot-checked it. It is not human annotation, it is not independent ground truth, and it is not the development inspection that study brief section 10 asks a person to do. Do not copy any field of it into `results/pilots/ohr_dev_v1/inspection/annotations.csv`, which stays blank (24 rows, 0 filled). The readiness review that uses it is [live-baseline-readiness-2026-09-30.md](live-baseline-readiness-2026-09-30.md).

The 20 cases come from the purposive quota selection `purposive-quota-hash-rank-v1`, which fills category quotas (such as OCR gap, multi-page evidence and Han script) in hash-rank order. `inspection_cases.json` warns that the category proportions are not prevalence estimates. The sample is diagnostic. No count in this review estimates an error rate for the dataset or for the 70 pilot questions.

## How the review was done

- **Commit.** `c9b7ad5` on `research/prebaseline-engineering`. No code under `src/`, `scripts/` or `config/` changed between `884e388` and `c9b7ad5`.
- **Reviewers.** Cases 1 to 10 and cases 11 to 20 went to two separate workers of agent type `faar-worker`, which is configured as `claude-sonnet-5-5` at effort `high`. That configuration is recorded, not observed at run time. The coordinating agent (`claude-opus-5-5`, effort `high`) re-checked the claims marked "lead check". A separate `faar-worker` reviewer then fact-checked the document.
- **Order of work.** For each case the worker first looked at the page PNG and the MinerU text of that page. Then it read the question, then the gold answer and evidence pages. The question was already visible in step 1, because the packet shows it first. Cases 1 to 10 used no clean reference text. Cases 11 to 20 used only the `evidence_context` field, which comes from the clean reference, and not the reference files.
- **Artifacts.**
  - Page images: `results/pilots/ohr_dev_v1/inspection/pages/<file>.png` (150 DPI, listed in `render_record.json`, git-ignored, rebuilt by `scripts/data/build_pilot.py`).
  - Noisy OCR: `OHR-Bench/data/retrieval_base/MinerU/<doc_id>.json`, entry for the page.
  - Gold answers and evidence pages: `results/pilots/ohr_dev_v1/evaluation_manifest.json`.
  - Retrieved evidence: `prepared_requests.jsonl` from `run_pilot_live.py dry-run` at `c9b7ad5`. It is byte-identical to `.local/work/prompt-preview-final/prepared_requests.jsonl` (sha256 `b6cb6fcb...`). It holds full document text, so it stays local.
- **Chunk names.** `pN-cM` is page index N and chunk M of the question's document. The prompt shows the page as N+1.
- **Excerpts.** Quotes of document text are kept to a few words. The gold answers quoted here come from the vendored `OHR-Bench/data/qas_v2.json`, which is already in the repository.

## Summary

| Case | question_id | Condition | Sent | OCR keeps the answer | Sent evidence holds it | Likely origin of a miss | Gold questionable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `ecbf8bf8` | empty OCR, handwriting | no (`no_text_chunks`) | lost | not sent | OCR | yes |
| 2 | `9cb9f9aa` | empty OCR, handwriting | no (`no_text_chunks`) | lost | not sent | OCR | uncertain |
| 3 | `a7005e84` | empty OCR, notes | no (`no_text_chunks`) | lost | not sent | OCR | no |
| 4 | `b7b6de06` | two pages | yes | survives | yes, rank 1 | question ambiguity | yes |
| 5 | `b839af51` | two pages | yes | survives | yes, rank 1 | none expected | no |
| 6 | `b7bd7bec` | two pages, two sources | yes | survives | yes, rank 2 | mild ambiguity | no |
| 7 | `b7bd77ce` | same pages as 6 | yes | uncertain | partly | question ambiguity | yes |
| 8 | `844629d9` | Han table | yes | survives | yes, rank 1 | none expected | no |
| 9 | `c1bd4467` | Han newspaper | yes | survives | yes, rank 1 | none expected | no |
| 10 | `8445b560` | Han table | yes | survives | yes, rank 1 | none expected | no |
| 11 | `6b0bbdf8` | formula | yes | survives | yes, rank 1 | answer format | no |
| 12 | `6af79ad2` | formula | yes | survives | yes, rank 1 | answer format | no |
| 13 | `c13c36b1` | Han, reading order | yes | survives | yes, rank 1 | answer format | no |
| 14 | `4f1728e0` | text | yes | survives | partly | question ambiguity | yes |
| 15 | `339bfba4` | text | yes | survives | yes, rank 1 | none expected | no |
| 16 | `76a933cf` | text | yes | survives | yes, rank 2 | none expected | no |
| 17 | `a0ec729f` | handwriting | yes | lost | no | OCR | yes |
| 18 | `99e0aff2` | photographed notice | yes | damaged | yes, rank 2 | none expected | no |
| 19 | `76a3d0f7` | table | yes | survives | yes, rank 1 | none expected | no |
| 20 | `33a1d1c2` | text | yes | survives | no | retrieval | no |

In the table, "survives" means the reviewer found the answer text in the MinerU output. It does not show that the rest of the page text is correct. The "Gold questionable" column records a provisional agent concern (see the count below). It is not a finding about the gold answer.

Counts over the 20 cases, all agent judgement on a purposive sample:

- **Sent.** 17 of 20. Cases 1 to 3 have empty MinerU text, so no request is sent for them.
- **Answer present in the sent evidence.** 13 of 17 sent cases hold it fully and 2 hold it partly (cases 7 and 14). It is absent in 2: case 17, where the OCR lost it, and case 20, a retrieval miss. In the cases where it is absent, the first prompt can be answered only by abstaining or guessing. That does not make them abstention tests. A later retrieval or OCR repair step could supply the missing evidence.
- **Gold answer questionable (provisional).** The agents raised a concern about the gold answer in 5 cases (1, 4, 7, 14, 17) and were unsure about case 2. Four of the five are sent (4, 7, 14, 17). If a concern is right, a correct model answer there can score 0. A person has not adjudicated any of these concerns, and the frozen gold answers stay unchanged.
- **Answer-format risk.** Cases 11, 12 and 13 have long or formula-shaped gold strings that a correct paraphrase would miss.
- **Page readability.** Every page is readable at 150 DPI when the full-resolution PNG is viewed. Cases 1 and 3 are only partly readable (skewed handwriting, a small source image), and cases 13 and 14 need a crop to read the body text.

## Cases

Each case lists observations first and interpretation second.

### Case 1: `ecbf8bf8-e836-4b50-8ca8-ebc629deb1d7`, `textbook/GNHK_eng_EU_298`, page 0

- **Artifacts.** `pages/46966036c648_p0.png`. The MinerU page is an empty string. The dry run skips it (`no_text_chunks`).
- **Observations.** A photo of cursive handwriting in two inks over a printed greeting card, cut off at both edges. The printed title and caption are clear print. MinerU returned nothing, not even the print. The question asks for the central theme, and the gold is "Christmas and adoration".
- **Interpretation.** The gold matches the printed card, while the handwriting appears to be about belonging and family. The question does not say which of the two is "the document". The empty OCR loses even the printed title.
- **Uncertain.** The handwriting was only partly legible to the reviewer. The image does not show why MinerU emitted nothing.

### Case 2: `9cb9f9aa-bf4d-4eaa-81f2-005bc7537df5`, `textbook/GNHK_eng_AS_022`, page 0

- **Artifacts.** `pages/ece23208a272_p0.png`. The MinerU page is empty, and the dry run skips it (`no_text_chunks`).
- **Observations.** A full page of legible cursive on lined paper. The line that names the sites reads close to "vk movie site, epub hub". The gold is "UK movies site, epub hub".
- **Interpretation.** The gold may mistranscribe "vk" as "UK". Lead check: in a full-resolution crop the first letter can be read either way.
- **Uncertain.** "vk" against "UK", and the exact form of the second name. The question is not sent, so this does not affect the run.

### Case 3: `a7005e84-52db-4271-931d-a9205b6fc91a`, `textbook/omnidocbench_notes_f7f010b78016aeebd76e56d9283eb67f_46`, page 0

- **Artifacts.** `pages/c7c7581d90d8_p0.png` (774 x 1095, a small source image). The MinerU page is empty, and the dry run skips it (`no_text_chunks`).
- **Observations.** Neat handwritten English grammar notes with Chinese glosses. The Chinese rule under item (8) says "little" goes with uncountable nouns. The gold is "No".
- **Interpretation.** The gold looks correct. The empty OCR removes both languages.
- **Uncertain.** Only the item (8) rule was read closely. The small Chinese glyphs are near the limit of legibility.

### Case 4: `b7b6de06-f264-4539-bed9-373b5285b1a8`, `news/DUDE_4e51a6e4193828e37bfe5655919acbc8`, pages 0 and 1

- **Artifacts.** `pages/ed1ac6075b08_p0.png` and `_p1.png`. The sent chunks are ranks 1 to 5: `p1-c0`, `p0-c0`, `p1-c1`, `p0-c2` and `p0-c1`, which are all 5 chunks of the document.
- **Observations.**
  - Body text survives with letter-level noise and missing spaces ("cmail", "Senawrote"). Headlines are lost.
  - On page 1, an email from Anthony Sena lists names not approved, and that text is in rank 1.
  - Page 0 says a different person, spelled Serna, told her by email that she was not approved. That text is in rank 5.
  - On page 1, in chunks `p1-c0` and `p1-c1`, Serna describes the list's criterion: faculty whose professionalism or time in the classroom may not meet expectations. Rodriguez then says she still does not know why she was not approved.
  - The gold names Sena and gives no reason.
- **Interpretation.** The "why" half has a general answer, the list's criterion, but no reason specific to her, and the gold omits it. The "who" half has two defensible answers. The agents doubt the gold because it answers only half the question. This is a provisional concern. OCR and retrieval are not the expected cause of a miss. Lead check: the question text, the gold text and the Serna passage are as stated. The first draft of this review wrongly said the article gives no reason, and an independent reviewer caught it.
- **Uncertain.** Whether Sena and Serna are the same person. The page prints them as two different names.

### Case 5: `b839af51-a618-411f-8eb4-51b948d2339f`, `law/KNOWLABS,INC_08_15_2005-EX-10-INTELLECTUAL PROPERTY AGREEMENT`, pages 0 and 2

- **Artifacts.** `pages/f05d69fe37f7_p0.png` and `_p2.png`. Rank 1 is `p2-c2` (section 19 and the signature block), and rank 2 is `p0-c0` (the party definitions).
- **Observations.** Clean monospaced text. The OCR drops the title lines on page 0 and a four-line paragraph at the top of page 2. Section 19 survives. The gold is "Yes".
- **Interpretation.** The gold is correct. The dropped text does not touch the answer.
- **Uncertain.** Page 1 is not in the packet and was not viewed.

### Case 6: `b7bd7bec-4295-4976-b0ee-0dd8eff42770`, `administration/DUDE_54c771a3fc5da43da9e53e6854ef8a77`, pages 0 and 1

- **Artifacts.** `pages/831ee938058c_p0.png` and `_p1.png`. Rank 1 is `p0-c0` and rank 2 is `p1-c0`, which is the whole document.
- **Observations.**
  - A grainy typewriter cable with character errors ("MIKOYaM", "equ1pment").
  - The source itself is full of "groups unrecovered" gaps.
  - The organisation in the gold appears in comment [iii] on page 1.
- **Interpretation.** Answering needs a link from a marker in the letter to its comment. The gold is defensible. Another reading, the Navy Department, is possible.
- **Uncertain.** Whether a person would accept the alternative reading.

### Case 7: `b7bd77ce-af3a-4f3c-b86f-591a6cf81cbf`, same document and pages as case 6

- **Artifacts.** The same images and chunks as case 6.
- **Observations.** "Personnel" appears only next to gap markers. No sentence says who oversees personnel. The gold is hedged: the administration "was mentioned in association with overseeing personnel".
- **Interpretation.** The question may have no answer in the source, clean or noisy. A miss here says little about the OCR.
- **Uncertain.** Whether the gold was meant as an inference from the fragments.

### Case 8: `844629d9-e065-41f1-a353-f41b1573ec4c`, `textbook/omnidocbench_docstructbench_dianzishu_zhongwenzaixian-o.O-61510621.pdf_161`, page 0

- **Artifacts.** `pages/cd495fcc3c77_p0.png`. The sent chunks are ranks 1 to 3: `p0-c2`, `p0-c1` and `p0-c0`, which are all 3 chunks.
- **Observations.** MinerU renders the printed Chinese table as a LaTeX `tabular` with its rows in order. The row the question needs survives, including "7年", in rank 1. The gold is "7".
- **Interpretation.** No failure expected.
- **Uncertain.** How the scorer treats "7年" against "7" was not checked.

### Case 9: `c1bd4467-7db1-4e3f-9c91-7bc9fc625b02`, `news/omnidocbench_newspaper_2de954934f60ac538c5bd5dc40833a61_1`, page 0

- **Artifacts.** `pages/db8db059e940_p0.png`. The sent chunks are 5 of 23, and rank 1 is `p0-c8`.
- **Observations.** A dense Chinese newspaper page, readable in a full-resolution crop. Many headlines come out as bare `#` marks. The answer sentence crosses a column break, and MinerU joins it correctly. The gold is "免费提供建设规划图纸".
- **Interpretation.** No failure expected.
- **Uncertain.** Reading order was checked only around the answer.

### Case 10: `8445b560-af74-4c54-b7ca-306245f188ea`, same page as case 8

- **Artifacts.** The same image as case 8. Rank 1 is `p0-c1`, and rank 3 repeats the count through the chunk overlap.
- **Observations.** The count "93" and the name survive. The word "目录" is corrupted to a stray quote followed by "目". The gold is "93".
- **Interpretation.** No failure expected. The nearby number 86 is a possible distractor.
- **Uncertain.** Whether the damaged word affects a model's answer.

### Case 11: `6b0bbdf8-3437-4798-8b5a-07c1234bf373`, `textbook/GTM84-A_Classical_Introduction_to_Modern_Number_Theory1990.pdf_174`, page 0

- **Artifacts.** `pages/6210fd7671a4_p0.png`. Rank 1 is `p0-c0`, the whole page as one chunk.
- **Observations.** The final display formula survives as LaTeX, character for character. Elsewhere on the page, `x` replaces alpha in one line and an exponent is dropped in another. Spaces are lost in the prose. The gold is one string that holds two equalities.
- **Interpretation.** OCR and retrieval are fine. A correct answer written differently (one side of the equality, or plain text) would miss under exact match.
- **Uncertain.** LaTeX normalisation in the scorer was not checked.

### Case 12: `6af79ad2-993e-48c6-a6b1-06e289110af0`, same page as case 11

- **Artifacts.** The same image and chunk as case 11.
- **Observations.** `q\equiv1\left(m\right)` survives verbatim. Formula (4) on the page is partly damaged, outside the answer. The section title that the question quotes is lost.
- **Interpretation.** Answer-format risk only, for example "mod m" against "(m)".
- **Uncertain.** Scorer behaviour for that pair.

### Case 13: `c13c36b1-9a26-4370-a351-c31601ffcd2c`, `news/omnidocbench_newspaper_c3c88b76543e2caf044c9f7689f23ec9_1`, page 0

- **Artifacts.** `pages/3f1b02dcd914_p0.png`. The sent chunks are 5 of 33, and rank 1 is `p0-c6`.
- **Observations.**
  - The body text is readable only in full-resolution crops.
  - The page's reading order is scrambled in several places: sentences join fragments from other columns, and the two articles interleave.
  - The answer sentence itself is in order and complete in `p0-c6`, with "1300多万元", "33名" and "问责".
  - The clean `evidence_context` has its own character errors.
- **Interpretation.** The case was chosen for reading order, but that defect does not touch the answer chunk, so this question does not test it. The gold is a long two-part Chinese span, which is a scoring risk.
- **Uncertain.** How much of the rest of the page is misordered.

### Case 14: `4f1728e0-a481-4fc5-95d8-3d89694241b7`, `news/08_02`, page 0

- **Artifacts.** `pages/ced6f84f9f82_p0.png`. The sent chunks are 5 of 13: `p0-c5`, `c4`, `c9`, `c2` and `c12`.
- **Observations.**
  - The article text survives in order.
  - Three people are quoted in turn: Loomer, then Carlson, then Lil Pump. The next sentence says "Their comments, seemingly aimed at suggesting to Black voters" and breaks off at "Continued on Page A15".
  - The gold is "Tucker Carlson".
  - Rank 1 holds the three quotes. The "Their comments" sentence is in `p0-c6`, which was not sent.
- **Interpretation.** The page attributes the aim to all three people together, so the agents doubt the gold that names only Carlson. Lead check: the attribution is confirmed in the MinerU text. Whether the gold is wrong is not adjudicated.
- **Uncertain.** Page A15 is not in the document.

### Case 15: `339bfba4-d220-4c8e-a2d4-174a81cf8010`, law document as case 5, page 0

- **Artifacts.** `pages/f05d69fe37f7_p0.png`. Rank 1 is `p0-c0`.
- **Observations.** "the sum of" is followed by `$\$10.00$` in LaTeX. The gold is "$10.00".
- **Interpretation.** No failure expected. Punctuation stripping in the scorer should reduce both forms to the same string.
- **Uncertain.** Scorer behaviour on the LaTeX form was not run.

### Case 16: `76a933cf-65fd-4f69-93a2-5c9b6e3a52d7`, `textbook/chem-323236.pdf_183`, page 0

- **Artifacts.** `pages/34bd89712a12_p0.png`. The sent chunks are `p0-c0` and `p0-c1`, both chunks of the document.
- **Observations.** Spaces are lost in many lines, but "18 atoms of hydrogen" and `C_8H_18` survive, in rank 2. The gold is "18".
- **Interpretation.** No failure expected.
- **Uncertain.** Whether the run-together words affect the model.

### Case 17: `a0ec729f-b41c-480a-bcfb-8a406fda1a8b`, `textbook/GNHK_eng_EU_310`, page 0

- **Artifacts.** `pages/449aa037e4ac_p0.png`. Rank 1 is `p0-c0`, the only chunk.
- **Observations.**
  - A legible handwritten list of businesses with addresses and phone numbers.
  - The MinerU text contains no word from the page. It is two trigonometric fractions and then a long repeated `R o_{\Delta}` fragment.
  - The chunk is sent as evidence anyway. `ocr_condition` reports the page as `ok`.
  - The gold is "18 2959355".
- **Interpretation.** This is a clear OCR loss, and the status field does not flag it. The `ok` status says the page has text, not that the text is correct. Lead check on the image: "18" follows "Sandyford Ind. Est.", which reads as the Dublin 18 postal district and not as part of the number. The agents therefore doubt the gold. A person has not confirmed the doubt.
- **Uncertain.** The reading of the last digits, and whether "18" belongs to the address. The address reading is an inference from the layout.

### Case 18: `99e0aff2-f4f1-4be2-b904-753e088e0faf`, `administration/DUDE_aae11c0a8c4869c113687f3413f47300`, page 1

- **Artifacts.** `pages/f078ec9d688d_p1.png`. Rank 1 is `p0-c0`, an unrelated notice on page 0. Rank 2 is `p1-c0`.
- **Observations.** A photographed all-caps notice. Spaces are lost and words are corrupted ("1YrE", "RENCHES"). The clause "unreasonably interferes with the use of" survives, twice.
- **Interpretation.** Damaged OCR, but the answer can be recovered. The gold is a fair paraphrase.
- **Uncertain.** Whether a model reads through the run-together capitals.

### Case 19: `76a3d0f7-998d-476a-aac4-55c92f7a8e02`, same page as case 16

- **Artifacts.** The same image as case 16. Rank 1 is `p0-c1`.
- **Observations.** The prefix table survives as a LaTeX `tabular`, including `4 & tetra-`. The gold is "tetra-".
- **Interpretation.** No failure expected.
- **Uncertain.** Whether the table continues beyond the page.

### Case 20: `33a1d1c2-ad4a-4155-809f-cd8f739cd488`, law document as case 5, page 0

- **Artifacts.** `pages/f05d69fe37f7_p0.png`. The sent chunks are `p2-c2`, `p1-c2`, `p0-c0`, `p1-c1` and `p1-c5`.
- **Observations.**
  - The needed definition, with "during or after working hours", is on page 0 in chunks `p0-c3` and `p0-c4`, and the OCR keeps it.
  - Neither chunk was sent. None of the five sent chunks contains "working hours".
  - The gold is "Yes".
- **Interpretation.** A retrieval miss. With this evidence the model can only abstain or guess. The gold page 0 is among the retrieved pages, because `p0-c0` was sent. A page-level check would therefore count this case as covered, and the passage the question needs was not retrieved. Lead check: confirmed on the prepared request.
- **Uncertain.** Why the lexical ranking placed the two chunks below the sent five.

## Patterns across cases

- **Whole documents reach the model.** In 11 of the 17 sent cases (4, 6, 7, 8, 10, 11, 12, 16, 17, 18 and 19) the retriever sent every chunk of the document, so retrieval cannot fail there. This matches study brief section 15.12.
- **OCR status misses degenerate text.** `ocr_condition` counts a page as `ok` whenever it has text. The status says only that text exists. It does not show that the text is correct. Case 17 shows that degenerate text passes as `ok` and is sent.
- **Cases that do not test their category.** Case 13 does not test reading order, and cases 6 and 7 share their evidence exactly.
- **No instructions aimed at a model.** Instruction-like wording in the evidence ("Please be advised", "Please wire") is ordinary document speech. The prompt audit found the same over all 63 prompts.

## What this review cannot show

- It estimates no rate. The 20 cases are a purposive diagnostic sample, so no count above says how often OCR loses an answer, retrieval misses one or a gold answer is wrong across the dataset.
- The concerns about gold answers are not adjudicated. A person has to confirm or reject each one.
- It is not a second labeller and gives no agreement figure. The diagnosis study needs at least two independent people (study brief section 10).
- No model was run, so every "no failure expected" is a judgement about the evidence, not an observed answer.
- Scorer normalisation of LaTeX, units and Chinese spans was not tested case by case.
- Pages outside the packet were not viewed: page 1 of the law document, and page 0 of the case 18 document.
