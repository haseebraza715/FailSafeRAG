# OHR-Bench scorer provenance

Date: 2026-09-29. Status: an evidence record with one open decision for the research lead. It is not legal advice and it does not settle whether the scoring code, or the repository that contains it, may be released.

`src/faar/ohr_scoring.py` reimplements the answer metrics of OHR-Bench. This report traces where each function comes from, what licence evidence exists for each source, and what is still unknown. Sections 1 to 5 are verified facts, each with a source. Section 6 is interpretation. Section 7 lists the options. Section 8 lists what was not checked.

All network retrievals were made on 2026-09-29 with `gh api`, `curl` and a PDF text extract. The sandbox clock read 2026-09-28 UTC at that time. Working copies of every retrieved file are in the investigator's scratch notes and are not committed.

## 1. Verified facts: sources and licence evidence

| Component | Location | Revision or date | Licence evidence found |
|---|---|---|---|
| OHR-Bench code | https://github.com/opendatalab/OHR-Bench | `1f421eb428f9f5b8ac0bc8064d6ad1f13fab7af7` (main, 2025-12-03) | None. `gh api repos/opendatalab/OHR-Bench` gives `license: null`, `repos/.../license` returns 404, and the git tree at that commit has no LICENSE, COPYING or NOTICE file and no `.github` directory. `src/metric/common.py` has only `# @Author : Shichao Song` and an email line. |
| OHR-Bench dataset terms, README | `README.md` at the same commit (sha256 `6ebee476...`, equal to the vendored copy) | same | Lines 445 and 446, "Copyright Statement": the PDFs come from public channels, the dataset "is for research purposes only and not for commercial use", and copyright concerns go to `OpenDataLab@pjlab.org.cn`. The statement names the dataset, not the code. |
| OHR-Bench dataset terms, Hugging Face | https://huggingface.co/datasets/opendatalab/OHR-Bench, card sha `7f833e3eda9a571a9ea545a8f6d476fa1685033d` (last modified 2025-08-28) | 2026-09-29 | Front matter `license: cc-by-4.0`. The card body repeats the research-only, non-commercial sentence (line 458 of the raw card) and says the repository "contains the official code". The repository file list has only `README.md`, `.gitattributes`, two parquet files, `pdfs.zip`, `retrieval.zip` and a `figs` directory. It holds no source code. |
| OHR-Bench paper | arXiv 2412.02592, v4 (last revised 2025-08-30) and v2 | 2026-09-29 | The abstract page shows the arXiv non-exclusive distribution licence for the paper text. The paper says the PDFs, Q&As and ground-truth data "are released at" the GitHub URL. It states no licence for code or data and does not mention exact match or SQuAD. It names an F1 metric for generation. |
| CRUD_RAG | https://github.com/IAAR-Shanghai/CRUD_RAG | main `1aace383994e1f68efa12cf2a8e2dadfb4102ceb` (2025-05-20), 38 commits | None. `license: null` and no LICENSE file in the tree. Its `src/metric/common.py` has the same author lines as OHR-Bench's file. |
| SQuAD v1.1 evaluation script | CodaLab bundle `0xbcd57bee090b421c982906709c8c27e1`, https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/ | bundle created 2016-08-27 07:28:20 UTC, described as "Official evaluation script v1.1"; file sha256 `f5a673db...` | None found. The bundle metadata has `"license": ""`. The file has no licence or copyright line. |
| Copy of the SQuAD v1.1 script | https://github.com/allenai/bi-att-flow, `squad/evaluate-v1.1.py` | commit `498c8026d92a8bcf0286e2d216d092d444d02d76`, 2016-09-15, author Minjoon Seo | Byte-identical to the CodaLab file (same sha256). The repository has an Apache-2.0 `LICENSE` whose copyright line is the unfilled template `Copyright [yyyy] [name of copyright owner]`. The file itself has no notice. |
| SQuAD v2.0 evaluation script | https://github.com/rajpurkar/SQuAD-explorer, `evaluate-v2.0.py` | repo `eee5fdbf62f8613a7812b03419e6b29617b74fd1` (2023-10-12); file added 2021-05-27 in `09eac9971f46889fa057ff2c870bf71092ba9d55`; file sha256 `710840ce...` | The repository `LICENSE` is MIT, "Copyright (c) 2020 Pranav Rajpurkar", created 2020-06-02 in `0274139326e6f6d55eca042065b74db87ab135f4`. The script has no per-file header. The SQuAD web page states CC BY-SA 4.0 for the dataset. |
| HotpotQA evaluation script | https://github.com/hotpotqa/hotpot, `hotpot_evaluate_v1.py` | repo `3635853403a8735609ee997664e1528f4480762a` (2019-02-14); file added 2018-09-25 and changed once on 2018-10-05 (`fa3a36370899e1d85822de61e58c85ea19993154`, a float cast); file sha256 `d35fc91a...` | `LICENSE.txt` is Apache License 2.0 (sha256 `9c264489...`). Its appendix boilerplate reads "Copyright 2018 Zhilin Yang, Peng Qi, Saizheng Zhang". The script has no per-file header, and the repository has no NOTICE file. The HotpotQA site mentions CC BY-SA 4.0 for the dataset and its Wikipedia dump. |
| Vendored OHR-Bench tree in this repository | `OHR-Bench/`, added in `fc935032791bb08ea32a807e29fc9254506fa0c1` (2026-04-04) | 3,158 tracked files: 3,120 under `data/retrieval_base`, three QA files, 22 `.py` files | No licence file is vendored. `common.py`, `quest_answer.py` and `evaluator.py` hash equal to upstream at the pinned commit (`9fe7eb52...`, `1feaee17...`, `b1098069...`). Four vendored files under `src/.cache/huggingface/` carry an Apache-2.0 header from "The HuggingFace Evaluate Authors". |

The upstream file hashes were re-fetched with `gh api` on 2026-09-29 and matched the vendored files.

### The repository's remote is public

`git remote -v` gives `origin https://github.com/haseebraza715/FailSafeRAG.git`. On 2026-09-29 `gh api repos/haseebraza715/FailSafeRAG` reported `visibility: public`, `license: null`, last push 2026-09-28T21:41:57Z. Its `main` and `research/prebaseline-engineering` branches serve the vendored `OHR-Bench/` tree. The blob `OHR-Bench/src/metric/common.py` there has git sha `d593d461dc5bef006b47e017fc60b042545848d4`, the same as the local file. `src/faar/ohr_scoring.py` is not on that remote's `main` (the contents API returned 404). So the pre-existing vendored copy of the upstream code and dataset files is already publicly redistributed. Section 7 covers this.

## 2. Verified facts: ancestry of each function

The upstream history of `src/metric/common.py` (`gh api "repos/opendatalab/OHR-Bench/commits?path=src/metric/common.py"`) has three commits, all by the GitHub user `Carkham`:

| Commit | Date | Message | Change to the scorer functions |
|---|---|---|---|
| `96a9b56b83b2b498e2859dbbea9d0bce58529b3c` | 2024-12-03 | init commit | `normalize_answer`, `exact_match_score`, `f1_score` with the yes/no/noanswer rule, `catch_all_exceptions` returning 0 on failure. No `has_chn_character` and no CJK branch in `f1_score`. The file already has `import jieba` and a separate `f1_zh` that tokenises with `jieba.cut`, which `f1_score` does not call. |
| `af58d0e032730dc9907349920040d31c500dc9b6` | 2024-12-06 | improve the evaluation code to enhance usability | `catch_all_exceptions` returns -1 instead of 0. No other change. |
| `92cf7be9e2cadc1478b917d76020491e376cea04` | 2025-03-10 | updates OHRBench-v2 | Adds `has_chn_character` and the `jieba.lcut` branch in `f1_score`. File sha256 `9fe7eb52...`, unchanged through `1f421eb`. |

The pinned commit therefore scores differently from OHR-Bench before 2025-03-10. Chinese text and the exception return value differ across that boundary.

| Function | Origin traced | Evidence |
|---|---|---|
| `normalize_answer` | SQuAD v1.1 script, unchanged in HotpotQA. OHR-Bench uses `regex.sub` for `re.sub` and moves `import string` inside `remove_punc`. | Same four nested helpers (`remove_articles`, `white_space_fix`, `remove_punc`, `lower`), same composition order, same pattern `\b(a|an|the)\b`. A `difflib` comparison of the function text against HotpotQA's gives 0.931 similarity, and the differences are quote style, type hints and the import placement. |
| `exact_match_score` | SQuAD v1.1 and HotpotQA `exact_match_score` (same name and body). | OHR-Bench wraps the comparison as `1 if ... else 0`, which returns an `int` where the ancestors return a `bool`. Similarity 0.946. |
| `f1_score` | HotpotQA `f1_score`. | Same name, same `ZERO_METRIC` constant, same two yes/no/noanswer guards, same Counter overlap and formula. Differences: `ZERO_METRIC = 0` instead of `(0, 0, 0)`, one return value instead of three, and the added CJK block. Similarity 0.888. The SQuAD v1.1 `f1_score` has no yes/no/noanswer rule, so SQuAD alone does not explain the function (similarity 0.563). The SQuAD v2.0 `compute_f1` is a different function. |
| `has_chn_character` | OHR-Bench, commit `92cf7be`. | No earlier source found. The closest predecessor of the `jieba` branch is OHR-Bench's own `f1_zh` in the init commit `96a9b56`. `gh search code "def has_chn_character"` returns only OHR-Bench and a copy of it. The test `'CJK' in unicodedata.name(char)` appears in about 10 other public repositories, some of them copies of each other, so the idiom is common. The function name, the `"""chn"""` docstring and the `try/except ValueError: continue` layout are OHR-Bench's. The code search tool is not exhaustive. |
| `catch_all_exceptions` | CRUD_RAG's decorator of the same name, changed by OHR-Bench. | CRUD_RAG's version (`src/metric/common.py`, single commit `34c93030f5a3f0b3153fb55a5b5da09ac2ae2bf0`, 2024-01-31) logs a warning and returns `None`. OHR-Bench's returns 0 (init) and then -1, and returns a five-zero tuple for `bleu_score`. |

CRUD_RAG is not the source of the EM and F1 functions. `git log --all -S<text> -- '*.py'` over the full CRUD_RAG history (38 commits, all refs) found no revision containing `normalize_answer`, `def f1_score`, `exact_match`, `noanswer` or `has_chn_character`. The README statement that the framework "is based on" CRUD_RAG (line 433) is consistent with the shared task and evaluator layout and the decorator. It does not cover these functions.

## 3. What `src/faar/ohr_scoring.py` contains

| Part | Origin | Note |
|---|---|---|
| `normalize_answer` | Flat rewrite of the SQuAD/HotpotQA function. | Same operations in the same order. The nested-helper structure and the identifiers differ. The authors had read the OHR-Bench file. |
| `exact_match_score` | OHR-Bench's `1 if ... else 0` form of the SQuAD/HotpotQA function. | Statement-level identical to OHR-Bench's after removing decorators and annotations. |
| `f1_score` | HotpotQA layout plus OHR-Bench's scalar zero and CJK block. | Text similarity to OHR-Bench's function is 0.96 after lowercasing `ZERO_METRIC`; to HotpotQA's it is 0.83. The one structural change is a module-level `_YES_NO` tuple in place of a repeated list literal. |
| `has_chn_character` | OHR-Bench only. | Same loop, same test, same exception handling. Statement-level similarity 0.80. |
| `_catch_all_exceptions` | OHR-Bench's -1 behaviour, in a small decorator. | A short decorator. It also copies `__name__` and `__doc__`, which upstream's does not. |
| Adapter (`score_predictions`, aggregates, `scorer_identity`, statuses, `upstream_valid_only`) | Written for FAAR. | `upstream_valid_only` reproduces the filtering behaviour of `evaluator.py`; it reuses the behaviour, not the code. |

The module is not a clean-room implementation. Its authors read `src/metric/common.py` at the pinned commit before writing it. `scripts/experiments/ohr_scoring_parity.py` runs the vendored upstream file as a test oracle. It is not imported by the package.

## 4. Terms by category

Each category is separate. The terms of one do not carry over to another.

1. **Dataset terms.** The README says research use only and no commercial use. The Hugging Face card says CC BY-4.0 in its front matter and repeats the research-only sentence in its body. The two statements conflict on commercial use, and no source resolves it. Both cover the dataset, and the Hugging Face repository contains no code.
2. **Evaluation-code terms, OHR-Bench and CRUD_RAG.** No licence in either repository. No licence statement in the README, the paper or the dataset card names the code.
3. **Third-party code origins.** HotpotQA script under Apache-2.0 (repository licence, no per-file header). SQuAD v2.0 script under MIT (repository licence, no per-file header). SQuAD v1.1 script with no licence at its primary location. A copy of the v1.1 script sits in an Apache-2.0 repository whose copyright line is blank.
4. **Our implementation.** `src/faar/ohr_scoring.py`. The repository has no LICENSE file of its own (`find . -maxdepth 1 -iname 'licen*'` finds none), so the module has no stated licence either.
5. **Pre-existing vendored copy.** The unmodified upstream tree under `OHR-Bench/`, including the same `common.py`. It has no licence file and the upstream repository has none.

## 5. Changes made in this branch

`src/faar/ohr_scoring.py`: the module docstring only. It now lists the third-party origins in section 2 with revisions and copyright lines, states that the module is not clean-room, and no longer says the functions are simply "the well-known SQuAD layout". No code, constant or behaviour changed. After the edit, on 2026-09-29:

- `python -W ignore scripts/experiments/ohr_scoring_parity.py` exited 0 with 0 mismatches in 99,045 pairs (107 edge cases, 93,478 `qas_v2.json` pairs, 5,460 pilot pairs).
- `python -W ignore scripts/experiments/ohr_scoring_parity.py --check-fixture` exited 0 ("fixture matches upstream").
- `pytest -q -p no:cacheprovider tests/test_ohr_scoring.py` gave 191 passed.
- `ruff check src/faar/ohr_scoring.py` passed.

The docstring gives attribution. It adds no Apache-2.0 licence text, and it makes no claim that any licence covers the module.

## 6. Interpretation (not verified fact)

- The HotpotQA and SQuAD portions have a permissive licence on the public record that was not tied to OHR-Bench. HotpotQA's is Apache-2.0 and SQuAD v2.0's is MIT. Whether the OHR-Bench authors complied with those licences when they copied the code is not visible. Whether our module, written from the OHR-Bench copy, receives those licences is a legal question this report does not answer.
- The SQuAD v1.1 script has no licence at its primary location. Its function bodies are the same as those in the MIT-licensed v2.0 script, so the v2.0 script is a licensed source for `normalize_answer` and the EM comparison. It is not a source for the yes/no/noanswer rule.
- The parts with no permissive ancestor are small: `has_chn_character` (10 lines), the four-line `jieba.lcut` branch, the `int` return in `exact_match_score`, and the -1 decorator. Their behaviour is fully described in the module docstring. The CJK-name test is a common Python idiom. Whether any of it is protected expression is a legal question.
- The dataset licence gives no permission for the code, and the OHR-Bench credit lines and the CRUD_RAG acknowledgement grant nothing. Attribution here is a courtesy and a research-integrity record, not a licence.

## 7. Disposition and open decision

**Disposition for the code actually reused.** Clear permission exists for part of it and is missing for the rest. It is not resolved for the module as a whole.

- Permissive terms exist on the public record for `normalize_answer`, the EM comparison and the `f1_score` layout (MIT via SQuAD v2.0, Apache-2.0 via HotpotQA). The docstring now carries attribution for them. No Apache NOTICE text was needed, because the HotpotQA repository has no NOTICE file.
- No permission was found for `has_chn_character`, the CJK branch, the `int` EM return and the -1 decorator, which come only from OHR-Bench.

Until the questions below are settled, treat the module as usable inside the project for research and not as redistributable.

**Open question 1 (release scope).** Will this repository, the paper's artifact, or `ohr_scoring.py` be published, given that the vendored tree is already public on `origin`?

**Open question 2 (OHR-only parts).** May the four OHR-only elements above be redistributed?

The options below answer question 2. They are not exclusive.

| Option | What it does | Tradeoff | Evidence that resolves it |
|---|---|---|---|
| A. Keep for private research use only | No change. Do not publish `ohr_scoring.py` or the paper artifact with it. | Cheapest. Does not fix the public `origin`. Blocks releasing reproducible scoring with the paper. | The lead's decision on release scope. |
| B. Seek written permission from OpenDataLab | Ask for a written licence for `src/metric/common.py`, or for the upstream repository. The README names `OpenDataLab@pjlab.org.cn` for copyright concerns. An unsent draft exists only in the lead's local scratch notes; it is not committed and was not sent. | Slow, and a reply is not guaranteed. A grant covers only what OHR-Bench owns. It cannot license the HotpotQA and SQuAD parts, which already have their own terms. | A written reply granting a named licence, or an upstream commit adding a LICENSE file. |
| C. Re-derive from the permissive ancestors | Rebuild the module from `hotpot_evaluate_v1.py` (Apache-2.0) and `evaluate-v2.0.py` (MIT), add their notices, and express the four OHR-only elements from the behaviour spec in the docstring. Keep the parity script as the check. | Cheap, and the parity run shows behaviour is preserved. The OHR-only elements are still written by people who have seen the OHR file, so this reduces the exposure and does not make the module clean-room. It changes code, so it is outside this task. | A parity run with 0 mismatches, plus a reviewer's judgment that the residual elements are functional. |
| D. Independent reimplementation | One person writes a behaviour spec from the docstring. A second person who has not read `common.py` writes the code from the spec and the permissive ancestors. | Strongest separation for the OHR-only elements. Costs two people and a review. | The same parity run, and a record that the implementer did not read `common.py`. |
| E. Use a different, plainly licensed scorer | Score with the SQuAD v2.0-style metric and report the difference. | Loses parity with the numbers OHR-Bench publishes, and contradicts the study's plan to use the official metric (study brief section 12). | A brief change approved by the lead. |

Recommendation, as the investigator's judgment: keep A in force now, and treat C as a small mechanical follow-up if release is planned, since the parity script makes it checkable. B costs an email and is worth asking in any case. The public `origin` needs a separate lead decision, because it involves dataset files under a research-only statement and code with no licence, and removing files from a public repository does not remove earlier commits.

**Decision needed from the lead.** Choose the release scope (question 1) and, for question 2, pick from A to E. Nothing in this report answers whether the vendored tree on `origin` should stay public.

## 8. Not verified

- Whether OpenDataLab or the OHR-Bench authors would grant permission. Nobody was contacted.
- The licence status of the SQuAD v1.1 script beyond the CodaLab bundle metadata and the two hosts checked. Older hosting pages on the SQuAD site were not retrieved.
- Whether the HotpotQA authors copied their script from SQuAD, or from another source. No header says so.
- Whether other public code contains the OHR-Bench wording of the CJK test. The GitHub code search is not exhaustive.
- Whether Apache-2.0 section 4 requires more than the docstring gives, for code that reached us through an intermediary. That is a legal question.
- The intent behind the public `origin`, and whether the Hugging Face card's CC BY-4.0 and its non-commercial sentence are meant to coexist.
- The OHR-Bench paper was searched as extracted text (`pypdf`) for licence, copyright and metric words. Figures and tables were not read.
- Any OHR-Bench history before the three commits listed for `common.py`. The repository was created on 2024-11-29 and the first `common.py` commit is 2024-12-03.

## 9. Reproduce the checks

```bash
gh api "repos/opendatalab/OHR-Bench/commits?path=src/metric/common.py&per_page=100"
gh api -H "Accept: application/vnd.github.raw" "repos/opendatalab/OHR-Bench/contents/src/metric/common.py?ref=<sha>" | shasum -a 256
gh api repos/opendatalab/OHR-Bench --jq .license
gh api repos/hotpotqa/hotpot --jq .license
gh api repos/rajpurkar/SQuAD-explorer --jq .license
curl -sL https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1 | head -c 600
git clone --filter=blob:none --no-checkout https://github.com/IAAR-Shanghai/CRUD_RAG.git
git log --all --oneline -S"normalize_answer" -- '*.py'   # run in the clone; prints nothing
PYTHONPATH=src python -W ignore scripts/experiments/ohr_scoring_parity.py
PYTHONPATH=src python -W ignore scripts/experiments/ohr_scoring_parity.py --check-fixture
```
