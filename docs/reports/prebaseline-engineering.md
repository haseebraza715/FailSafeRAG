# Pre-baseline engineering (2026-09-28)

This report covers the branch `research/prebaseline-engineering` (draft PR
[#2](https://github.com/haseebraza715/FailSafeRAG/pull/2)). The branch prepares
offline infrastructure for the first FAAR study. It made no model call, paid
request, OCR run, model download or GPU job. Its one pilot run uses a
rule-based extractor and no repair. That run checks the data flow, the record
format and the scoring join. It is not a baseline, and its answer score says
nothing about whether FAAR improves answer quality.

## What changed and why

| Problem found | Evidence | Change |
| --- | --- | --- |
| CI failed on `main` (run 36459962433 on `8832ce4`: 18 failed, 863 passed, 6 skipped) | Saved log. 10 failures were `No module named 'pypdfium2'`. 7 were preflight tests whose `dependencies` check found `docling` and `pypdfium2` missing. 1 needed `OPENAI_API_KEY`. The previous run, 31318349678 on `253b255`, passed. The 131 commits between them had no CI run of their own. | `pypdfium2`, `jieba`, `regex` and `pillow` declared in the base dependencies. `test` and `lint` extras. Preflight tests stub `docling` and keep a test for a missing dependency. The OpenAI test uses a mocked client and asserts that the mock was called. |
| `faar-demo --help` raised `TypeError: Secondary flag is not valid for non-boolean flag` | Reproduced with click 8.4.2 and 8.5.0 against typer 0.12.5 | `click==8.1.8` pinned. `tests/test_cli_help.py` renders help for every command. |
| Tests wrote to the real `logs/vlm_calls.jsonl` | Reproduced: one test, and only when a key is set | `tests/conftest.py` redirects API-call logs to a temporary directory and fails the session if any file under `logs/` changes. |
| Tests could reach the network and read real keys | Three preflight tests sent HEAD requests to huggingface.co | `tests/conftest.py` removes provider keys, blocks non-loopback connections and sets `HF_HUB_OFFLINE=1`. |
| The full suite segfaulted on macOS | Reproduced. The `faiss-cpu` and `torch` wheels each bundle `libomp`. | On macOS only, `tests/conftest.py` limits both libraries to one thread and sets `KMP_DUPLICATE_LIB_OK`. The pytest header says so. No test is skipped. |
| `uv.lock` was stale and `uv lock --check` failed | The lock described 9 packages with typer 0.26.8 | The lock was regenerated (279 packages) for Linux x86_64 and macOS arm64. `config/environment/constraints-*.txt` match it. |
| The study brief overstated or misstated several points | Section-by-section review | See [the study brief changes](#study-brief-changes). |
| No official OHR-Bench scorer | `src/faar/metrics.py` erases non-ASCII text | `src/faar/ohr_scoring.py` (see [scoring](#scoring)). |
| No code could run the frozen pilot from its manifests | Study brief section 11 | `src/faar/pilot_runner.py` and `scripts/experiments/run_pilot_offline.py` (see [offline runner](#offline-runner)). |

CI now has two jobs in `.github/workflows/tests.yml`:

- **Lint and package checks:** `ruff`, `uv lock --check`, and a wheel and sdist build checked with `twine`.
- **Offline tests:** an install with the CPU build of torch, `pip check`, a `faar-demo --help` smoke check, `registry.py check`, the full pytest suite and the scorer parity script.

It needs no key, no network service beyond package indexes, and no ignored research data.

## Scoring

`src/faar/ohr_scoring.py` reimplements the two functions that the upstream QA
task calls. `QuestAnswer.scoring` in `src/tasks/quest_answer.py` calls
`exact_match_score` and `f1_score` from `src/metric/common.py` on the
`answers` string. The upstream repository is `opendatalab/OHR-Bench` at
`1f421eb428f9f5b8ac0bc8064d6ad1f13fab7af7`, the `commit_at_lookup` in
`config/ohr_pdf_source_lock.json`.

- **Upstream identity.** On 2026-09-28 the lead fetched the upstream `common.py`, `quest_answer.py` and `evaluator.py` at that commit with `gh api`. Each has the same sha256 as the vendored copy under `OHR-Bench/` (`common.py`: `9fe7eb52...`).
- **Licence.** The upstream repository has no licence file, and GitHub reports none. The README restricts the dataset to research use, and the Hugging Face dataset card declares CC-BY-4.0 for the dataset. Neither covers the code. The [scorer provenance record](ohr-scorer-provenance.md) of 2026-09-29 traces each function to SQuAD v1.1, HotpotQA (Apache-2.0) or OHR-Bench, and records the open question. Redistribution terms for the code are a lead decision.
- **Parity.** `scripts/experiments/ohr_scoring_parity.py` compares the module with the vendored upstream functions by value and by Python type. It uses `jieba==0.42.1` and `regex==2024.7.24`, the upstream pins. On 2026-09-28 it found 0 mismatches in 99,041 pairs:
  - 103 hand-written edge cases;
  - 11 variants of each of the 8,498 `qas_v2.json` references;
  - 8 variants of each of the 70 pilot references plus all 4,900 cross pairs.

  After the review added four upstream-recorded edge cases, the committed fixture holds 105 and the script compares 107 edge cases. On 2026-09-29, at `8479309`, the same command reported 0 mismatches in 99,045 pairs (107 + 93,478 + 5,460). CI runs the script.
- **Quirks kept.** ASCII punctuation is deleted without a space, so `1,000` equals `1000`, `3.5` equals `35` and `50%` equals `50`. An empty prediction against a reference that normalises to empty scores EM 1 and F1 0. 5 of the 8,498 references do so, and none of the 70 pilot references. The yes, no and noanswer rule applies on both sides. Text with a CJK character goes through `jieba.lcut`.
- **Old normaliser.** `src/faar/metrics.py` gives a false perfect score only when the prediction and the reference both normalise to empty. 521 of 8,498 references and 7 of 70 pilot references normalise to empty under it.
- **Aggregates.** `score_predictions` keeps every question in `all_questions` (denominator 70 for the pilot). It scores `execution_failed` as 0 and scores `no_evidence` by the official metric on the empty abstention. It also reports `answered_only` and `upstream_valid_only`. The second matches upstream's headline `overall`, which drops blank answers (`evaluator.remove_invalid`). Counts of `answered`, `no_evidence`, `execution_failed` and `abstained` are reported separately. No repair, damage or FAAR-comparison field exists.
- **Scorer environment.** `score_summary.json` records the jieba, regex and Unicode database versions.

## Offline runner

The runner writes to `results/engineering/<run_id>/`. The flow is:

1. `generate` reads `results/pilots/ohr_dev_v1/runtime_manifest.json` and each document's MinerU file. It verifies the file hash and every page hash.
2. It builds one local-hash retriever per document, retrieves within the question's own document, and answers with `faar.answering.answer_from_hits`.
3. It writes `predictions.jsonl`, `run_config.json` and `generation_summary.json`.
4. `score` joins the saved predictions to `evaluation_manifest.json` and writes `scores.jsonl` and `score_summary.json`.

- **Separation.** `generate_run` never receives an evaluation path. The noisy-text root must equal `OHR-Bench/data/retrieval_base/MinerU`, and every noisy-text path must resolve inside it. A runtime question may hold only `question_id`, `doc_id` and `question`. Tests confirm three things:
  - generation opens only the runtime manifest and the MinerU files;
  - poisoned evaluation files leave predictions byte-identical;
  - a manifest that points at the gt tree is refused.
- **Scope.** A hit from another document raises and becomes `execution_failed`. The per-document index makes such a hit impossible in normal operation.
- **Outcomes.** Every question gets exactly one record: `answered`, `no_evidence` or `execution_failed`. A failure is recorded per question and the run continues. The exit code is 0 for a clean run, 1 for a refusal, 2 when any `execution_failed` exists, and 3 for an internal error.
- **No-evidence reasons.** Each `no_evidence` record has one reason:
  - `no_text_chunks`: every page is empty or missing, which is an OCR status.
  - `no_text_content`: the text is non-empty but holds no letter or digit in any script.
  - `no_retrieval_tokens`: the text has content, but no token the engineering tokeniser indexes.
  - `no_hits`: retrieval returned zero hits.
- **Query tokens.** Every record has `query_retrieval_tokens`, and the summary counts questions with none. On 2026-09-28 the retrieval tokeniser matched only `[a-z0-9%$]`, so a Han-only question got hits ranked by position alone. The 2026-09-29 extension below replaces this with `multilingual-v1`.
- **Overwrite rules.** The runner never overwrites. For a run directory that already holds a run:
  - With a different fingerprint, the run is refused.
  - With the same fingerprint, it is regenerated in memory. Identical bytes are reported as "already complete, verified identical". Different bytes are refused.
  - A partial directory is refused.

  Inside the project, only `results/engineering/<run_id>/` is accepted. A path with a `results/pilots` pair is refused in any letter case.
- **Provenance.** `run_config.json` records:
  - the commit, the dirty flag and dirty paths;
  - the fingerprint, which covers the `src/faar` code, the CLI script, the manifest and MinerU hashes, retrieval settings, backend, injected failures and package versions;
  - the retrieval settings, labelled as engineering settings and not the approved protocol;
  - the backend identity, with `engineering_only: true` and `model_calls: false`;
  - the command.

  Paths are relative to the project.

## Pilot runs

All three runs are `engineering_check` records in `experiments/registry.jsonl`.

| run_id | Code | Outputs | Status |
| --- | --- | --- | --- |
| `2026-09-28-ohr-dev-v1-offline-engineering` | `8953dfd` | `.local/work/runs/` in the lead's checkout only. That directory is ignored by git and has no backup elsewhere (`backup: none`). | Completed. Not committed because its `run_config.json` recorded two absolute home-directory paths. |
| `2026-09-28-ohr-dev-v1-offline-engineering-r2` | `58e1693` | `results/engineering/2026-09-28-ohr-dev-v1-offline-engineering-r2/` | Completed. Superseded by r3 after the review fixes. |
| `2026-09-28-ohr-dev-v1-offline-engineering-r3` | `adfb2d3` | `results/engineering/2026-09-28-ohr-dev-v1-offline-engineering-r3/` | Completed. It was the current run on 2026-09-28, and r4 superseded it on 2026-09-29 (see the extension below). |

Each run recorded one dirty path, `experiments/registry.jsonl`, because the
registry line was written before the run started. All three produced the same
answers. r3 differs from r2 only in the new `query_retrieval_tokens` field and
in the reason recorded for one question.

The r3 outcomes for the 70 questions are as follows.

| Measure | Count |
| --- | --- |
| `answered` | 63 |
| `no_evidence` | 7: 6 `no_text_chunks`, 1 `no_text_content`, 0 `no_retrieval_tokens`, 0 `no_hits` |
| `execution_failed` | 0 |
| `abstained` | 7 |
| Questions without a retrieval token in the query | 19: 18 `answered`, 1 `no_evidence` |
| Official EM, all questions | 4 of 70 (0.0571). F1 0.0960. |
| Official EM, `answered_only` and `upstream_valid_only` | 4 of 63 (0.0635). F1 0.1066. |

- **The 6 `no_text_chunks` questions.** Each is on a single-page document whose page has empty MinerU text: 4 GNHK handwriting pages and 2 OmniDocBench note pages.
- **The `no_text_content` question.** Question `ecccdf67-bc03-447c-a990-d5989185f21a` is on `textbook/omnidocbench_notes_1ba14cb325bc448f7201b20502ecf2b5_124`.
  - Its one page has MinerU text `"# \n\n#"`: two empty Markdown heading markers. The runtime manifest calls the page `ok` because the text is non-empty.
  - The gt reference page holds a full page of Chinese text. The lead read it for this diagnosis only, and no runtime code reads it.
  - The noisy input therefore has no usable evidence for the question. The cause is lost content in the noisy text, not missing OCR and not a tokeniser limit.
  - Before the review fix, the runner reported this case as `no_retrieval_tokens`.
- **Rerun checks on the real pilot.**
  - A rerun into the same directory was a verified no-op.
  - Injecting one failure gave exit 2 and 1 `execution_failed`, and the scoring denominator stayed 70.
  - A run directory under `results/pilots/` was refused.

To reproduce r3 in a new directory, check out `adfb2d3` or `8479309` first, because the runner has no flag for the older retrieval policy and later code produces r4-style runs. Then run the following. The fingerprint includes package versions, so a different environment gives a new fingerprint.

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_offline.py generate --run-dir results/engineering/<new_run_id>
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_offline.py score --run-dir results/engineering/<new_run_id>
```

## Validation

Local checks ran on macOS arm64 at `adfb2d3`, in `.local/venv-prebaseline`, a
fresh environment built with
`pip install -c config/environment/constraints-aaai.txt -e ".[test,lint]"`:

- `pytest -q -ra -p no:cacheprovider`: 1197 passed, 0 skipped. The real `logs/vlm_calls.jsonl` was byte-identical before and after.
- `ruff check .`, `pip check`, `uv lock --check`, `faar-demo --help` and `scripts/experiments/registry.py check` (0 errors) passed.
- `scripts/experiments/ohr_scoring_parity.py` found 0 mismatches, and `--check-fixture` matched.

Remote CI runs on every push to PR #2. On `9e9aa90`, the commit the reviewer
saw, run 36484651758 passed both jobs. The offline tests there gave 1173
passed and 1 skipped: the smoke test that needs local assets CI does not have.
The PR records the CI result of its final commit. A report committed with the
branch cannot record the CI result of its own commit.

## Independent review

A reviewer who wrote none of this code reviewed `9e9aa90`. It ran the suite in a clean clone under a no-network sandbox, fuzzed the scorer against upstream on 548,751 pairs with 0 mismatches, and ran 50 mutation tests. It found no critical issue and no route for gold data into generation through the committed manifests.

| Finding | Disposition |
| --- | --- |
| M1: a case variant such as `results/Pilots` bypassed the frozen-directory refusal, and other shared directories were unprotected | Fixed. Case-folded check. Inside the project only `results/engineering/<run_id>/` is accepted. Tests cover case variants and `config/`, `OHR-Bench/`, `annotation/`, `logs/` and `experiments/`. |
| M2: a crafted manifest could point generation at the gt text | Fixed. The noisy-text root is pinned. `run_config.json` lists the enforced input checks. |
| M3: the brief called steps done before CI had run, and 18 answered questions had position-only retrieval that no record reported | Fixed. The brief ties acceptance to CI. Every record has `query_retrieval_tokens`, and the summary counts such questions. |
| L1: `answered_only` is not upstream's headline aggregate | Fixed. `upstream_valid_only` added and documented. |
| L2: scorer dependency versions not recorded | Fixed. They are in `score_summary.json`, with the Unicode database version. |
| L3: `.local/` ignored only locally | Fixed. It is in `.gitignore`. |
| L4: three scorer quirks and the `abstained` counter survived mutation | Fixed. Upstream-recorded fixture cases and a stronger counter test. CI runs the parity script. Two runner fingerprint terms that also survived mutation are covered by the code and manifest hashes, so they stay unchanged. |
| L5: the fingerprint omitted library versions | Fixed. They are in the fingerprint. |
| L6: exit code 1 covered both refusals and crashes | Fixed. Internal errors exit 3. |
| L7: CI cancelled superseded runs on `main` too | Fixed. It now cancels only superseded pull request runs. Pinning actions to commit SHAs was not done: the workflow uses the maintained major tags of official GitHub actions, and SHA pinning is a separate supply-chain decision. |
| L8: `AGENTS.md` commands assumed the old environment | Fixed. It points to the README steps. |

## Study brief changes

[study-brief.md](../research/study-brief.md) now does the following:

- It labels unapproved defaults as proposed defaults.
- It states the exact failure condition of the old normaliser and the verified numeric-punctuation behaviour.
- It separates page coverage from passage and evidence coverage.
- It separates a visual-request limit from matched total cost, and defines what happens when a budget runs out.
- It allows development all-repair results to inform method development, with each change logged, and requires the method to be frozen before any validation or test run.
- It describes the existing controller code as unconnected to the pilot, and the offline run as an engineering check.
- It explains the CI failure from the log and the commit history.

## Preservation

The lead hashed 3,550 files under `config/`, `OHR-Bench/data/`, `data/`,
`results/`, `artifacts/`, `annotation/`, `logs/` and `experiments/` before any
edit, and again after the last run.

- **Frozen research inputs are unchanged.** These are the source locks and split under `config/`, `OHR-Bench/data/` including `qas_v2.json` and the MinerU and gt trees, `results/pilots/ohr_dev_v1/` including annotations, `annotation/`, `data/` including `pdfs.zip`, and `logs/` including `vlm_calls.jsonl`.
- **Environment pins changed as intended.** `config/environment/constraints-aaai.txt` and `constraints-py312.txt` changed. These are dependency pins, not research inputs.
- **Records were added, not rewritten.** `experiments/registry.jsonl` gained 7 lines, and its first 7 lines are byte-identical to `e1d88a8`. `experiments/README.md` gained the new rows, and two run directories are new under `results/engineering/`.
- **Local git state is intact.** Both stashes and the tag `recovery/laptop-2026-09-22-source-only` are intact.

## Agent configuration

- **Lead, requested:** the task prompt of 2026-09-28T20:23Z asked for `claude-opus-5-5` at effort High. The continuation prompt of 21:21Z said "Keep Opus 5.5 at Medium".
- **Lead, observed:** the session metadata reported `claude-opus-5-5` at effort `high` throughout. The session tool refuses to change a session's own effort, so the lead could not switch to Medium itself. At 21:22Z the user wrote in the chat "its all good high is ok dont worry". That message is in the session transcript. The lead therefore ran at `high` for the whole task.
- **Workers 1 to 3** (environment, scoring, runner): configured by the model alias only (`model: sonnet` in the Agent tool call). No explicit model ID and no per-worker effort were set. A probe agent with the same alias reported `claude-sonnet-5-5`, which is a self-report, not an independent check. They inherited the session effort `high`. The explicit `faar-worker` definition did not load in time for them.
- **Worker 4** (review): the `faar-worker` agent definition sets `model: claude-sonnet-5-5` and `effort: high`. A probe of that definition reported `claude-sonnet-5-5`. The effective effort of any worker could not be confirmed independently, and an agent's self-report is not treated as confirmation.

## Limitations and decisions before real inference

- The rule-based extractor returns long spans: 30 of the 63 r3 answers exceed 1,000 characters. Its EM measures nothing about FAAR.
- As of 2026-09-28, local-hash retrieval gave Han-script queries no signal. The extension below records the multilingual fix. 23 of the 30 pilot documents are single pages. No repair, gate or diagnosis ran.
- The pilot has 70 questions, too few for significance or generalisation claims.
- The lock covers Linux x86_64 and macOS arm64 only.
- The local `.venv-aaai` still needs the new pins (see [README.md](../../README.md#local-checks-and-known-issues)).

These are the lead decisions from study brief section 12 that come first:

1. The answer model, its provider, the prompt and the token limits.
2. The spending limit and the cost-accounting rules.
3. The image budget and the page-selection policy for longer documents.
4. The maximum execution-failure rate for a valid run.
5. The gate threshold and where it is calibrated.
6. The repair-comparison protocol and a longer-document development sample.
7. The redistribution terms for the reimplemented OHR-Bench scoring code, and what to do about the public repository that already serves the vendored `OHR-Bench/` tree (see the [scorer provenance record](ohr-scorer-provenance.md)).

## Extension of 2026-09-29: multilingual retrieval and readiness

This section adds the work of 2026-09-29 on the same branch. The sections
above describe the state on 2026-09-28 and stay as written, except for the
dated corrections to the agent configuration and the parity count.

### Multilingual retrieval

- **The r3 problem.** In r3, 19 of 70 questions had no query token, because the retrieval tokeniser matched only `[a-z0-9%$]`. 18 of them were still answered from chunks ranked by position. An unspaced Chinese page also became one chunk, because chunks counted whitespace-separated words.
- **What changed.** `src/faar/text_units.py` defines two versioned policies:
  - Tokeniser `multilingual-v1`: NFKC, casefold, and removal of whitespace between CJK characters. Word tokens are runs of Unicode letters or digits plus `%` and `$`, and each CJK run becomes character bigrams. On pure-ASCII text it equals the old tokeniser.
  - Chunk policy `cjk-weighted-words-v1`: two CJK characters weigh one word, so a chunk holds at most 360 CJK characters at the default size. Text without CJK gets the old boundaries.
- **Scope.** The offline runner passes both policies explicitly. `HybridRetriever`, `LocalHashEmbedder` and `build_page_chunks` keep the old rules as defaults, and `graph.py`, `benchmarks.py` and `RetrievalSettings` are unchanged.
- **Tests.** `tests/test_multilingual_retrieval.py` holds synthetic multi-page Chinese, mixed-language and English documents with distractors. The failing cases were reproduced on the old code first: 8 failed there. The tests check the following:
  - the relevant passage ranks first;
  - English results are identical under both tokenisers;
  - numbers and `%`, repeated terms and symbol-only text behave as expected;
  - chunk length stays bounded, ordering is deterministic across hash seeds, and hits stay inside documents;
  - generation works with evaluation files absent.
- **Why not jieba.** jieba segments by context, so the same phrase can get different tokens in a question and a passage. Bigrams have no state and need no dictionary.

### macOS OpenMP

- **Cause.** The `faiss-cpu` and `torch` macOS wheels each bundle a different `libomp.dylib`, and both load into one process. `scripts/diagnostics/openmp_check.py` reproduces the fault in fresh subprocesses. The lead reran it with 2 trials per case and got the same outcomes as Worker B's 5-trial run:
  - With no setting, the faiss-then-torch order hangs and the torch-then-faiss order segfaults.
  - `KMP_DUPLICATE_LIB_OK` alone does not help.
  - A one-thread limit on both libraries, or `OMP_NUM_THREADS=1`, passes.
  - Pointing faiss at torch's `libomp` in a scratch copy removed the fault, which confirms the cause.
- **No dependency fix.** Every macOS faiss wheel checked (1.9.0.post1 to 1.15.1) bundles its own `libomp`, so no dependency pin fixes it.
- **Supported execution.** The test setup keeps only the one-thread limit, and `KMP_DUPLICATE_LIB_OK` was removed. With the limit disabled, the suite still segfaults. Local-hash runs and `faar-demo --help` need no setting. On macOS, real sentence-transformers retrieval needs `OMP_NUM_THREADS=1`. That path was checked with stand-in models, not the real ones. Linux, the cluster and Intel macOS were not tested.

### Scorer provenance

The [scorer provenance record](ohr-scorer-provenance.md) traces
`normalize_answer` and the exact-match comparison to the SQuAD v1.1 script, and
`f1_score` with its yes, no and noanswer rule to HotpotQA (Apache-2.0).
`has_chn_character`, the jieba branch and the -1 error return come from
OHR-Bench, which has no licence. The module was written after reading the
upstream file, so it is not a clean-room implementation. Permission for the
OHR-only parts is unresolved. The record also notes that the GitHub repository
is public and has served the vendored `OHR-Bench/` tree since `fc93503`. The
scorer's behaviour did not change. At `8479309` and after the docstring change,
the parity script reported 0 mismatches in 99,045 pairs.

### First real baseline

[Study brief section 15](../research/study-brief.md#15-first-real-baseline-proposed-protocol)
proposes the protocol and lists the decisions. Nothing in it is approved, and no
credential was obtained and no model called.

### Run r4

`2026-09-29-ohr-dev-v1-offline-engineering-r4`, code `ecaadd0`, output in
`results/engineering/2026-09-29-ohr-dev-v1-offline-engineering-r4/`, supersedes
r3. It was registered before it started. The registry attempt therefore says `dirty: false`, the state when the record was added, while `run_config.json` says `dirty: true` with `experiments/registry.jsonl` as the only dirty path. r2 and r3 have the same pattern.

| Measure | r3 | r4 |
| --- | --- | --- |
| `answered`, `no_evidence`, `execution_failed` | 63, 7, 0 | 63, 7, 0 |
| `no_evidence` reasons: `no_text_chunks`, `no_text_content`, `no_retrieval_tokens`, `no_hits` | 6, 1, 0, 0 | 6, 1, 0, 0 |
| Questions with no query token | 19 | 0 |
| Chunks | 145 | 201 |
| Evidence outside the question's document | 0 | 0 |
| Longest answer (characters) | 8,545 | 2,254 |
| Official EM, all questions | 4 of 70 | 4 of 70 |
| Official F1, all questions | 0.0960 | 0.1142 |

- **Answer changes.** 14 answers changed from r3, all on Han-script questions. The 48 non-Han questions got the same evidence and answers.
- **Settings were not tuned.** The retrieval settings were not tuned to scores. The F1 change is not evidence about FAAR: the answer step is still the rule-based extractor, whose term matching is ASCII-only, so a Chinese question gets roughly the top-ranked chunk as its answer.
- **Rerun checks.** A rerun into the r4 directory was a verified no-op. Running the new code into the r3 directory was refused (exit 1), and r3 is unchanged.

### Agent configuration on 2026-09-29

- **Lead, requested and observed:** the task prompt asked for `claude-opus-5-5` at High. The session metadata reported `claude-opus-5-5` and effort `high`.
- **Workers A to E, configured:** every worker used the agent definition `faar-worker` in the workspace `.claude/agents/`. It sets `model: claude-sonnet-5-5` and `effort: high`. No runtime record of a worker's model or effort was available beyond that configuration, and a worker's statement about itself is not treated as confirmation.

### Independent review of 2026-09-29

A reviewer (agent definition `faar-worker`) that wrote none of the overnight
code reviewed `cb5d151`. It found no critical or high issue.

- **Fuzzing.** Its differential fuzzing against the `8479309` code covered 20,000 random non-CJK pages and 50,000 ASCII strings, with 0 mismatches.
- **Chunk invariants.** 30,000 random pages were checked for chunk coverage and termination.
- **Mutation tests.** Of 9 mutants, the tests killed 8.
- **Rerun and leakage.** A byte-identical regeneration of r4 matched the committed files, and a file-access spy saw no evaluation file opened during generation.
- **Matching evidence.** `openmp_check.py` gave the same verdicts as above, and its spot checks of the cited provenance sources matched.

| Finding | Disposition |
| --- | --- |
| M1: this report still called r3 current, and its r3 reproduction command now yields an r4-style run | Fixed. r3 is marked superseded, and the reproduction note says to check out `adfb2d3` or `8479309`. Pre-r4 limitation lines are dated. |
| M2: study brief section 15.4 cited `pilot_runner.py` line numbers from `8479309` | Fixed. It cites symbols instead. |
| L1: the section 15.6 cost estimate used r3 evidence sizes and called the retrieval change in progress | Fixed. It names the r3 figures and adds the r4 figures (mean 3,190 and maximum 6,252 characters, recomputed by the lead). |
| L2: the provenance record missed `f1_zh`, which uses jieba, in OHR-Bench's init commit, and cited an uncommitted draft path | Fixed in the record. The module docstring says these parts come from OHR-Bench, which stays true, so it was not changed. |
| L3: `multilingual-v1` splits Devanagari, vowelled Arabic and pointed Hebrew at combining marks | Accepted as a documented limit in study brief section 11. No code change, because the pilot has none of these scripts and a code change would alter the r4 fingerprint. |
| L4: removing the katakana prolonged sound mark from the CJK range survived the tests | Fixed. A test case for `コンピューター` was added, and the lead confirmed that the mutant now fails it. |
| L5: r4's registry attempt says `dirty: false`, while its `run_config.json` says `dirty: true` | Explained above, with no change. Both values are true at their own moments. |

## Extension of 2026-09-29 (afternoon): answer-model execution path, offline only

No live research-model request, connectivity probe or credential load took
place. Every run below used the in-process fake provider or a mocked HTTP
transport.

### What exists

- **Run driver.** `src/faar/live_runner.py` and `scripts/experiments/run_pilot_live.py` provide `dry-run`, `run`, `status`, `reconcile`, `export` and `score`. Study brief sections 15.4, 15.7, 15.8 and 15.15 describe the behaviour. `src/faar/live_contract.py` holds the shared types.
- **Prompt.** `src/faar/answer_prompt.py` builds the draft prompt `faar-answer-draft-v1`, which is not approved. Evidence blocks sit between tilde fences, and the system message says that instructions inside the evidence are document text.
- **Providers.** `src/faar/answer_providers.py` has a scripted `FakeProvider` with 13 behaviours, from success and abstention to unknown outcomes and a crash after sending. It also has an `OpenAIChatProvider` adapter that is tested only against `httpx.MockTransport`. The adapter refuses a client with SDK retries enabled. OpenAI documents no idempotency key for Chat Completions (checked 2026-09-29), so an unknown outcome is never resent automatically.
- **Budget and retries.** `src/faar/request_budget.py` and `src/faar/retry_policy.py` keep a conservative ledger in integer micro-units. A dispatch goes ahead only if measured cost, reserved cost and its own upper bound fit within the safety ceiling. The retry policy retries only failures whose request never left the machine or was rejected by the provider.

### Verification

- **Tests.** In the lead's checkout the suite gave 2,038 passed and 2 skipped at `0e992c2`. The 2 skips are an optional tokenizer cross-check that needs local vocabulary files. A clean checkout also skips `tests/test_b0_one_doc_smoke.py:227`, which needs untracked smoke assets, so the reviewer saw 2,037 passed and 3 skipped. Mutation checks by the workers are recorded in their notes.
- **Dry run.** The dry run on the frozen pilot covered 70 questions: 63 to send and 7 skipped (6 `no_text_chunks`, 1 `no_text_content`). The largest input bound was 8,237, below the 12,000 limit. The worst case over 3 attempts per request was $2.94 at option A's unapproved rates. The run made zero provider calls. The output is in `.local/work/prompt-preview/` in the lead's checkout, which is not committed and has no backup elsewhere.
- **Registered run.** `2026-09-29-ohr-dev-v1-fake-provider-r1` ran at code `7250dad` in `results/engineering/2026-09-29-ohr-dev-v1-fake-provider-r1/`.
  - It completed with 63 answered, 7 no_evidence and 0 execution_failed, from 63 attempts, all measured.
  - Its cost was 0.228869 simulated USD against a 5 simulated USD ceiling. The fictional rates are numerically equal to option A's.
  - Its scores are of fictional answers and measure nothing.
  - `requests.jsonl` holds the full prompts. It is git-ignored and is regenerated byte for byte from the inputs.
- **Lead demonstrations** on the real pilot in scratch, not registered:
  - A process killed after dispatch left its question `needs_reconciliation`. Three later invocations did not resend it, and the other questions finished.
  - After `reconcile ... allow_new_attempt`, the question was sent exactly once more. After `mark_failed` it was not sent again.
  - A retryable error was retried within 3 attempts, with every attempt kept.
  - A complete run rerun added only invocation events.
  - A changed fake script was refused as an identity change before any dispatch.
  - A 0.05 ceiling stopped the run as `budget_limited` with 56 unserved questions. The raise needed an authorisation note and was recorded as `from 0.05 to 5.0`.
  - A second process could not take the run lock.
  - Live mode without its enablement refused before creating anything.

### Limits

- The prompt, the answer model, the spending ceiling and the input-limit rule are not approved.
- The live adapter has never met the real API.
- Provider-side exactly-once delivery is not available. The path guarantees safe local resume only.
- Query tokens do not show that the evidence is relevant, and the pilot is mostly single-page.
- The macOS OpenMP limitation, the open scorer licence question and the unfinished human inspection still stand.

### Independent review of the answer-model path (2026-09-29)

A reviewer with the `faar-worker` definition, who wrote none of this code,
reviewed `b87a48e`. It found no critical defect, and no path to a double spend,
a gold-data leak or a ceiling breach under the stated assumptions. Two fix
workers (`faar-worker`) and the lead fixed the accepted findings. Every fix had
a failing test first, except where the table says otherwise.

| Finding | Disposition |
| --- | --- |
| H1: an interrupt during an event append duplicated a sequence number, and the log became unloadable with a paid response unreachable | Fixed. The in-memory state advances as soon as the bytes are written. A cut-short write is truncated, and the interrupt handler re-reads the log before it recovers open attempts. |
| H2: every prompt contained the document name, because chunk IDs carry it | Fixed. The header shows a document-free label such as `p2-c2`, and the new template SHA-256 is `52044de2...`. On the frozen pilot, 0 of 63 prompts now contain a `doc_id`. |
| H3: the configured endpoint was ignored, and `OPENAI_BASE_URL` could redirect live requests with the key | Fixed. The client uses the configured base URL. Live mode refuses the SDK's redirecting environment variables, and the run records the client's base URL. Proxy variables remain unchecked (study brief 15.4). |
| M1: `reconcile` and `export` changed scored runs, and `score` ran on unfinished runs | Fixed. `score` needs a `complete` run. After scoring, writers refuse, and `export` only confirms identical files. |
| M2: `export`, `reconcile` and `score` created `run.lock` in any directory | Fixed. The run-directory rules and `run_config.json` are checked first. |
| M3: a 200 reply without an answer message was saved as a final empty answer | Fixed. It is an unknown outcome (`malformed_response`). |
| M4: a failed question could not be retried | Fixed with `reopen`, which gives up to 3 further attempts and keeps the earlier ones. |
| M5: a systematic fault could turn every question into an unknown outcome | Fixed with a circuit breaker after 3 consecutive such attempts. Releasing reserved cost on reconciliation was rejected, because a free-text note cannot prove that nothing was billed. |
| M6: `valid_baseline` ignored a returned-model mismatch and anomalies | Fixed. The summary lists blockers and says that the flag checks mechanical completeness only. |
| L1: a gateway 504 was retried | Fixed. It is now an unknown outcome. |
| L2: request parameters were checked against a denylist | Fixed. They are checked against an allowlist of six keys. |
| L3: the attempt cap was not enforced across stopped invocations | Fixed. |
| L4: `fsync` on macOS does not flush the drive cache | Fixed. The runner also calls `F_FULLFSYNC`, best effort. This was not tested on real hardware. |
| L5 to L11: ignore rule, status wording, test counts, code freeze, the note as a speed bump, socket wording, registry placeholder | Fixed in the files and documents. The registry placeholder stays corrected by an appended line, since the registry is append-only. |

**The successor run is `2026-09-29-ohr-dev-v1-fake-provider-r2`** (code `a6935aa`), and it supersedes r1.

- It completed with 63 answered, 7 no_evidence and 0 execution_failed, from 63 attempts, all measured, with 0 reserved.
- Its cost was 0.221480 simulated USD.
- A rerun before scoring added only invocation events. After scoring, `run` was refused and `export` was a verified no-op.
- The new local preview is `.local/work/prompt-preview-v2/`. Its largest input bound is 7,920, and its worst case is $2.85 at option A's unapproved rates.
- The lead reran the demonstrations on the fixed code: an unknown outcome was not resent, reconcile and reopen each gave exactly one further attempt, the incompatible change, the budget raise without a note, the second process and live mode without enablement were all refused, and the budget stop was recorded. Reopening an answered question was refused.
- The suite at `a86f27e`, in the lead's checkout, gave 2,177 passed and 2 skipped.

### Fix verification of 2026-09-29

A second reviewer (`faar-worker`), who wrote none of the code, reran the first
reviewer's reproductions against `11ce1a4`. All thirteen accepted findings
were confirmed fixed, L4 on a best-effort basis. Its own dry run and fake run
matched the lead's files byte for byte. It found nothing above Low.

| Finding | Disposition |
| --- | --- |
| N1: help and brief gave the circuit breaker's state as `stopped` only | Fixed. The state is `needs_reconciliation` when a counted attempt had an unknown outcome, and `stopped` otherwise. |
| N2: the brief's retry row still said "5xx" after 504 became unknown | Fixed. |
| N3: `status` advised `reopen` on a scored run | Fixed. |
| N4: Cloudflare origin timeouts 522 and 524 were retried | Fixed. They are unknown outcomes, like 504. HTTP 499 was already a non-retryable rejection. |
| N5: a network outage failed every question without tripping the breaker | Fixed. Any failure that ends a question counts, and only a saved response resets the count. |
| N6: the payload of a malformed 200 reply was dropped | Fixed. It is kept, bounded to 20,000 characters, in the `outcome_unknown` event. |
| N7: a crash between the two score-file writes leaves a half-scored run | Not fixed. The window is two consecutive atomic writes, and recovery means removing one file by hand. |
| N8: a refused `reopen` or `reconcile` can still append recovery events for other questions | Kept. It is the same honest bookkeeping as in `reconcile`. |
| N9: some endpoint spellings pass the parser and fail only at client build | Kept. It fails closed before any request. |

**The successor run is `2026-09-29-ohr-dev-v1-fake-provider-r3`** (code `578a613`), and it supersedes r2.

- It completed with 63 answered, 7 no_evidence and 0 execution_failed, at 0.221480 simulated USD, with 0 reserved.
- It differs from r2 only in request, attempt and response IDs, which derive from the identity hash.
- The suite at `578a613`, in the lead's checkout, gave 2,182 passed and 2 skipped.

