# Live-baseline readiness review (2026-09-30)

## Corrections (2026-09-30, later)

This note lists what changed after the first version of this review. It changes no frozen question, gold answer, pilot selection or annotation.

- **Agent review counts.** The agent review summary said the sent evidence holds the answer fully in 15 of 17 sent cases. The table rows give 13 fully, 2 partly (cases 7 and 14) and 2 not at all (cases 17 and 20). [The agent review](ohr-dev-v1-agent-review-2026-09-30.md) is corrected. This review did not repeat the wrong number.
- **Wording.** Five interpretations were reworded in sections 3 and 6:
  - noisy OCR text that exists is not shown to be correct, and the `ok` status only says text exists;
  - a gold page among the retrieved pages does not show that the needed passage was retrieved;
  - evidence missing from the first prompt does not mean a case tests only abstention, because it also bears on later retrieval or OCR repair;
  - agent concerns about gold answers are provisional and not adjudicated corrections;
  - the 20 inspection cases are a purposive diagnostic sample and give no estimate of dataset-wide error rates.
- **Audit scripts.** The audit scripts are now in [scripts/audits/](../../scripts/audits/README.md), with offline tests in `tests/test_audits.py`. Sections 1, 5 and 7 point to them. Run against the c9b7ad5 dry run, they reproduce these figures: 70 records (63 sent, 7 skipped), 243 evidence blocks, input bounds of 1,550, 6,563, 7,920 and 347,702 in total, cost bounds of $0.949909 and $2.849727, the expected cost of $0.24 (range $0.19 to $0.31), the Fast-tier bounds ($1.614844 and $4.844532), the stop after 57 of 63 questions (58 with 5 output tokens) when every question has two rejected attempts under the $2.00 ceiling, the 8 live-check ids and the derived manifest hash `a1e147c0...`.
- **Safeguards added after this review.** The section below, "Update after the safeguards", records what the code at `ca26e31` now enforces and which findings it closes. Sections 2 to 5 below still describe the runner at `c9b7ad5`. Where they conflict with the update, the update is current.

## Update after the safeguards (code `ca26e31`)

These are facts about the code at `ca26e31`, checked offline with the fake provider and `httpx.MockTransport`. No live request was made. Nothing here approves the model, the prompt, the ceilings or the data handling.

- **Storage.** Every request sends `store: false`. The CLI refuses provider-side storage, and the storage policy is part of the run identity. This removes the code part of B1. The data-handling decision in B1 remains open: `store: false` does not remove OpenAI's 30-day abuse-monitoring copy, and it does not settle whether the dataset terms allow sending the documents.
- **Service tier.** Every request sends `service_tier: "default"`, the Standard tier, so a project setting cannot change it. The price table must declare that tier. A saved response whose returned tier is not `default` stops the run before the next dispatch, and so does a missing tier or an unusable reply that names another tier. A restart builds no provider. Such a response keeps its usage, but its cost is left out of `cost.measured` and reported under `cost.unverified_tier`, labelled as not an actual-cost claim. This closes the code part of B2. A dashboard check of the Project Service Tier is still sensible, and the live check prints the returned tiers.
- **Transport.** The live client ignores proxy variables, macOS system proxies and `SSL_CERT_FILE` / `SSL_CERT_DIR` (`trust_env=False`), and it never follows redirects. An injected client that trusts the environment, follows redirects or routes through its own proxy is refused. A network that reaches OpenAI only through a proxy gets connect errors, and nothing is sent. The `env -u` wrapper in the commands is no longer needed.
- **Run kind and subsets.** A live run needs `--run-kind`. The identity records whether the run's manifest is byte-identical to the frozen `results/pilots/ohr_dev_v1/runtime_manifest.json`. A subset, or a same-sized altered selection, is not canonical. A `development_pilot` on a non-canonical manifest is refused. An `engineering_check` never has `valid_baseline: true`. `valid_baseline` is a mechanical eligibility flag, not scientific approval.
- **Successor fake runs.**
  - `2026-09-30-ohr-dev-v1-fake-provider-r5` supersedes r4. It is complete (63 answered, 7 no_evidence, 0 execution_failed, 0.221480 simulated USD), with every returned tier `default`, a canonical manifest, and `valid_baseline: false`.
  - `2026-09-30-ohr-dev-v1-live-check-subset-fake-r1` rehearses the live check on the 8-question subset: 6 answered and 2 no_evidence. It is `engineering_check`, not canonical, and ineligible, with three blockers.
  - Against r4 and the previous preview, only request, attempt and response ids changed. Question text, evidence, prompts, bounds and scores are identical. The ids changed because the identity gained the new fields and the code changed.
- **Audit outputs.** Compact outputs with commands and hashes are in [results/audits/2026-09-30-readiness/](../../results/audits/2026-09-30-readiness/README.md). At `ca26e31` they reproduce every figure in this review.
- **Decisions in section 4 after this update.**
  - Decision 2 is reduced to an optional dashboard check.
  - Decision 3 is done in code. The code freeze still needs the lead's approval.
  - Decision 10 no longer needs a registry workaround: the check runs as `engineering_check`.
  - All other decisions stay open: data handling, model, prices, ceilings, rate limits, prompt, input limit and retrieval, human inspection and licensing.
- **Live-check commands (NOT EXECUTED), current version.** Run them from the repository root. They replace the commands in section 5.

```bash
RUN=YYYY-MM-DD-ohr-dev-v1-live-check-r1
CFG=.local/work/provider-config.<approved>.json
MAN=results/audits/2026-09-30-readiness/live-check-selection/runtime_manifest.livecheck-v1.json
shasum -a 256 "$MAN"
```

The hash must be `a1e147c0dbe47274fa7bdbc381035f54490ac8b3b567ea54fdcf1807ad23a0c3`. The approved config must include `"service_tier": "default"` in `prices`.

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py dry-run --runtime-manifest "$MAN" --provider-config "$CFG" --out .local/work/livecheck/dry-run
```

This offline dry run must report 8 questions, 6 to send and a one-attempt bound of 0.087581.

NOT EXECUTED. The next command spends money. `OPENAI_API_KEY` is set by the lead.

```bash
FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY .local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py run --mode live --run-kind engineering_check --runtime-manifest "$MAN" --provider-config "$CFG" --run-dir "results/development/$RUN" --run-id "$RUN" --safety-ceiling 0.15
```

```bash
.local/venv-prebaseline/bin/python scripts/audits/verify_live_check.py "results/development/$RUN"
```

Score only if the verifier reports no FAIL, using the subset evaluation manifest described in section 5.

Status: a review for the research lead, to be checked by an independent reviewer. It makes no decision and it is not an authorization to spend. No paid request, connectivity probe or credential load took place. No human annotation was written.

The review asks one question: are the selected data, the retrieved evidence, the prompts and the proposed execution settings fit for the first real-model development baseline on `ohr_dev_v1`? It does not test whether diagnosis-selected repair beats simpler recovery.

**Verdict at `c9b7ad5`** (see the update above for the current state). The prompts, the retrieval records, the cost bounds and the runner's accounting pass every offline check. Two findings block paid requests until the research lead acts on them.

- **B1.** Nobody has decided how the transmitted documents are handled, and the runner cannot send `store=false`. This blocks every paid request, including the live check.
- **B2.** The runner cannot see which service tier bills the requests. A free dashboard check clears it for the live check. The lead should decide on a code safeguard before the full pilot.

In this report, "the lead" is the human research lead, and "the coordinating agent" is the model that wrote the report. Once B1 is decided and the dashboard check for B2 is done, the repository is ready for the lead to approve a small live check (section 5), and after that the full pilot.

## 1. Ready items

- **Prompts: all seven audit checks pass on the 70 prepared records** (63 sent, 7 skipped).
  - Every question matches its manifest text, and every evidence block's `doc_id` matches the question's document.
  - All 243 evidence blocks match the MinerU page text exactly, with the right rank, page and label. All `evidence_sha256` and `prompt_sha256` values recompute.
  - Blocks: 0 empty, 0 markup-only, 0 duplicated and 0 fence collisions.
  - The full `doc_id`, its basename and its folder prefix appear 0 times in any prompt.
  - An audit hook shows that the dry run opens only the runtime manifest, the provider config and the 30 MinerU files. It never opens `evaluation_manifest.json`, the clean reference text, annotations or `selection_record.json`. Importing the modules also reads `config/model_revisions.json` (through `faar.settings`), which is not part of the dry run itself. `scripts/audits/audit_prompts.py --trace-dry-run` reproduces the trace and lists that import-time read separately.
  - No gold answer of 4 or more characters appears outside the OCR evidence. The 15 gold strings found inside evidence blocks are legitimate OCR text.
- **Stable construction.** Dry runs with different `PYTHONHASHSEED`, working directory, `TZ` and `LC_ALL` values give byte-identical files. The dry run at `c9b7ad5` (not later code, whose identity changes every `request_id`) equals `.local/work/prompt-preview-final/` byte for byte. `audit_prompts.py --compare-dry-run` checks that byte equality for two dry-run directories. The runs under other environment values were made by hand and are not scripted.
- **Size accounting.** The input bound is `16 + sum over the two messages of (UTF-8 bytes + 4)`, from `src/faar/request_budget.py:232-243`.
  - It recomputes for all 63 prompts: minimum 1,550, median 6,563, maximum 7,920 (`65ea29d1`), total 347,702. None exceeds the 12,000 limit.
  - `max_tokens` = 128 is sent on every request (`src/faar/answer_providers.py:558`).
- **The 7 skipped questions** have true reasons.
  - Six (`9cb9f9aa`, `9d99eb49`, `9e95f8bc`, `a7005e84`, `ecbf8bf8`, `eccc944b`) have MinerU pages that are empty strings (`no_text_chunks`).
  - `ecccdf67` has 5 characters of heading marks only (`no_text_content`).
  - None is dropped. They export as `no_evidence`.
- **Instruction-like text.** No evidence text addresses a model. There are 19 weak hits in 14 questions, such as "必须", "you should" and "Mounting Instructions". All sit inside the tilde fences, and the system rule covers them.
- **Model and parameters, checked on 2026-09-30.**
  - `gpt-4o-2024-11-20` is listed as a GPT-4o snapshot and is absent from the deprecations table. The model page lists Chat Completions as supported, a 128,000-token context and 16,384 maximum output tokens.
  - `temperature: 0` is valid.
  - `max_tokens` is deprecated in favour of `max_completion_tokens` but still accepted for non-o-series models.
- **Prices, checked on 2026-09-30.** The Standard row for `gpt-4o` on the pricing page is $2.50 input, $1.25 cached input and $10.00 output per 1M tokens, the same as the candidate config. The model page agrees.
- **Retry and reconciliation.** The table was checked by feeding constructed SDK and httpx errors through `classify_openai_error` and `RetryPolicy`. It matches study brief section 15.7:

  | Error | Outcome | Handling |
  | --- | --- | --- |
  | Connect error, connect or pool timeout | `not_sent` | Retried up to 3 attempts. Reserves nothing |
  | 408, and 503 with `server_is_overloaded` | `rejected` | Retried up to 3 attempts. Backoff 2.47 s, then 4.57 s |
  | 401, 403, unknown model, quota or spend 429 | `rejected` | Stops the run |
  | 400, 422, other 4xx | `rejected` | Fails the question |
  | 429 rate limit, 409, 500, 502, other 503, 504, 522, 524, any 3xx, read timeout, lost connection, malformed 200 | `unknown` | Waits for `reconcile`. Its reservation stays |

- **Permanent safety stops.** There are three triggers, and each is final for its run directory:
  - `input_bound_exceeded`;
  - `returned_model_mismatch`, an exact string match (`src/faar/live_runner.py:2414-2422`);
  - `ledger_anomaly` (`src/faar/live_runner.py:2220-2276`).

  Only a successor run with corrected code or config continues.
- **Subset runs work with no code change.** A derived runtime manifest outside `results/pilots/` is accepted (`src/faar/pilot_runner.py:225-318`, `358-395`). The run identity records its hash.
  - Offline, a subset gave prompts, evidence hashes and bounds identical to the full run for all 8 questions.
  - A fake subset run completed with one dispatch per sendable question.
  - `score --evaluation-manifest <subset>` scored it. The default `score` refuses a subset: "62 missing from predictions".

## 2. Blocking findings

### B1. Data handling for the transmitted documents is undecided, and the runner cannot send `store=false`

- **Evidence.**
  - The adapter builds its request at `src/faar/answer_providers.py:555-566`. It sends `store` only when `tag_attempts` is on, and then it sends `store=True`. The live builder never turns `tag_attempts` on.
  - `store` is not in `ALLOWED_OPENAI_PARAMS` (`src/faar/live_contract.py:47`), so a config cannot add `store=false`.
  - OpenAI's pages, read 2026-09-30, disagree about the default:
    - The migration guide says "Chat completions are stored by default for new accounts. To disable storage in either API, set `store: false`." (https://developers.openai.com/api/docs/guides/migrate-to-responses, lines 132 and 202 of the Markdown page).
    - The data-controls table gives `/v1/chat/completions` an application-state retention of "None, see below for exceptions" (https://developers.openai.com/api/docs/guides/your-data, line 72).
  - The same table gives 30 days of abuse-monitoring retention and no training use by default. `store: false` does not remove that copy. Only an approved Zero Data Retention or Modified Abuse Monitoring control excludes customer content from it.
- **Why it blocks.** Every paid request, including the live check, sends OHR-Bench document text to OpenAI.
  - Each request is at most 7,896 bytes, counting the system message, the question and the block headers.
  - In total the pilot sends 346,190 bytes from 23 documents.
  - The OHR-Bench README limits the documents to research use.
  - Study brief section 15.13 item 13 treats provider-side storage as something `tag_attempts` switches on, but the account may store completions with it off.

  This is a data-handling decision for the lead. It covers decision 12 as well, and no code or documentation settles it.
- **Options for the lead.**
  - Accept the transmission, with the 30-day abuse-monitoring copy, and possible storage.
  - Confirm in the dashboard whether stored completions are on for the project.
  - Approve a small reviewed change that sends `store=False` whenever `tag_attempts` is off. The code change alters the code digest, so it must land before the code freeze (section 4).
  - Apply for Zero Data Retention or Modified Abuse Monitoring, which needs OpenAI's approval.

### B2. The service tier is outside the runner's control, so the ledger can under-count

- **Evidence.**
  - The adapter never sends `service_tier`, and the allowlist refuses it.
  - The Fast-mode guide, read 2026-09-30 (https://developers.openai.com/api/docs/guides/priority-processing), says that when a project's **Project Service Tier** is set to Fast, "Requests that don't specify a `service_tier` then default to Fast mode."
  - The pricing page's Fast row for `gpt-4o` is $4.25, $2.125 and $17.00, which is 1.7 times Standard.
  - The ledger prices usage at the config's Standard rates. The safety check does not read the response's `service_tier` field (`grep service_tier src/faar/*.py` finds only comments).
- **Consequence.** On a Fast project every recorded cost would be 1.7 times too low, and no anomaly would fire. A $2.00 ceiling would then allow about $3.40 of real spend. The full-pilot bounds at Fast rates are $1.614844 for one attempt and $4.844532 for three. The error is bounded at 1.7 times, and live-check criterion 8 would show it after the fact.
- **What clears it.**
  - Before the live check, a free dashboard check confirms that the Project Service Tier is the default. At the $0.15 live-check ceiling, a Fast project would mean at most about $0.26 of real spend.
  - Before the full pilot, the lead should also decide whether to approve a reviewed change that adds a returned `service_tier` other than `default` (or absent) to the safety-stop conditions, next to `returned_model_mismatch`.

## 3. Nonblocking limitations

### Execution path

- **Proxy and certificate variables are honoured.**
  - `build_live_provider` uses `openai.DefaultHttpxClient(follow_redirects=False)` (`src/faar/live_runner.py:1614`), and its `trust_env` is `True`. Checked locally with openai 1.68.2 and httpx 0.28.1.
  - `HTTPS_PROXY`, `ALL_PROXY`, `SSL_CERT_FILE` and similar variables would route the key and the prompts through a proxy. Only the four `OPENAI_*` names are refused (`src/faar/live_runner.py:159`).
  - No such variable is set in this session's environment, but the lead's shell may differ.
  - The commands in section 5 clear these variables with `env -u`. That does not cover a proxy set in the macOS system network settings, which httpx also reads when `trust_env` is on. None is set on this machine now.
  - The smallest code change is `trust_env=False` at line 1614, which ignores both sources.
- **A complete subset run reports `valid_baseline: true`.**
  - The blocker list (`src/faar/live_runner.py:2461-2486`) has no manifest check, and a subset keeps `pilot_id: ohr_dev_v1`. Only the manifest hash and the question count show that the run is not canonical.
  - The smallest change is about six lines in `summarise_run`: add a blocker when the runtime manifest hash differs from `results/pilots/<pilot_id>/runtime_manifest.json`.
  - Until then, the run id and the registry record must carry the label.
- **The live `kind` is hard-coded as `development_pilot`** (`src/faar/live_runner.py:1696`). A 6-question transport check is closer to `engineering_check` (`experiments/README.md`: "kind follows what the run measured").
- **Rate limits can stall the full pilot.**
  - The gpt-4o model page lists Tier 1 at 30,000 TPM and Tier 2 at 450,000 TPM, and the driver does not pace requests.
  - Sent back to back with replies of one to two seconds, the pilot's roughly 92,000 input tokens arrive at about 45,000 to 90,000 tokens a minute. That estimate is an assumption, not a measurement.
  - On Tier 1, 429 responses are likely. Each one is an `unknown` outcome that needs `reconcile`, and three in a row trip the circuit breaker. That costs the operator time and holds reservations, but it adds no spend beyond what was sent.
  - The live check is too small to reach the limit. The account's tier is unknown.
- **Unverifiable from public documentation:**
  - account access to the snapshot;
  - the returned `model` string. No page states that a dated request returns the same string, and a different string stops the run after its first paid response;
  - the account's `store` default;
  - any project spend limit.

### Data and evidence (from the agent review and the prompt audit)

The 20 inspection cases are a purposive diagnostic sample. The counts taken from the agent review describe those cases and estimate no rate for the dataset or for the 70 pilot questions.

- **Gold answers.** The agent review of the 20 inspection cases ([ohr-dev-v1-agent-review-2026-09-30.md](ohr-dev-v1-agent-review-2026-09-30.md)) raises a concern about the gold answer in 5 cases (1, 4, 7, 14, 17) and is unsure about case 2. These concerns are provisional agent judgement. A person has not adjudicated them.
  - Four of the five are sent to the model.
  - Two examples. Case 14's page attributes the quoted aim to three people and the gold names one. In case 17 the gold appears to fold the "18" of the Dublin 18 postal district into the phone number.
  - The frozen QA file stays unchanged. These questions stay in the denominator, and a person should confirm or reject each concern.
- **Retrieval.** Case 20 is a retrieval miss. The needed definition is in `p0-c3` and `p0-c4`, neither was sent, and no sent chunk contains "working hours".
  - Over all 63 prompts, 61 include every gold page. `8e391133` and `b89d6523` miss one.
  - A gold page among the retrieved pages does not show that the needed passage was retrieved. Case 20 shows it: its gold page 0 is present through `p0-c0`, and the needed chunks are not. Page coverage overstates evidence coverage.
- **Degenerate OCR is sent as evidence.** In case 17 the MinerU text is a hallucinated formula and a repetition loop, and `ocr_condition` still counts the page as `ok`. That status says only that text exists. It does not show that the text is correct. At least 6 sent prompts lack the answer in their evidence. The prompt audit found 4 (`8e391133`, `ecb5f40d`, `a1ab17c0`, `a12ab315`), and the agent review adds cases 17 (`a0ec729f`) and 20 (`33a1d1c2`). In the first prompt, the model can only abstain or guess on these. That does not make them abstention tests, because the missing evidence also bears on later retrieval or OCR repair.
- **A control character in a frozen question.** `6ac85f42` sends U+0007 where the source had the `a` of `\alpha`. The vendored `OHR-Bench/data/qas_v2.json` contains 12 `\u0007` escapes. This is an upstream data defect, logged and not edited.
- **Category coverage.**
  - Case 13 was selected for reading order, but the misordering does not touch its answer chunk.
  - Cases 6 and 7 share identical evidence.
  - 11 of the 17 sent inspection cases (4, 6, 7, 8, 10, 11, 12, 16, 17, 18 and 19) send every chunk of their document, so retrieval cannot fail there (study brief section 15.12).

### Prompt weaknesses and proposed changes

These are proposals only. Each changes the template hash and starts a new run.

1. **Page and chunk labels.** The block header mixes a 1-based page with a 0-based label (`page 3, chunk p2-c2`). Proposal: drop the label, or number blocks within the page.
2. **Output limit.** 128 output tokens is tight for the two Han references above 64 characters (`3e896c64`, `843f2774`). Proposal: raise the limit to 192 or 256. That adds $0.040 or $0.081 to the one-attempt bound, and $0.121 or $0.242 to the three-attempt bound.
3. **Rule conflict.** Rule 3 (`Yes` or `No`) and rule 5 (answer in the evidence's language) conflict for a Han yes/no question. The sent pilot has none. Proposal: say which rule wins before a set that has such questions.
4. **Abstention parse.** `NO_ANSWER` needs an exact match, so a reply of `NO_ANSWER.` counts as an answer. Proposal: keep the rule, and read the first replies for near misses.
5. **Short answers.** The "shortest answer" rule costs F1 on sentence-length references. Proposal: keep it. The development run measures this trade-off (study brief section 15.7).
6. **Markup in answers.** 46 of 243 blocks carry LaTeX or table markup, so answers may copy it. Proposal: check the scorer's normalisation. Do not clean the evidence.

### Documentation drift

Study brief sections 15.6 and 15.7 still quote a largest bound of 8,237 and a worst case of $2.94. The current dry run gives 7,920 and $2.849727. Section 15.5 says the pricing page does not list text rates for `gpt-4o`, but on 2026-09-30 it does.

## 4. Decisions required from the research lead

1. **Data handling (B1).** Decide whether OHR-Bench text may be sent to OpenAI under the dataset terms, with the 30-day abuse-monitoring copy. Then accept storage, confirm that it is off, or approve sending `store=False`.
2. **Service tier (B2).** Confirm the Project Service Tier, and decide whether to approve the safety-stop change.
3. **Code changes.** Say which of the proposed changes to make before the freeze: `store=False`, the service-tier check, `trust_env=False`, and the subset blocker. After that, freeze the code. Every `src/faar/*.py` file is in the run identity, so the live check and the canonical run must use the same frozen commit, or the check does not test the code that spends the pilot budget. Changed code also needs a successor fake run (r5) and a new dry run.
4. **Answer model.** `gpt-4o-2024-11-20` is proposed. Confirm that the account can call it.
5. **Prices.** Approve the price source, and update `source_date` in the config. Prices are part of the run identity.
6. **Ceilings.** $0.15 is proposed for the live check and $2.00 for the full pilot, so the total exposure is the sum, $2.15. The lead may also set a monthly project spend limit in the dashboard, which works independently of the runner. When it is reached, OpenAI returns 429 `project_spend_limit_exceeded` (error-codes page), and the runner stops the run on that code (`_QUOTA_CODES`, `src/faar/answer_providers.py:644-650`).
7. **Rate limits.** Keep the conservative 429 handling, or decide on pacing or retry once the usage tier is known.
8. **Prompt.** Approve the draft prompt `faar-answer-draft-v1`, or pick changes from section 3.
9. **Input limit and retrieval.** Keep or replace the input-limit rule, and approve the retrieval settings, knowing about the case 20 miss.
10. **Live-check setup.** Approve the live-check selection rule and its registry kind (`engineering_check` recommended, with the hard-coded `development_pilot` noted in `limitations`).
11. **Human inspection.** Arrange it (section 6) and decide whether a second person labels blind. Decide whether to create the versioned annotation form.
12. **Licensing.** Decide the release scope and the scorer licence. The dataset-terms question is part of decision 1.

## 5. Proposed small live check (NOT EXECUTED)

### Selection rule `faar-live-check-v1`

The rule uses runtime fields only: question text, evidence text, page count and input bound. It never reads answers or scores. The rank of a question is `sha256("faar-live-check-v1|" + question_id)` in hex, ascending. The strata below run in order, and each one draws only from questions not yet chosen.

| Stratum | Pool | Count | Selected |
| --- | --- | --- | --- |
| L | Sendable, the largest input bound | 1 | `65ea29d1-adae-4e45-a036-e74be34d7804` (bound 7,920) |
| H | Sendable, question contains Han characters | 2 | `84403b37-b13b-4d4e-a368-f7d2d9732b86`, `ecd44593-9f98-4aa0-9890-46dbf35c54ea` |
| M | Sendable, no Han in the question, document of 3 or more pages | 1 | `b7ad5f22-7e21-476c-9933-ea8fb579e050` (5 pages, blocks from 5 pages) |
| E | Sendable, no Han in the question or the evidence, single page | 2 | `50e1fdda-86c3-4047-9758-9cc633f33b00`, `76a3d0f7-998d-476a-aac4-55c92f7a8e02` |
| S | Skipped: lowest-ranked `no_text_chunks`, and the only `no_text_content` | 2 | `ecbf8bf8-e836-4b50-8ca8-ebc629deb1d7`, `ecccdf67-bc03-447c-a990-d5989185f21a` |

- **Result.** The rule gives 8 questions from 8 documents: 6 sent and 2 skipped.
- **Verification.** The coordinating agent and the independent reviewer each re-derived the selection from `prepared_requests.jsonl`, and both got the same 8 IDs.
- **Derived manifest.** `.local/work/livecheck/runtime_manifest.livecheck-v1.json`, sha256 `a1e147c0dbe47274fa7bdbc381035f54490ac8b3b567ea54fdcf1807ad23a0c3`. It is an exact subset of the frozen runtime manifest, in manifest order, with identical document entries and top-level fields. The frozen pilot is untouched.
- **Regeneration.** `scripts/audits/select_live_check.py --dry-run-dir <dir> --out .local/work/livecheck` applies the rule and writes the ids and the derived manifest. Run on the `c9b7ad5` dry run, it gives the same 8 ids and a manifest that is byte-identical to the scratch one. `scripts/audits/check_subset.py --derived <manifest>` verifies the exact-subset property and prints the sha256, the question count (8) and the digest of the question ids (sha256 of the ids in manifest order joined by a newline, `75ed63bf...`).

### Cost of the check

The figures below use $2.50 input and $10.00 output per 1M tokens. `scripts/audits/cost_and_ceiling.py --subset-ids <ids file> --ceiling 0.15` reproduces them and states every assumption of the estimate as an argument.

| Quantity | Value | Kind |
| --- | --- | --- |
| Expected | $0.021, range $0.018 to $0.029 | Estimate, from the heuristic token count and 5 to 40 output tokens |
| One attempt per question | $0.087581 | Bound, from the UTF-8 byte bound and 128 output tokens |
| Three attempts per question | $0.262743 | Bound |
| Proposed ceiling | $0.15 | Lead decision |

At $0.15, the check completes if each question needs at most one extra reserved attempt. Two rejected attempts per question would stop it after 3 of the 6 sent questions. A larger ceiling would only pay for retries in a check whose job is to show faults.

### Separation from the canonical run

- **Run directory.** `results/development/<YYYY-MM-DD>-ohr-dev-v1-live-check-r1/`. `refuse_run_directory` allows a live run inside the project only there, and `requests.jsonl` and `run.lock` are git-ignored.
- **Distinct ids.** Attempt and request ids derive from an identity that includes the manifest hash. No code reads another run's directory, and the canonical run gets its own run id and directory.
- **No reuse.** Never copy live-check responses into the canonical run. The canonical run pays for the same 6 questions again. Their one-attempt bound is $0.087581, and it is already part of the canonical run's $0.949909 bound.
- **Registry.** Register the check as its own record, list the derived manifest's sha256 in `identity.files`, and state its extra cost.

### Operational success criteria

Accuracy is not a criterion. The check passes only if all of the following hold:

1. `run_config.json` records mode `live`, model `gpt-4o-2024-11-20`, endpoint `https://api.openai.com/v1` and params exactly `{"temperature": 0}`.
2. Each of the 6 sent questions has exactly one `dispatch_started` and one `response_saved`. The 2 skipped questions have no dispatch and export as `no_evidence`.
3. Every saved response has integer input and output token counts, and its `measured_cost` equals usage times the configured rates.
4. Every returned model equals `gpt-4o-2024-11-20` exactly, and no input bound is exceeded.
5. There are no `attempt_failed`, `outcome_unknown` or safety-violation events. Any that occur need a written cause, and none counts as a pass.
6. `run_state` is `complete` and 0 questions are `execution_failed`.
7. The ledger has no anomaly and nothing reserved. The summary's measured cost equals the sum of the response costs.
8. `raw.service_tier` is `default` or absent in every response.
9. The dashboard usage for the check's time window matches the run summary, checked by hand.
10. `score --evaluation-manifest <subset>` runs. Its numbers are not used for any decision.

A helper, `scripts/audits/verify_live_check.py`, checks criteria 1 to 8 from the run directory. It is committed and has no independent review. It passed on a fake subset run and failed as intended when given wrong expectations. It has never seen a live run. It reads the fields that newer runner code adds (`returned_service_tier`, `run_kind`, `pilot_manifest`, the storage policy) when they are present and reports `absent` otherwise, so re-run it after the code lands.

### Commands (NOT EXECUTED)

These commands are for the code at `c9b7ad5` and are superseded. Use the current version in "Update after the safeguards". They are kept as the record of what this review proposed.

Pre-flight by hand: resolve B1 and B2, freeze the code, write the approved config, and add a `planned` registry record. Run from the repository root.

```bash
RUN=YYYY-MM-DD-ohr-dev-v1-live-check-r1
CFG=.local/work/provider-config.<approved>.json
MAN=.local/work/livecheck/runtime_manifest.livecheck-v1.json
shasum -a 256 "$MAN"
```

The hash must be `a1e147c0dbe47274fa7bdbc381035f54490ac8b3b567ea54fdcf1807ad23a0c3`.

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py dry-run --runtime-manifest "$MAN" --provider-config "$CFG" --out .local/work/livecheck/dry-run
```

This offline dry run must report 8 questions, 6 to send and a one-attempt bound of 0.087581.

NOT EXECUTED. The next command spends money. `OPENAI_API_KEY` is set by the lead.

```bash
env -u OPENAI_BASE_URL -u OPENAI_ORG_ID -u OPENAI_PROJECT_ID -u OPENAI_ORGANIZATION -u OPENAI_LOG -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy -u SSL_CERT_FILE -u SSL_CERT_DIR FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY .local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py run --mode live --runtime-manifest "$MAN" --provider-config "$CFG" --run-dir "results/development/$RUN" --run-id "$RUN" --safety-ceiling 0.15
```

Then inspect the run. Both commands are read-only.

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py status --run-dir "results/development/$RUN"
```

```bash
.local/venv-prebaseline/bin/python scripts/audits/verify_live_check.py "results/development/$RUN"
```

Score only if every criterion holds. Scoring is final: after it, `run`, `reopen` and `reconcile` refuse the run. First derive the subset evaluation manifest. It holds reference answers, so keep it local.

```bash
.local/venv-prebaseline/bin/python -c "import json; ids=set(json.load(open('.local/work/livecheck/selected_question_ids.json'))); m=json.load(open('results/pilots/ohr_dev_v1/evaluation_manifest.json')); m['questions']={k: v for k, v in m['questions'].items() if k in ids}; json.dump(m, open('.local/work/livecheck/evaluation_manifest.livecheck-v1.json', 'w'), ensure_ascii=False, indent=1)"
```

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py score --run-dir "results/development/$RUN" --evaluation-manifest .local/work/livecheck/evaluation_manifest.livecheck-v1.json
```

If the run ends `needs_reconciliation`, `stopped`, `budget_limited` or `safety_stopped`, do not raise the ceiling to finish it. Read `attempts.jsonl`, compare with the dashboard, and record the cause. A `safety_stopped` directory is final.

### Full-pilot cost, recalculated offline

The table covers the 63 prepared requests at the candidate rates.

| Quantity | Value | Kind |
| --- | --- | --- |
| Expected | $0.24, range $0.19 to $0.31 | Estimate. Input is 91,785 heuristic tokens, with a range of 74,535 to 115,066 from assumed characters-per-token ratios. That range is a sensitivity check, not an envelope. Output is 15 tokens per request (range 5 to 40), with no retries. There is no tokenizer count, because `o200k_base` is not cached and was not downloaded |
| Expected, study brief assumptions | $0.27 | Estimate. 20 output tokens and 10% billed retries |
| One attempt per question | $0.949909 | Bound. Matches `dry_run_summary.json` |
| Three attempts per question | $2.849727 | Bound. Matches `dry_run_summary.json` |
| At Fast-tier rates (B2) | $0.41 expected, $1.614844 and $4.844532 bounds | Same inputs multiplied by 1.7 |

A $2.00 ceiling admits every first attempt plus about $1.75 of extra reservations. Spread evenly, that is about 1.8 extra reserved attempts per question.

- The ledger reserves each attempt's bound before dispatch. It counts rejected, unknown and usage-less attempts at their bounds.
- The coordinating agent replayed this against `SafetyLedger`, and an independent reviewer repeated the replay.
  - With two rejected attempts on every question, the run stops as `budget_limited` after 57 of 63 questions. The result is 58 if answers use 5 output tokens instead of 15.
  - Two unknown attempts per question reach the same limit, but only after many `reconcile` steps and restarts, because three counted attempts in a row trip the circuit breaker.
  - A run whose responses all lack usage still completes, at $0.949909 committed.
- The ceiling does not guarantee completion or limit the bill.
  - A reservation is a bound, not a charge.
  - A charge already made cannot be undone.
  - The ceiling holds only as far as the provider reports usage and the bounds are valid.
  - It does not see the service tier or any other use of the same key.

## 6. What this pilot can support, and what remains

- **What it can support.** The 70-question development run can show:
  - that the real-model path works end to end;
  - the parse, abstention and failure rates;
  - the tokens, dollars and latency per question;
  - a first EM and F1 with a wide interval (study brief section 15.10).
- **What it cannot support.** Claims of significance, behaviour on unseen documents, benchmark failure frequencies, anything about diagnosis or repair, chart questions, or long documents. 23 of its 30 documents are single pages.
- **Coverage.** The six empty-OCR questions and one heading-only question never reach the model. The agents doubt the gold answer of four of the sent questions (a provisional concern), and at least six sent questions have no answer in their evidence. Report the Han-script and empty-OCR groups as their own rows.
- **Human inspection still needed.**
  - The 20-case packet is unlabelled: 24 rows, 0 filled.
  - A person should label the text defects and evidence impact blind to this review (study brief section 10), then compare with it.
  - In particular, a person should confirm or reject the provisional gold-answer concerns for cases 1, 2, 4, 7, 14 and 17, and check the retrieval miss in case 20 and the degenerate OCR in case 17.
  - The diagnosis study needs a second independent labeller.
- **Unapproved.** Nothing in section 4 is decided. The prompt, model, prices, ceilings, retrieval settings, failure-rate limit, input-limit rule, storage, service tier and code freeze all remain open.
- **Licensing, unchanged since [ohr-scorer-provenance.md](ohr-scorer-provenance.md).**
  - Permissive terms cover the SQuAD and HotpotQA parts of the scorer. No permission was found for the four OHR-only elements.
  - The OHR-Bench dataset is stated as research-only and non-commercial in its README, while its Hugging Face card says CC BY 4.0.
  - The repository `haseebraza715/FailSafeRAG` is public and serves the vendored `OHR-Bench/` tree.
  - No source addresses sending the documents to a third-party model API.
  - Nobody was contacted, no legal clearance is claimed, and visibility is unchanged.

## 7. Artifacts and commit inspected

- **Commit.** `c9b7ad5` on `research/prebaseline-engineering`, equal to `origin`. The working tree was clean before this review, which adds the two reports and one line in `docs/reports/index.md`. PR #2 is a draft and unmerged. `git diff 884e388 c9b7ad5 -- src scripts tests config` is empty, so the code identity is that of `884e388`.
- **Frozen inputs, read only:**
  - `results/pilots/ohr_dev_v1/runtime_manifest.json`, sha256 `08be5719...`;
  - `evaluation_manifest.json`;
  - `inspection/` (`annotations.csv`, `inspection_cases.json`, `render_record.json`, and the 17 page PNGs, git-ignored);
  - `OHR-Bench/data/retrieval_base/MinerU/`;
  - `OHR-Bench/data/qas_v2.json`.
- **Dry runs at `c9b7ad5`, local only because they hold full document text:**
  - Default: `prepared_requests.jsonl` sha256 `b6cb6fcb...`, identical to `.local/work/prompt-preview-final/`.
  - Candidate config (`.local/work/provider-config.UNAPPROVED.gpt-4o-2024-11-20.json`): `prepared_requests.jsonl` sha256 `639d8400...`, template `faar-answer-draft-v1` sha256 `52044de2...`.
- **Provider documentation, read 2026-09-30 as raw Markdown:**
  - https://developers.openai.com/api/docs/models/gpt-4o
  - https://developers.openai.com/api/docs/pricing
  - https://developers.openai.com/api/docs/deprecations
  - https://developers.openai.com/api/docs/guides/your-data
  - https://developers.openai.com/api/docs/guides/migrate-to-responses
  - https://developers.openai.com/api/docs/guides/priority-processing
  - https://developers.openai.com/api/docs/guides/rate-limits
  - https://developers.openai.com/api/docs/guides/error-codes
  - https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create, read by a worker through a summarising fetch tool, not as raw Markdown
- **Case-level evidence.** [ohr-dev-v1-agent-review-2026-09-30.md](ohr-dev-v1-agent-review-2026-09-30.md).
- **Workers.**
  - Four `faar-worker` agents did the inspections: cases 1 to 10, cases 11 to 20, the prompt audit, and configuration with costs. The agent type is configured as `claude-sonnet-5-5` at effort `high`, which is configured and not observed at run time.
  - The coordinating agent is `claude-opus-5-5` at effort `high`, per the session metadata. It reconciled the findings and re-checked:
    - the case 2, 4, 14, 17 and 20 claims;
    - the storage, service-tier, price and deprecation text in the raw documentation;
    - `trust_env`;
    - the live-check selection.
- **Worker outputs.** The audit scripts are now in [scripts/audits/](../../scripts/audits/README.md), and the structured agent-review rows are in [ohr-dev-v1-agent-review-2026-09-30.cases.json](ohr-dev-v1-agent-review-2026-09-30.cases.json). Full prompts, OCR text and page images stay local by policy and are not preserved. Also not preserved, and kept only in the session's scratch directory without a backup: the workers' full markdown notes (they quote document text), the fake subset run, the retry-table script (`error_table.py`), the multi-environment stability runs, and the extra prompt-audit pass (OCR-noise ranking, reference-answer lengths, multi-page structure).
- **Live-check files.** `select_live_check.py`, `check_subset.py` and `verify_live_check.py` are in `scripts/audits/`. The derived manifest and `selected_question_ids.json` in `.local/work/livecheck/` are git-ignored. Run `select_live_check.py` to regenerate them from a dry run. The regenerated manifest is byte-identical to the scratch one.
- **Independent fact-check.** A separate `faar-worker` reviewer checked both reports. It found no critical issues, 4 high, 6 medium and 6 low.
  - All four high findings were fixed before commit. They covered B1 and B2 overstated as blocking, an output-limit cost that was 10 times too low, and the case 4 reason.
  - One medium finding was dismissed. It claimed the helper's safety-violation check reads a missing key. `run_summary.json` has `safety_violations` whenever there are violations (`src/faar/live_runner.py:2504`), so the check works.
  - The other medium and low findings were fixed.
- **Preservation.** No file under `results/pilots/`, `annotation/`, `experiments/`, `logs/`, `config/` or `src/` changed. Stashes `stash@{0}` and `stash@{1}` and the recovery tag are untouched.
