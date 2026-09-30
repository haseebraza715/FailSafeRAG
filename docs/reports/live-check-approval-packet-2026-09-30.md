# Approval packet: small live check and full development pilot (2026-09-30)

Status: a request for two separate human decisions. Nothing in it is approved. No paid request, connectivity probe or credential read has been made. The facts come from the code at `aaf9a4a`, the offline audits reproduced at that commit, and vendor pages read on 2026-09-30. The long form is the [live-baseline readiness review](live-baseline-readiness-2026-09-30.md). This packet repeats only what the lead needs to decide.

Two approvals are asked for, in order. Approval A covers the small live check alone. Approval B covers the full development pilot and is asked for only after the small check has run and its records have been reviewed.

## 1. Code and configuration identity

| Item | Value |
| --- | --- |
| Branch and commit | `research/prebaseline-engineering` at `aaf9a4a`. CI run 36731761957 passed. PR #2 is a draft |
| Code identity | `git diff --stat ca26e31 aaf9a4a -- src scripts tests config pyproject.toml` is empty. The code is that of `ca26e31`, which the readiness review describes |
| Full suite at `aaf9a4a` | 2494 passed, 2 skipped, exit code 0, full output in `.local/work/suite-full-aaf9a4a-r7.log`. One earlier full run on 2026-09-30 reported 1 failed and 2493 passed, but its command piped pytest through `tail -1`, so the failing test's name was lost. Three later runs and this one pass. That failure is recorded as unresolved, not fixed |
| Provider config | `.local/work/provider-config.UNAPPROVED.gpt-4o-2024-11-20.json`, sha256 `a1dd872833d63918233a0b3f286cca2e51fa291666bc34da93888a987b754d75`. Its content is in section 2. On approval, the lead copies it to a name without `UNAPPROVED` and the run records the new path. The bytes must stay the same, or the digests below change |
| Prompt template | `faar-answer-draft-v1`, sha256 `52044de279b32482e1b8b7f75f0b3455f5cb8e44e9178f8cec09c1615a89914d` |
| Frozen runtime manifest (70 questions) | `results/pilots/ohr_dev_v1/runtime_manifest.json`, sha256 `08be57192da5b74423b9028d0c5ab8ca0f1807250a6754c9cdb3320ff970f3e1` |
| Live-check manifest (8 questions) | `results/audits/2026-09-30-readiness/live-check-selection/runtime_manifest.livecheck-v1.json`, sha256 `a1e147c0dbe47274fa7bdbc381035f54490ac8b3b567ea54fdcf1807ad23a0c3`, an exact subset of the frozen manifest |
| Full-pilot dry-run identity | `dry_run_identity_sha256` `6a0075d5359dc4209409358cb272d442b7b61085e0759ae4af1a133be7fd9267` (63 sent, 7 skipped, largest bound 7,920, one-attempt bound $0.949909) |
| Live-check dry-run identity | `dry_run_identity_sha256` `e8beee912bd144ab670dc43d1b1edb3bde710ccf0ada64e7fd61945e178980ba` (6 sent, 2 skipped, largest bound 7,920, one-attempt bound $0.087581, question-id digest `75ed63bf...`) |
| Reproduction at `aaf9a4a` | Both dry runs and all seven audit scripts were re-run on 2026-09-30. Every figure above and every compact output in `results/audits/2026-09-30-readiness/` reproduced byte for byte, except the `prompts/audit_prompts.json` trace, which differs only in output paths and in a `.env` entry that a credential-free shim removed |
| Retrieval | The settings recorded in `run_config.json` of `2026-09-30-ohr-dev-v1-fake-provider-r5` (`multilingual-v1` tokeniser, `cjk-weighted-words-v1` chunking, five hits per question). They are engineering settings. The readiness review notes one known retrieval miss (case 20) |

The run identity covers every `src/faar/*.py` file and the CLI script. An edit to any of them after the first paid request makes a partly paid run refuse to resume. Approval A therefore freezes the code at `aaf9a4a` for both runs. If a change is needed between the two runs, the full pilot runs on the changed code as a new run, and this packet is reissued.

## 2. Model and parameters

The candidate config, byte for byte except for the `source` note:

```json
{
  "provider": "openai",
  "model": "gpt-4o-2024-11-20",
  "endpoint": "https://api.openai.com/v1",
  "params": {"temperature": 0},
  "max_output_tokens": 128,
  "max_input_tokens": 12000,
  "timeout_seconds": 60,
  "tokenizer_bound": "utf8-bytes",
  "token_limit_param": "max_tokens",
  "prices": {
    "currency": "USD",
    "input_per_million": 2.5,
    "cached_input_per_million": 1.25,
    "output_per_million": 10.0,
    "source": "OpenAI pricing page Standard gpt-4o row and gpt-4o model page, read 2026-09-30",
    "source_date": "2026-09-30",
    "service_tier": "default"
  },
  "service_tier": "default",
  "store": false
}
```

Vendor facts re-read on 2026-09-30 from the [gpt-4o model page](https://developers.openai.com/api/docs/models/gpt-4o), the [pricing page](https://developers.openai.com/api/docs/pricing) and the [deprecations page](https://developers.openai.com/api/docs/deprecations):

- `gpt-4o-2024-11-20` is listed as a snapshot. The undated alias `gpt-4o` points to `gpt-4o-2024-08-06`, so the config must keep the dated name. The returned-model check is an exact string match.
- Standard rates: $2.50 input, $1.25 cached input, $10.00 output per million tokens. The pricing page has no row for the dated snapshot. The Fast tier is 1.7 times Standard. There is no Flex row.
- The deprecations page lists `gpt-4o-2024-05-13` (shutdown 2026-10-23) and `chatgpt-4o-latest`. It does not list `gpt-4o-2024-11-20`.
- Rate limits: 30,000 tokens per minute at usage Tier 1 and 450,000 at Tier 2. The account's tier is unknown. A 429 is an unknown outcome that waits for `reconcile`, and three in a row stop the invocation.
- Public documentation does not show that this account can call the snapshot. Only the first request shows that.

Other request settings, all from the code: three attempts per question, backoff from 2 seconds, only connect errors, HTTP 408 and 503 `server_is_overloaded` are retried. Every other failure waits for a person.

## 3. What leaves the machine

Each request is one HTTPS POST to `https://api.openai.com/v1/chat/completions` with the API key in the `Authorization` header. The body holds the system prompt, the question text and the retrieved noisy OCR chunks (at most five), each headed by a 1-based page number and a document-free chunk label. It holds `store: false`, `service_tier: "default"`, `model`, `temperature`, `max_tokens` and nothing else.

| Run | Requests | Bytes of message text | Documents |
| --- | --- | --- | --- |
| Small check | 6 | at most 7,896 per request | 6 |
| Full pilot | 63 | 346,190 in total, at most 7,896 per request | 23 |

No image, gold answer, reference text, evidence label or document name leaves the machine. The audits `audit_prompts.py` and `audit_prompt_leakage_diagnostic.py` check this on the prepared requests, and both pass at `aaf9a4a`. The live client ignores proxy variables and system proxies and follows no redirect, so a request reaches the configured endpoint or fails.

## 4. Storage and retention

- Every request sends `store: false`. The CLI refuses any other value. This is not zero retention.
- OpenAI's [data-controls page](https://developers.openai.com/api/docs/guides/your-data), read 2026-09-30, gives `/v1/chat/completions` an application-state retention of "None, see below for exceptions", keeps abuse-monitoring logs for up to 30 days, and does not use API data for training by default. Zero Data Retention and Modified Abuse Monitoring need OpenAI's prior approval, and this account has not applied.
- Sending OHR-Bench document text to a third-party API is a data-handling decision. The dataset README limits the documents to research use, and the Hugging Face card says CC BY 4.0. No source addresses transmission to a model API. Approval A accepts this transmission for the six live-check documents. Approval B accepts it for all 23.

## 5. Costs

Figures from `results/audits/2026-09-30-readiness/` (`cost_and_ceiling.py`). Estimates use a heuristic token count and stated assumptions. Bounds use the UTF-8 byte bound on input and the full 128-token output limit, priced at Standard rates.

| Quantity | Small check | Full pilot | Kind |
| --- | --- | --- | --- |
| Expected | $0.021 (range $0.018 to $0.029) | $0.24 (range $0.19 to $0.31) | Estimate |
| One attempt per sent question | $0.087581 | $0.949909 | Bound |
| Three attempts per sent question | $0.262743 | $2.849727 | Bound |
| Proposed safety ceiling | $0.15 | $2.00 | Lead decision |
| Same bounds at Fast-tier rates | $0.148888 and $0.446663 | $1.614844 and $4.844532 | Bound, 1.7 times |

What the ceiling does and does not do:

- The ledger reserves each attempt's bound before dispatch and refuses an attempt that would exceed the ceiling. Rejected and unknown attempts stay reserved at their bound.
- At $0.15 the check completes if each question needs at most one extra reserved attempt. At $2.00 the pilot completes with about 1.8 extra reserved attempts per question spread evenly.
- A response from a tier other than Standard stops the run before the next dispatch. Its charge is already made and may be 1.7 times the reserved bound. A project whose Service Tier is set to Fast would bill the first response that way. An optional dashboard check before the run removes that case.
- The ceiling holds only as far as the provider reports usage and the bounds are valid. It is not a bill limit and does not see other uses of the same key. A project spend limit in the OpenAI dashboard works independently and stops the run with a quota error when reached.
- Total exposure under both approvals is the sum of the ceilings, $2.15, under those conditions.

## 6. Small check: selection and denominator

The rule `faar-live-check-v1` ranks questions by `sha256("faar-live-check-v1|" + question_id)` and fills five strata in order from the prepared requests. It reads no answer or score. `scripts/audits/select_live_check.py` regenerates it, and `check_subset.py` proves the exact-subset property. The result is 8 questions from 8 documents:

| Stratum | Question ids |
| --- | --- |
| Largest input bound | `65ea29d1-adae-4e45-a036-e74be34d7804` |
| Han-script question | `84403b37-b13b-4d4e-a368-f7d2d9732b86`, `ecd44593-9f98-4aa0-9890-46dbf35c54ea` |
| Document of three or more pages | `b7ad5f22-7e21-476c-9933-ea8fb579e050` |
| Single page, no Han | `50e1fdda-86c3-4047-9758-9cc633f33b00`, `76a3d0f7-998d-476a-aac4-55c92f7a8e02` |
| Skipped, no usable text | `ecbf8bf8-e836-4b50-8ca8-ebc629deb1d7`, `ecccdf67-bc03-447c-a990-d5989185f21a` |

Six are sent. Two get `no_evidence` with no model call. All eight stay in the export. The check runs as `engineering_check`, is not canonical, can never be `valid_baseline`, and its answers are never reused. The full pilot pays for the same six questions again.

## 7. Open licensing and data-use questions

These are recorded, not resolved, and no legal clearance is claimed.

- Whether the OHR-Bench terms allow sending document text to a third-party model API (section 4).
- The four OHR-only parts of the vendored scorer have no licence ([scorer provenance](ohr-scorer-provenance.md)).
- The public repository already serves the vendored `OHR-Bench/` tree.

Approval A and B do not settle these. They record that the lead accepts the transmission for the development runs knowing the questions are open.

## 8. Operational success conditions

Accuracy is not a criterion. `scripts/audits/verify_live_check.py <run dir>` checks items 1 to 8 from the records. Items 9 and 10 are by hand.

1. `run_config.json` records mode `live`, kind `engineering_check`, model `gpt-4o-2024-11-20`, endpoint `https://api.openai.com/v1`, params exactly `{"temperature": 0}`, `store: false` and `service_tier: default`.
2. Each of the 6 sent questions has exactly one `dispatch_started` and one `response_saved`. The 2 skipped questions have no dispatch and export as `no_evidence`.
3. Every saved response has integer input and output token counts, and `measured_cost` equals usage times the configured rates.
4. Every returned model equals `gpt-4o-2024-11-20` exactly. No input bound is exceeded.
5. No `attempt_failed`, `outcome_unknown` or safety-violation event. Any that occur get a written cause, and none counts as a pass.
6. `run_state` is `complete` with 0 `execution_failed`.
7. The ledger has no anomaly and nothing reserved. The summary's measured cost equals the sum of the response costs.
8. Every returned `service_tier` is `default`.
9. The dashboard usage for the check's time window matches the run summary.
10. `score --evaluation-manifest <subset>` runs. Its numbers inform no decision.
11. A second `run` on the finished directory dispatches nothing and changes no record (resume check, free).

If the run ends `needs_reconciliation`, `stopped`, `budget_limited` or `safety_stopped`, the operator preserves everything, reads `attempts.jsonl`, compares with the dashboard, and records the cause. The ceiling is not raised to finish a check. A `safety_stopped` directory is final. No uncertain attempt is resent without the lead's decision, because a resend may be a second charge.

## 9. Commands (NOT EXECUTED)

Run from the repository root at `aaf9a4a` with a clean working tree. The API key comes from the existing mechanism: the live CLI imports `faar.pilot_runner`, which imports `faar.settings`, which calls `load_dotenv(override=False)` and finds the repository-root `.env` from any working directory. That file is git-ignored, already present, and holds `OPENAI_API_KEY`. `build_live_provider` reads the variable only after every offline check has passed, and the test suite asserts that no earlier step reads it. No command prints the key. `OPENAI_MODEL` in `.env` is loaded into the process but the live path takes the model from the config only. `.env` holds no `FAAR_ALLOW_LIVE_REQUESTS` line and none of the four refused `OPENAI_*` names, checked by name on 2026-09-30, so the spending gate is still the explicit variable on the command line.

The safeguards were verified on 2026-09-30 by running the nine live-path test files (1,050 tests, every file exit 0) and reading the code they exercise. The evidence table is in `.local/work/live-baseline-preparation.md`, round 7. Two coverage gaps were found and judged low risk, and no test was added: the tier-stop variant of the "reconcile and reopen do not clear the stop" test does not call `reconcile`, and the `allow_new_attempt` resolution after a safety stop has no direct test. The stop check runs before the provider is built on every `run`, whatever the violation, so both paths share the tested code.

Pre-flight, free:

```bash
git status --short && git rev-parse --short HEAD
```

```bash
CFG=.local/work/provider-config.gpt-4o-2024-11-20.approved.json
MAN=results/audits/2026-09-30-readiness/live-check-selection/runtime_manifest.livecheck-v1.json
shasum -a 256 "$CFG" "$MAN"
```

The manifest hash must be `a1e147c0dbe47274fa7bdbc381035f54490ac8b3b567ea54fdcf1807ad23a0c3`. The config is the candidate file with the `source` note edited; record its hash in the run's registry entry.

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py dry-run --runtime-manifest "$MAN" --provider-config "$CFG" --out .local/work/livecheck/dry-run-approved
```

It must report 8 questions, 6 to send and a one-attempt bound of 0.087581.

Approval A, spends money:

```bash
RUN=2026-10-01-ohr-dev-v1-live-check-r1
FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY .local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py run --mode live --run-kind engineering_check --runtime-manifest "$MAN" --provider-config "$CFG" --run-dir "results/development/$RUN" --run-id "$RUN" --safety-ceiling 0.15
```

Inspection, free:

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py status --run-dir "results/development/$RUN"
```

```bash
.local/venv-prebaseline/bin/python scripts/audits/verify_live_check.py "results/development/$RUN" --expect-run-kind engineering_check --expect-sends 6 --expect-skips 2 --expect-manifest-sha256 a1e147c0dbe47274fa7bdbc381035f54490ac8b3b567ea54fdcf1807ad23a0c3
```

Scoring, only if the verifier reports no FAIL. Scoring is final for the directory:

```bash
.local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py score --run-dir "results/development/$RUN" --evaluation-manifest .local/work/livecheck/evaluation_manifest.livecheck-v1.json
```

Approval B, spends money, only after the review of the small check:

```bash
RUN=2026-10-01-ohr-dev-v1-baseline-r1
FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY .local/venv-prebaseline/bin/python scripts/experiments/run_pilot_live.py run --mode live --run-kind development_pilot --provider-config "$CFG" --run-dir "results/development/$RUN" --run-id "$RUN" --safety-ceiling 2.00
```

Then `status`, `score --run-dir "results/development/$RUN"` (the frozen evaluation manifest is the default), and a `development_pilot` registry record. The dates in the run ids are placeholders for the day the run happens.

## 10. What each approval covers

**Approval A, the small live check.** Six paid requests to `gpt-4o-2024-11-20` with the config in section 2, on the 8-question manifest in section 6, at a $0.15 ceiling, registered as `engineering_check`, with the transmission and retention in sections 3 and 4 accepted for those six documents. It freezes the code at `aaf9a4a`. It does not approve the full pilot, the prompt as the final baseline prompt, the retrieval settings as approved settings, or any repair experiment.

**Approval B, the full development pilot.** 63 paid requests with the same code and config on the frozen 70-question manifest at a $2.00 ceiling, registered as `development_pilot`, with the transmission accepted for all 23 documents. It is asked for separately, after the small check's records have been reviewed against section 8. It does not make the result a scientific evaluation, and it does not approve the study brief's open lead decisions beyond items 1, 2, 3, 4, 5, 11 and 13 of section 15.13 for this development run.

Neither approval settles the human inspection of the 20 cases, the second labeller, the maximum failure rate as a rule, the licensing questions, or the repair experiments. Those stay on the list in section 6 of the readiness review.

To approve, reply with the approval letter, the config digest and the ceiling, for example: "Approve A: config a1dd8728, ceiling $0.15". A reply that names different values is a change to this packet, not an approval of it.
