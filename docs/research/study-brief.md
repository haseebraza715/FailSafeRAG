# First study: diagnosis-based recovery for OCR-backed document QA

This brief is the current design of FAAR's first study. It replaces
[aaai-plan.md](aaai-plan.md) as the plan to follow. It describes intended
behaviour. It does not report results, and it does not authorise model calls,
OCR, paid services or experiment runs.

Each rule below carries one of three labels:

- **Agreed:** the research lead has decided it. Change it only with the lead's approval.
- **Proposed default:** a proposal, not an approved rule. Engineering and development work may use it provisionally and must say so in its run record. No scientific evaluation relies on it until the lead accepts, changes or rejects it.
- **Lead decision:** open. Do not implement a choice until the lead makes it.

Section 12 lists every rule with its label. Section 11 compares the intended
behaviour with the code at commit `8832ce4` and notes what the pre-baseline
engineering branch changed since then.

## 1. Question and claim

FAAR asks: does identifying a failure in OCR-backed document question answering,
and choosing a repair that matches it, answer more questions correctly than
simpler recovery policies at comparable cost?

The question has two links, and the study tests them separately:

1. Does the controller's diagnosis agree with an independent observation of what is wrong?
2. Does the repair the controller chooses improve the answer more than another repair would?

Agreement between a diagnosis and the policy that consumes it proves neither
link. A null or negative result is a valid outcome and stays in the record.

The first experiment can establish, on development data, whether the choice of
repair changes answers at all and whether the current signals separate the cases
where it does. It cannot establish generalisation to unseen documents or
publications, or statistical significance. Section 9 explains why.

## 2. Agreed scope

- OHR-Bench is the first dataset. ArXivQA, MP-DocVQA and collection-wide retrieval are later work.
- The existing MinerU text in `OHR-Bench/data/retrieval_base/MinerU/` is the fixed noisy-text input. Its engine version and settings are unknown upstream, and every report says so.
- The locked QA file, the fixed split and the PDF source lock stay unchanged. The [data audit](../reports/data_audit.md) records their hashes and coverage.
- `ohr_dev_v1` stays frozen. Its 30 documents, 70 questions and 54 pages form a development sample, not the final evaluation. The [pilot readiness report](../reports/pilot_readiness.md) records its selection.
- The first pilot searches within each complete benchmark document: all pages of the document the question names, never only the gold evidence pages.
- Gold answers, evidence-page labels, evidence contexts and clean reference text stay outside runtime retrieval, gating, diagnosis, recovery and answering.
- Missing or empty OCR is an explicit input condition, never an automatic exclusion.
- Failed runs and negative findings stay in the record. See [experiments/README.md](../../experiments/README.md).
- Nothing is tuned on test outcomes.

A frozen record, such as `ohr_dev_v1`, the locks or a completed run, can get a
separately versioned successor with a new identifier when the lead approves one.
The original is never edited.

## 3. Three observations, kept apart

The study records three different things about each case. They answer different
questions, and one does not imply another.

| Observation | Question it answers | Who can make it | When |
| --- | --- | --- | --- |
| Text defect | What is wrong with the noisy text or layout of a page, compared with the page image? | A person comparing image and text; the controller's text signals are a prediction of it | Before any retrieval |
| Evidence impact | Does that defect remove or distort the evidence this question needs? | A person who can see the question and the reference answer | Before any retrieval |
| Repair effect | Does a given repair change the answer from wrong to right, or right to wrong? | The scoring step, after the run | After answering |

Each observation allows more than one value and an explicit "uncertain" value.
A page can show word corruption and lost structure at once. A case can have
intact evidence on a badly damaged page.

An OCR defect is not a retrieval failure. Whether retrieval missed the evidence
is a fourth fact, and it exists only after retrieval has run. Call it a
page-level retrieval miss only when the run record shows that the retrieved
chunks did not include the evidence page.

Page coverage is weaker than evidence coverage. A retrieved chunk can come from
the evidence page and still leave out the passage that holds the answer,
because one page can produce several chunks. On a single-page document, page
coverage is 1 by construction. A hit on the evidence page therefore does not
show that the answer text reached the answer step, and it says nothing about
whether that text survived OCR. Report page coverage, passage coverage (a
retrieved chunk contains the answer evidence) and evidence survival (the
evidence-impact observation) as separate quantities. Passage coverage needs a
matching rule against `evidence_context` that the lead has not approved, so no
current report measures it.

**Controller categories are not observation categories.** The controller emits
`semantic`, `word_level` or `structural` and maps each to one action:
`retry_retrieval`, `correct_text` or `invoke_vlm` (`src/faar/graph.py:22-26`).
The human inspection form records word corruption, lost or misleading
structure, missing content, apparently adequate OCR, and uncertain or mixed.
The two schemes do not correspond one to one:

- Word corruption suggests `correct_text`, but a visual reading can also repair it.
- Lost structure suggests `invoke_vlm`.
- Missing content has no text-only repair. A visual reading can help only if the page is shown.
- `semantic` has no human-observable counterpart. The current code assigns it whenever neither text signal fires (`src/faar/quality.py:96-103`).

**Proposed default.** Compare the controller with people at the level of
each signal: does the layout signal fire when a person saw lost structure, and
does the corruption signal fire when a person saw word corruption? Report the
agreement per signal, with mixed and uncertain cases as their own rows. Do not
collapse human labels into the controller's three categories to compute a single
accuracy.

## 4. What each component may receive

| Component | May receive | Must not receive |
| --- | --- | --- |
| Retrieval | Question text; the document identity the question names; every page of that document from the PDF-derived inventory; MinerU text per page, including a record that a page's text is empty or missing | Gold answer, evidence pages, evidence context, ground-truth text, annotations |
| Gate and diagnosis | Question; retrieved chunks with scores; per-page OCR status | As above |
| Text repair (`correct_text`) | Retrieved chunk text | As above; ground-truth text as a correction target |
| Retrieval retry (`retry_retrieval`) | Question; retrieved chunks; the same document's pages | As above |
| Visual repair (`invoke_vlm`) | Question; page images selected by a declared policy (section 7); the same retrieved text the text answer received | As above; any page chosen by using evidence labels |
| Answer model | Question and the evidence its condition provides | As above |
| Scoring | Final and initial answers joined to the evaluation manifest after the run | Nothing flows back into runtime |

The frozen pilot already separates these inputs. `runtime_manifest.json` holds
the question, the document identity, the complete page inventory and references
to the noisy text. `evaluation_manifest.json` holds everything else. No current
runtime code reads either file (section 11).

## 5. Policies and fair comparison

### Compared policies

1. **No recovery.** Answer from the initial retrieval.
2. **Gate-triggered visual recovery.** Every case the gate flags gets the visual repair.
3. **Random recovery.** Every flagged case gets a repair drawn at random from the three, with several seeds fixed before the run.
4. **Diagnosis-based recovery (FAAR).** Every flagged case gets the repair its diagnosis selects.

### What all policies share

For a fair comparison, the policies differ only in which repair a flagged case
receives. These are shared:

- The same eligible questions and the same scoring rules.
- The same starting retrieval results, computed once and reused.
- The same initial answer, computed once. It is the answer that no recovery returns and the reference for repair and damage.
- The same gate decisions. The gate runs once, and every recovery policy acts on the same flagged set.
- The same answer model identity and settings.
- The same prompt content, except the input-format differences a condition requires, such as attaching images.
- The same text budget (chunks and characters) and the same image budget (pages, resolution and detail setting per call).

### Two views of the results

- **End-to-end policy performance:** accuracy on all eligible questions, including cases the gate passes.
- **Repair performance on the flagged set:** repairs and damage on the shared set of gated cases only. This view isolates the effect of the repair choice.

Report both views.

### The answer model

The current text path has no answer model. `answer_from_hits`
(`src/faar/answering.py:50-77`) picks a span by rule-based term overlap, and its
tokeniser matches only `[a-z0-9%$]` (`answering.py:7`). The visual path sends
images to a large vision-language model. Comparing the two confounds the repair
with the answer model. A visual repair could look better only because a strong
model wrote its answer. The rule-based extractor also cannot score term overlap
for Han-script questions.

**Proposed default.** Use one answer model for every condition. Keep the
recorded snapshot `gpt-4o-2024-11-20`, which `src/faar/settings.py:226` and the
checks at `settings.py:306-308` already pin. The text condition gives it the
retrieved text, and the visual condition gives it the same text plus page
images. OpenAI's deprecations page, read on 2026-09-28, does not list this
snapshot for shutdown. That is documentation, not a test of account access.
Keep the rule-based extractor only as a zero-cost reference row, not as the
baseline.

**Lead decision.** Approve the answer model, its prompt, and any spending.
Nothing here authorises API calls.

### Cost reporting

Report cost per policy, per question and in total:

- API requests, including retries.
- Input and output tokens, where the API reports them.
- Image count per request, image resolution, and the `detail` setting. The current OpenAI call sets no `detail` value, so the API default applies.
- API cost in US dollars, from the rates recorded with the run.
- Local processing time and wall-clock latency per question.
- One-time preparation cost, such as rendering pages and building indexes, reported separately from per-question cost.

Two policies are cost-matched only when the match is enforced and stated.
**Proposed default:** match on visual requests with an identical image budget
per request. On the flagged set, restrict random and gate-triggered visual
recovery to the number of visual requests FAAR made, choosing which cases get
them by the seeded draw. Report dollar cost alongside the matched count.

A visual-request limit matches one cost component. It does not match total
cost. Two policies with the same number of visual requests can still differ in
text-only requests, retries, input and output tokens, dollars and latency, for
example when `correct_text` or `retry_retrieval` adds work that the visual
policy does not. Call such a comparison "matched on visual requests", not
"cost-matched", and report every component in the list above for each policy.
Claim matched total cost only if a total budget is enforced and the recorded
totals agree within a tolerance stated before the run.

### When a budget runs out

**Proposed default.** The lead has not set any budget value. When a run has a
request or dollar budget and the budget runs out:

1. The run sends no further paid request.
2. Each remaining case that would have needed a request keeps its initial answer. Its record says `budget_exhausted`. That is a recorded recovery outcome, not a repair and not an `execution_failed` row.
3. Cases consume the budget in an order fixed before the run, such as manifest order or the seeded draw, never in an order that depends on answers or scores.
4. The run record reports how many cases each policy served and how many hit `budget_exhausted`. A comparison in which any policy ran out of budget is reported as budget-limited, with those counts next to the rates.

## 6. Development diagnostic: try every repair on the same cases

**Proposal for later execution. It is not authorised now.**

On the development pilot only, run each available repair on every flagged case:
no recovery, `correct_text`, `retry_retrieval` and `invoke_vlm`. Keep the
retrieval, initial answer and answer model shared, as in section 5. This shows
whether the choice of repair matters at all. If every repair gives the same
answers, diagnosis cannot help, whatever its accuracy.

- Evaluation answers are joined only after all repairs have run, to score them.
- The best repair per case in hindsight is a diagnostic upper bound. It is not a deployable policy, because choosing it needs the answer.
- The results may inform method development on development data, including changes to the diagnosis, the repairs or their settings. Record each change and the result that prompted it in the development log (section 9).
- These results must never be used to choose or tune anything on validation or test outcomes. Validation and test data enter only after the method is frozen (section 9).

## 7. Empty evidence and failed execution

### Current behaviour

- The OHR asset-manifest builder rejects any inventory page with empty OCR text (`src/faar/benchmarks.py:456-457`), so a document with one empty page cannot be registered.
- When a page does enter the corpus with no words, `build_page_chunks` returns no chunk for it (`src/faar/chunking.py:42-43`). The page then cannot be retrieved, and the visual repair cannot see it, because that repair reads only the images of retrieved chunks (`src/faar/graph.py:246-250`).
- On the per-example path, a document with no words raises `ValueError` (`graph.py:91-95`), and the runner records the question as a failed row.
- With no retrieval hits, the gate fails (`quality.py:49-57`), the diagnosis defaults to `semantic` (`quality.py:97-98`), and `semantic_backtrack` returns the query unchanged (`src/faar/recovery.py:546-547`). The repeated retrieval returns nothing again, and the answer is an empty string (`answering.py:73`).

In the pilot, 6 of 54 pages have empty MinerU text, and they are the evidence
pages of 6 questions.

### Intended baseline behaviour (proposed default)

- Every question stays in the evaluation. Empty OCR is recorded per page and per question.
- The complete PDF-derived page inventory is kept. A page with empty text produces no text chunk but remains addressable as an image.
- Nothing fabricates evidence or substitutes clean text.
- When no text evidence exists, the answer step returns an explicit no-evidence outcome (`no_evidence`) and an abstention, not an empty string.
- A `no_evidence` outcome is a scored answer. A software, API or infrastructure failure is an `execution_failed` outcome. The two are counted separately and never merged.

### Recovery when no text hits exist

Recovery may use the question, the known document identity, the full page
inventory and whatever noisy text exists. It may not use evidence-page labels.
For a short document, the visual repair can show every page. For a longer one,
page selection must follow a declared policy within a declared image budget.

**Lead decision.** No justified page-selection policy exists yet. With a
budget of five images, 28 of the 30 pilot documents fit whole, and the 8-page and
9-page documents do not.

### Limits of the current repairs

These are implementation requirements for later work, or limitations to report:

- `correct_text` edits only the retrieved chunks (`graph.py:147-151`). It cannot restore evidence on a page that was never retrieved.
- `invoke_vlm` sees only the images of retrieved chunks (`graph.py:246-250`). It cannot inspect a missed page.
- `retry_retrieval` appends the first 24 words of the retrieved text to the query (`recovery.py:544-549`). If the first retrieval was wrong, the expansion can repeat the same error.

### Failed execution

The runner currently refuses to finish when any question fails. It raises after
checkpointing the completed rows (`src/faar/experiment_runner.py:220-225`), and
a failed visual call raises before scoring (`experiment_runner.py:103-108`).
**Proposed default:**

1. Retry transient failures within the same run as new attempts (`experiments/README.md`).
2. Record any question that still fails as `execution_failed`, with its reason.
3. Finish the run, with status `completed` and the failure count in its limitations, or `failed` if the lead sets a maximum failure rate and the run exceeds it.
4. Count `execution_failed` questions as incorrect in end-to-end accuracy, and report the execution-failure rate next to it. Never drop them.

## 8. Scoring contract

### Primary metric

Use the official OHR-Bench QA metrics: `exact_match_score` and `f1_score` in the
vendored `OHR-Bench/src/metric/common.py`. Its normalisation lowercases the
text and removes ASCII punctuation and English articles. It keeps Han
characters, and F1 tokenises text that contains CJK characters with `jieba`.
The vendored `common.py` and its caller `src/tasks/quest_answer.py` are
byte-identical to upstream `opendatalab/OHR-Bench` at commit `1f421eb`, the
`commit_at_lookup` in `config/ohr_pdf_source_lock.json` (checked 2026-09-28).
`QuestAnswer.scoring` calls exactly these two functions on the `answers`
string. `src/faar/ohr_scoring.py` reimplements them without upstream's heavy
imports, with `jieba==0.42.1` and `regex==2024.7.24` as upstream pins them.
`scripts/experiments/ohr_scoring_parity.py` compares it with the vendored
functions. On 2026-09-28 it found 0 mismatches in 99,041 pairs: 103
hand-written edge cases, 11 variants of each of the 8,498 `qas_v2.json`
references, and 8 variants plus all cross pairs of the 70 pilot references.
The upstream repository has no licence file for its code. The module credits
the authors and pins the source, and redistribution terms are open (section 12).

Upstream's headline `overall` is not an all-question mean. `evaluator.py`
drops every result whose generated text is blank (`remove_invalid`, with
`valid` set in `quest_answer.py`) before it averages. `score_predictions`
reports that view as `upstream_valid_only`, with its denominator, and only next
to the all-question view defined below.

FAAR's own `src/faar/metrics.py` must not serve as the primary metric.
`normalize_text` deletes every character outside `[a-z0-9]`
(`metrics.py:8-11`). 521 of the 8,498 reference answers, and 7 of the 70 pilot
answers, normalise to an empty string under it. On those references a
prediction scores EM 1 and F1 1 exactly when the prediction also normalises to
an empty string: an empty answer, an abstention, or an answer written only in
Han script and punctuation. A prediction that keeps any ASCII letter or digit
scores 0 on them. The false perfect score therefore requires both normalised
strings to be empty. It is not given to every prediction. 785 reference answers
contain Han characters.

### Rules for special cases

| Case | Rule |
| --- | --- |
| Reference answers | Each QA row has one reference string (`answers`). If a later dataset has several, score against the best match. |
| Numbers, units, percentages | The official metric deletes every ASCII punctuation character, including `,`, `.`, `%` and `$`, before it compares strings. `1,000` and `1000` are therefore equal, and so are `3.5` and `35`, and `50%` and `50`. Keep this behaviour in the primary score. A separate FAAR numeric match (parsed value, unit and percent sign, stated tolerance) may be reported as a secondary metric. Never replace the official score with it. |
| Han-script answers | Official normalisation and `jieba` F1. Report Han-script questions as their own row. |
| Multi-part (`List`) answers | Official EM and F1 on the full string. A separate set-level F1 may be reported as secondary. |
| Empty answer or abstention | Scored by the official metric. Counted separately as `abstained` or `no_evidence`. Official EM and F1 are 0, except that EM is 1 (and F1 stays 0) when the reference also normalises to an empty string. 5 of the 8,498 references do so under the official normalisation, and none of the 70 pilot references. |
| Truncated or unparsable output | Scored as returned after extracting the answer field. Flagged in the row with an output status. |
| API or infrastructure failure | `execution_failed`, as in section 7. Not an answer. |

Every secondary metric is named as FAAR-defined in tables and in the paper.

### Denominators

Let N be the eligible questions and G the flagged (gated) questions, which all
recovery policies share. "Correct" means official EM equal to 1. The same
counts may be reported with F1 as a secondary view.

| Quantity | Numerator | Denominator |
| --- | --- | --- |
| Recovery rate | Questions the gate flags | N |
| Successful repair rate | Flagged questions wrong at first and correct after recovery | Flagged questions wrong at first |
| Damage rate | Flagged questions correct at first and wrong after recovery | Flagged questions correct at first |
| Visual-call rate | Questions with at least one visual request | N (also report images per question) |
| Execution-failure rate | Questions with `execution_failed` | N, per policy and per component |

Always report counts with rates. Report the initial and final answers on the
same questions. The current `_harm_rate` (`src/faar/final_analysis.py:124-144`)
counts any F1 decrease over all rows. That is not the damage rate defined here.

### Uncertainty

Questions that share a document are not independent. Resample by document for
confidence intervals, and note the source families that link documents (the
[pilot report](../reports/pilot_readiness.md) lists them). The pilot is too small
to show statistical significance or generalisation, and no report should claim
either from it.

## 9. What the pilot can and cannot test

`ohr_dev_v1` is preserved without reselection. Its limits, from the
[pilot readiness report](../reports/pilot_readiness.md):

- 23 of its 30 documents are single pages.
- It has no chart questions.
- A single page can still produce several chunks (180 words each by default, `src/faar/settings.py:102`), so retrieval can still pick the wrong passage on the right page.
- Wrong-page retrieval and retrieval retry get little coverage.
- Searching the complete document makes every evidence page a retrieval candidate. That is page coverage of the search space, not passage coverage of the results. On the 23 single-page documents every retrieved chunk is on the evidence page, whatever passage it holds (section 3).
- Three source families also appear in validation or test, so the pilot cannot support claims about unseen publications.

**Proposed default.** For retrieval-retry work, define a separately
versioned development sample of longer, train-exclusive documents, with its own
`pilot_id` and output directory. Do not create it until the lead approves it.

### Development logging and untouched data

Development iteration is allowed: inspect outputs, change code, adjust settings.
It must be visible.

- Register every development run in `experiments/registry.jsonl` as `development_pilot`, with its commit, settings and outputs.
- A change to data, measurement code or settings makes a new run, linked to the previous one. Do not overwrite an earlier result.
- Keep a short development log in the run records' `limitations` and `related_runs`, covering what was inspected, what changed and why.
- Validation and test questions stay untouched: no runs, no output inspection and no threshold selection, until the lead freezes the protocol.

### Freeze the method before evaluation

**Agreed** (no tuning on test outcomes, section 2), with the procedure as a
**proposed default**:

1. Before any validation or test run, freeze the method: the code commit, settings, thresholds, prompts, answer model, budgets and page-selection policy.
2. Register the frozen method in `experiments/registry.jsonl` and cite that record from every evaluation run.
3. After the freeze, a change to any frozen item makes a new method version with its own record. Evaluation results already produced stay in the record and are reported.
4. Never tune, select or change anything based on test-set outcomes. Validation outcomes are used only as the lead decides below.

**Lead decision.** Where the gate threshold is chosen. The proposed default is
development (train) data only, with validation kept for one confirmation run.
The earlier plan chose it on validation.

## 10. Inspection procedure for the 20 pilot cases

The packet is `results/pilots/ohr_dev_v1/inspection/`, with `index.html` and
the blank form `annotations.csv`. Nobody fills the form during design work.

1. Open the case in `index.html`. Compare the page image with the MinerU text. Do not open the ground-truth reference yet.
2. Mark each defect you see: word corruption, lost or misleading structure, missing content, or apparently adequate OCR. Mark more than one if several apply. Use uncertain or mixed when you cannot decide, and explain in `uncertainty_note`.
3. Then read the question and the reference answer, and judge whether the evidence the question needs survives in the noisy text: survives, damaged, lost, or uncertain. The current form has no column for this. Record it at the start of `notes` as `evidence: <value>` until a versioned form adds one.
4. Put your name or initials in `annotator`. Never fill the form with a model's suggestions. Keep any model-generated note in a separate file, marked as such.

Controller code exists: the gate and diagnosis in `src/faar/quality.py` and
the routing graph in `src/faar/graph.py`. It is not connected to the pilot
manifests, and no registered run has produced a prediction for the pilot cases, so no
controller output can reach annotators yet. The offline engineering runner
(`src/faar/pilot_runner.py`) runs no gate, diagnosis or repair. Once the
controller runs on the pilot, annotators must not see its output.

These 20 cases were chosen purposively to cover categories. They are for
development inspection only, and their proportions do not estimate how often
failures occur in the dataset.

For the eventual diagnosis study, the text-defect and evidence-impact
observations need at least two people labelling independently, blind to the
controller, with agreement reported. An agent's own labels are not independent
validation.

## 11. Implementation gaps

The table compares the intended behaviour with the code at `8832ce4`.

| Intended behaviour | Current behaviour (evidence) | Required later change | Evidence to accept it |
| --- | --- | --- | --- |
| Search within the question's document | One retriever over the whole registered split corpus (`graph.py:75-81`, `graph.py:90`, `benchmarks.py:144-162`) | Build the retrieval corpus per document from the page inventory | A test in which every hit has the question's `doc_id` |
| Run the pilot from its manifests | No code reads `runtime_manifest.json` | A loader for pilot manifests and MinerU text | A test that no evaluation field is reachable from the loaded example |
| Keep gold data out of runtime | Runtime records carry `correct_answer` and evidence `page_ids` (`benchmarks.py:349-357`) into the example the graph holds (`benchmarks.py:129-142`, `graph.py:84-86`) | A runtime example without gold fields; scoring joins the evaluation manifest afterwards | A structural test on the runtime example type |
| Empty OCR is a condition | Builder rejects empty pages (`benchmarks.py:456-457`); empty pages give no chunk (`chunking.py:42-43`) | Keep empty pages in the inventory and addressable as images | Pilot build with its 6 empty pages recorded, not rejected |
| Explicit no-evidence outcome | Empty answer string (`answering.py:73`); retry repeats the same query (`recovery.py:546-547`) | Return `no_evidence` and abstain | Test with a document whose pages are all empty |
| One answer model for all conditions | Rule-based text extractor (`answering.py:50-77`) against a VLM | Common answer model and prompt family | Run records showing the same model identity in every condition |
| Same evidence for visual and text answers | OpenAI prompt is image only (`recovery.py:291`); the Anthropic path adds up to 4,000 characters of retrieved text (`recovery.py:421-424`) | Same text in both conditions, with images added | Prompt fixtures per condition |
| Declared image budget | All retrieved-hit images, no count cap, no `detail` setting | Page-selection policy and budget | Budget recorded in every run record |
| Diagnosis as signals | Top-1 chunk only; any one layout signal gives `structural` (`quality.py:59-62`, `quality.py:99-100`, `structural_threshold` default 1 in `settings.py`); `semantic` is the residual | Report signals separately, and validate against human observations | Per-signal agreement table on inspected cases |
| Gate threshold from development data | Top reranker score against a default of 0.5 (`quality.py:79-82`, `settings.py:183-186`) | Choose the threshold on development data under the logging rules | Registry record of the selection run |
| Official scoring | `metrics.py:8-15` erases non-ASCII answers | Use `OHR-Bench/src/metric/common.py` as primary | Scoring tests on Han, numeric and empty cases |
| Damage rate on initially correct cases | `_harm_rate` counts any F1 drop over all rows (`final_analysis.py:124-144`) | Denominators as in section 8 | Test with hand-computed counts |
| Finish runs with recorded failures | Runner raises after any failure (`experiment_runner.py:220-225`) | `execution_failed` rows and a completed or failed run status | Test with an injected API failure |
| Enforced cost matching | Random recovery draws per question with no request budget (`graph.py:30-40`, `graph.py:124-131`) | Visual-request cap on the flagged set | Run records with matched request counts |
| CI covers the tested code | CI installs `pip install -e . pytest ruff` (`.github/workflows/tests.yml:26`). Run 36459962433 on `8832ce4` (log read 2026-09-28): 18 failed, 863 passed, 6 skipped. 10 failures are `No module named 'pypdfium2'` (`tests/test_arxivqa_prepare.py`, `tests/test_external_assets.py`), 7 are in `tests/test_preflight_checks.py`, where the `dependencies` preflight check reports `docling` and `pypdfium2` missing, and 1 raises for a missing `OPENAI_API_KEY` (`tests/test_recovery_hardening.py`). The previous CI run on `main`, 31318349678 on `253b255`, passed. The 131 commits between the two were pushed together and had no CI run of their own, so the failures are cumulative and are not attributable to `8832ce4`, which changed only documentation | Declare and install the dependencies the tests need, and isolate tests from credentials | A green CI run on the branch |

### Changes on the pre-baseline engineering branch

The branch `research/prebaseline-engineering` added an offline engineering path
and changed the test setup. It did not change the controller, the graph or the
existing runner. Against the table above:

- **Search within the question's document, run the pilot from its manifests, keep gold data out of runtime.** `src/faar/pilot_runner.py` loads only `runtime_manifest.json` and the MinerU files, verifies their hashes, builds one retriever per document and refuses a hit from another document. Its runtime question type holds only `question_id`, `doc_id` and `question`. Tests in `tests/test_pilot_runner.py` check that generation opens no evaluation file and that poisoned evaluation files leave predictions byte-identical. The graph path (`graph.py`, `benchmarks.py`) is unchanged.
- **Empty OCR and the no-evidence outcome.** In the offline path, empty and missing pages stay in the per-question OCR record and produce no chunk. A question with no usable text gets `no_evidence` with an abstention and a `no_evidence_reason`. The reasons are `no_text_chunks` (every page empty or missing), `no_text_content` (the text is non-empty but holds no letter or digit in any script, such as only Markdown heading markers), `no_retrieval_tokens` (the text has letters or digits, but none that the engineering tokeniser indexes, such as Han-only text) and `no_hits`. Only the first is an OCR status. The runner has no image path, so it shows no page as an image.
- **Questions the engineering retriever cannot read.** Every record carries `query_retrieval_tokens`, and `generation_summary.json` counts questions with none. Such a question, for example one written only in Han script, gets hits ranked by position alone.
- **Official scoring.** `src/faar/ohr_scoring.py` (section 8). `src/faar/metrics.py` is unchanged and must not serve as the primary metric.
- **Finish runs with recorded failures.** The offline runner records `execution_failed` per question, finishes the run and exits with status 2 when any failure exists. `experiment_runner.py` still raises.
- **CI.** Dependencies are declared (`pypdfium2`, `jieba`, `regex`, pinned `click`), the tests no longer need credentials or network, and the workflow has separate lint, package and offline-test jobs.

The remaining rows are unchanged.

## 12. Decisions

| Decision | Status |
| --- | --- |
| OHR-Bench first; other datasets and collection-wide retrieval later | Agreed |
| MinerU as the fixed noisy input, version unknown and disclosed | Agreed |
| `ohr_dev_v1` frozen; development sample only | Agreed |
| First pilot searches within each complete document | Agreed |
| Gold data outside runtime | Agreed |
| Empty OCR recorded, not excluded | Agreed |
| Failed and negative runs kept | Agreed |
| No tuning on test outcomes | Agreed |
| Three observations kept apart; per-signal comparison with people | Proposed default |
| Shared retrieval, initial answer, gate and budgets across policies | Proposed default |
| Report end-to-end and flagged-set views | Proposed default |
| One answer model, `gpt-4o-2024-11-20`, for text and visual conditions | Proposed default; spending needs lead approval |
| Matching on visual requests with an identical image budget, reported as visual-request matching, not total-cost matching | Proposed default |
| Budget exhaustion keeps the initial answer and records `budget_exhausted` | Proposed default |
| Development all-repair results may inform method development, with each change logged | Proposed default |
| Method frozen and registered before any validation or test run | Proposed default (no test-set tuning is agreed) |
| `no_evidence` outcome distinct from `execution_failed` | Proposed default |
| Execution failures retried, then counted as incorrect and reported | Proposed default |
| Official OHR-Bench EM and F1 primary; FAAR metrics secondary and labelled | Proposed default |
| Document-level resampling for uncertainty | Proposed default |
| Evidence-impact judgement recorded in `notes` until a versioned form exists | Proposed default |
| Answer prompt and API spending | Lead decision |
| Image budget and page selection for documents longer than the budget | Lead decision |
| Where the gate threshold is chosen | Lead decision |
| Maximum execution-failure rate for a valid run | Lead decision |
| A second, longer-document development sample | Lead decision |
| When to run the all-repairs diagnostic (section 6) | Lead decision |
| Claim scope: new questions on familiar documents, or unseen documents (needs a document-disjoint protocol) | Lead decision |
| Licence terms for redistributing the reimplemented OHR-Bench scoring code outside the project | Lead decision |

## 13. Historical approaches

[aaai-plan.md](aaai-plan.md) is the earlier AAAI plan, kept as evidence.
Where it agrees with this brief, it still applies: the fixed split, the model
pins, logging cost and runtime, and the before-submitting checks. Its B0, B1 and
B2 baselines correspond to no recovery, visual recovery and random recovery in
section 5. B3 and B4 (ColPali and VisRAG retrieval) are later work. These instructions are retired:

- "If FAAR does not beat B2 by a clear margin, fix the diagnosis module before proceeding." This conflicts with the rule against changing an experiment until it gives a positive result.
- Labelling each case as exactly one of `semantic`, `word_level`, `structural` or `other`. This conflates observation with repair choice (section 3).
- Using GOT-OCR as the noisy input for OHR-Bench in the first study. MinerU is the agreed input. GOT-OCR remains for the calibration job.
- Running FAAR on the test split and on three datasets as the next step.
- Gate-threshold targets set before any data (precision 0.75, recall 0.70).

The prototype's 40-example mock evaluation and the August status report are in
[docs/history/](../history/README.md).

## 14. Next assignment

**Status.** The three engineering steps of the earlier assignment (environment,
scoring, offline run) are implemented on `research/prebaseline-engineering`.
The environment step was accepted only when CI passed on the branch. The
[pre-baseline engineering report](../reports/prebaseline-engineering.md) lists
the checks, the registered runs and what they cannot show. The offline run uses
the rule-based extractor and no repair. It is an `engineering_check` of the
data flow, the record format and the scoring join. It is not a no-recovery
baseline and not evidence about answer quality.

Before any real-model run, the lead decides the items marked "Lead decision" in
section 12. The ones that block a first no-recovery baseline are:

1. The answer model, its provider, the prompt and the token limits.
2. The spending limit and the cost-accounting rules.
3. The image budget and the page-selection policy for documents longer than the budget.
4. The maximum execution-failure rate for a valid run.

The gate threshold, the repair-comparison protocol (section 6), a longer-document
development sample and the claim scope follow after the baseline.
