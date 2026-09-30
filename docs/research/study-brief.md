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
engineering branch changed since then. Section 15 proposes the protocol for the
first real baseline and lists the decisions it needs.

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
functions. On 2026-09-29 it found 0 mismatches in 99,045 pairs: 107
hand-written edge cases, 11 variants of each of the 8,498 `qas_v2.json`
references, and 8 variants plus all cross pairs of the 70 pilot references.
The [scorer provenance record](../reports/ohr-scorer-provenance.md) traces each
function to SQuAD v1.1, HotpotQA or OHR-Bench.
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
3. Then read the question and the reference answer, and judge whether the evidence the question needs survives in the noisy text: survives, damaged, lost, or uncertain. The current form has no column for this. Record it at the start of `notes` as `evidence: <value>` until a versioned form adds one. For a case with several page rows, write it on the first row only. Section 15.14 lists the 20 cases and proposes a versioned form.
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
- **Empty OCR and the no-evidence outcome.** In the offline path, empty and missing pages stay in the per-question OCR record and produce no chunk. A question with no usable text gets `no_evidence` with an abstention and a `no_evidence_reason`. The reasons are `no_text_chunks` (every page empty or missing), `no_text_content` (the text is non-empty but holds no letter or digit in any script, such as only Markdown heading markers), `no_retrieval_tokens` (the text has letters or digits, but none that the engineering tokeniser indexes) and `no_hits`. Under the `multilingual-v1` tokeniser that the runner now uses, Han text is indexed, so `no_retrieval_tokens` is almost unreachable. Only the first is an OCR status. The runner has no image path, so it shows no page as an image.
- **Multilingual retrieval.** Known limit: Python's `\W` treats combining marks as non-word characters, so Devanagari, Arabic with vowel marks and Hebrew with niqqud split into single-letter tokens. The pilot has none of these scripts. The runner tokenises with `multilingual-v1` (NFKC, casefold, Unicode word tokens and CJK character bigrams) and chunks with `cjk-weighted-words-v1` (two CJK characters count as one word, so an unspaced Chinese page splits into chunks of at most 360 characters). Both are defined in `src/faar/text_units.py`. The graph path keeps the original ASCII tokeniser and whitespace chunks. Every record carries `query_retrieval_tokens`, and `generation_summary.json` counts questions with none. In r3, 19 of 70 questions had none. In r4, none has zero. Having tokens does not show that retrieval finds the evidence.
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
| One answer model, `gpt-4o-2024-11-20`, for text and visual conditions (options and recommendation: section 15.5) | Proposed default; spending needs lead approval |
| Matching on visual requests with an identical image budget, reported as visual-request matching, not total-cost matching | Proposed default |
| Budget exhaustion keeps the initial answer and records `budget_exhausted` | Proposed default |
| Development all-repair results may inform method development, with each change logged | Proposed default |
| Method frozen and registered before any validation or test run | Proposed default (no test-set tuning is agreed) |
| `no_evidence` outcome distinct from `execution_failed` | Proposed default |
| Execution failures retried, then counted as incorrect and reported | Proposed default |
| Official OHR-Bench EM and F1 primary; FAAR metrics secondary and labelled | Proposed default |
| Document-level resampling for uncertainty | Proposed default |
| Evidence-impact judgement recorded in `notes` until a versioned form exists | Proposed default |
| First baseline: text-only, no recovery, real answer model, `development_pilot` run on `ohr_dev_v1` (section 15.1) | Proposed default |
| The text-only baseline does not wait for the image budget decision (section 15.1) | Proposed default |
| Runtime and evaluation inputs of the baseline (section 15.2) | Proposed default (applies the Agreed gold-data rule) |
| Determinism settings, replay of saved responses, and per-attempt request records (sections 15.4 and 15.8) | Proposed default |
| Prompt structure, `NO_ANSWER` abstention token, answers in the evidence's language (section 15.7) | Proposed default |
| Token limits, timeout and retry values (section 15.7) | Proposed default |
| Later policies reuse the baseline's retrieval and initial answers by run ID (section 15.9) | Proposed default |
| Exact answer prompt text, spending cap value and API spending (section 15.7) | Lead decision |
| Image budget and page selection for documents longer than the budget | Lead decision |
| Where the gate threshold is chosen | Lead decision |
| Maximum execution-failure rate for a valid run (suggestion in section 15.7) | Lead decision |
| A second, longer-document development sample | Lead decision |
| When to run the all-repairs diagnostic (section 6) | Lead decision |
| Claim scope: new questions on familiar documents, or unseen documents (needs a document-disjoint protocol) | Lead decision |
| Licence terms for redistributing the reimplemented OHR-Bench scoring code outside the project | Lead decision |
| Create a versioned annotation form with an `evidence_impact` column (section 15.14) | Lead decision |

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

Section 15 proposes a protocol for the first baseline and lists the decisions the
lead must make. It approves nothing.

Before any real-model run, the lead decides the items marked "Lead decision" in
section 12. The ones that block a first no-recovery baseline are:

1. The answer model, its provider, the prompt and the token limits.
2. The spending limit and the cost-accounting rules.
3. The image budget and the page-selection policy for documents longer than the budget. Section 15.1 proposes that a text-only baseline does not need it. A visual repair does.
4. The maximum execution-failure rate for a valid run.

The gate threshold, the repair-comparison protocol (section 6), a longer-document
development sample and the claim scope follow after the baseline.

## 15. First real baseline: proposed protocol

**This section is a proposal.** The lead has approved none of its choices. It marks a rule Agreed only where sections 2 to 14 already agree it. It authorises no model call, no credentials and no spending. It rests on the code at `8479309` and on vendor documentation read on 2026-09-29. Every choice carries a label, and the decisions it leaves open are collected in section 15.13.

### 15.1 What the first baseline measures

**Proposed default.**

- The first baseline is the "no recovery" policy of section 5 with a real answer model. For each of the 70 questions of `ohr_dev_v1` it retrieves within the question's document, sends the retrieved noisy text to one answer model, and scores the reply with the official OHR-Bench EM and F1 (section 8).
- It is a `development_pilot` run (`experiments/README.md`), not a `scientific_evaluation`. It measures how one answer model reads the current noisy text under one retrieval setting. It says nothing about diagnosis or repair.
- It sends text only. No image leaves the machine, so the image budget and page-selection decision (section 14, item 3) do not block it. That decision still blocks any visual repair.
- Its answers become the shared initial answers of section 5. Later repair policies cite this run and never regenerate its answers (section 15.9).
- A question with no usable text evidence (7 of 70 in r3: 6 empty pages and 1 page with no letter or digit) gets `no_evidence` with an abstention and no model call, as in the offline runner. It costs nothing and stays in every denominator.

### 15.2 Runtime inputs and evaluation inputs

**Proposed default.** This applies the Agreed rule that gold data stays outside runtime (section 2).

| Runtime (the answer backend may receive) | Evaluation only (joined after the run) |
| --- | --- |
| Question text, document identity, page inventory with per-page OCR status | Gold answer (`answers`), `answer_form`, `evidence_context`, evidence pages |
| Retrieved chunk text, page number and chunk ID | Clean reference text (`gt_reference`, `reference_text`) |
| Prompt template, model identity, request settings, price table | Analysis flags: `evidence_source`, `question_scripts`, `multi_page_evidence`, `list_valued_evidence`, `evidence_page_ocr_status` |
| | Inspection labels and `annotations.csv` |

The backend receives `(question, hits)` and nothing else. It opens no file. The document name stays out of the prompt. Only the page number and a document-free chunk label (the chunk ID without its `<doc_id>-` prefix, for example `p2-c2`) identify the evidence. Before 2026-09-29 (template `56afceb6...`) the header showed the full chunk ID, which contains the document name; the reviewer found this and the template was changed. Answer form, evidence type and script are evaluation fields, so the prompt cannot vary with them.

### 15.3 Within-document retrieval

**Agreed:** the search covers all pages of the document the question names (section 2). The rest is **Proposed default**, stated as policy.

- One index per document, built from the noisy text of every page. A question never sees another document. Empty and missing pages produce no chunk and stay in the per-question OCR record.
- The retrieval settings (chunk size, overlap, top-k, embedding and ranking method) are fixed and recorded before the run, hashed into the run fingerprint, and reused for every later policy. The current values are engineering settings and are not approved (`run_config.json` says so).
- The retriever must give Han-script and mixed-language questions a real ranking signal before the first real run. In r3, 19 of 70 questions had no indexed query token and ranked by position alone. The `multilingual-v1` tokeniser and `cjk-weighted-words-v1` chunking, used in the engineering run r4, give every pilot question query tokens. Whether the retrieved chunks hold the evidence is not measured, and the lead approves the retrieval settings before the run.
- The retrieval result is saved once per question (chunk IDs, page indexes, scores, and a hash of each chunk text). The prompt is built from the saved result.
- 23 of 30 pilot documents are single pages, so page coverage is 1 by construction there (section 3). The baseline can still miss the passage inside the page.

### 15.4 Answer-model interface

**Implemented for offline verification (2026-09-29); nothing here is approved for a live run.** The answer-model path is separate from the offline engineering runner, whose engineering-only restriction is unchanged.

- `src/faar/live_runner.py` with `scripts/experiments/run_pilot_live.py` drives the run. Its subcommands are `dry-run`, `run`, `status`, `reconcile`, `reopen`, `export` and `score`.
- `src/faar/answer_prompt.py` builds the draft prompt (`faar-answer-draft-v1`) and parses replies. `src/faar/answer_providers.py` holds the provider interface, a scripted `FakeProvider` and an `OpenAIChatProvider` adapter that is tested only against a mocked transport. `src/faar/request_budget.py` bounds and prices requests, and `src/faar/retry_policy.py` decides retries.
- **Fake mode is the default.** It sends nothing, and its runs are `engineering_check`.
- **Live mode needs all of the following:** `--mode live`, `--provider-config PATH` (provider, model, endpoint, parameters, output limit, timeout, dated price table and `tokenizer_bound: utf8-bytes`), `--safety-ceiling` and the environment variable `FAAR_ALLOW_LIVE_REQUESTS=I_UNDERSTAND_THIS_SPENDS_MONEY`. Credentials are read only after every check passes. Live runs write to `results/development/<run_id>/`, and a live run needs `--run-kind engineering_check` or `--run-kind development_pilot` (no default). No live run has been made.
- The live client never follows HTTP redirects: the adapter refuses a client that would, and any 3xx reply is an unknown outcome (kind `redirect`), since a followed redirect would resend the prompt outside the driver's accounting. API-key values echoed in provider error text are replaced with `[redacted-api-key]` before anything is stored.
- The live client is built with the configured `endpoint` as its base URL, and the run records the base URL the client reports. Live mode refuses to start when `OPENAI_BASE_URL`, `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` or `OPENAI_ORGANIZATION` is set, since the SDK would otherwise redirect requests. The live HTTP client is built with `trust_env=False`, so proxy variables (`HTTPS_PROXY` and similar), macOS system proxy settings and the certificate variables `SSL_CERT_FILE` and `SSL_CERT_DIR` are ignored, not refused. The adapter and the builder refuse an injected client that trusts the environment, follows redirects or routes through a proxy of its own (`proxy=`, `mounts=` or a proxy transport). They check when the provider is built, so a client changed afterwards is not seen. The CLI never injects a client. A machine that reaches OpenAI only through a proxy or a TLS-intercepting gateway gets connect errors, and nothing is sent. Request parameters must be in a fixed allowlist (`temperature`, `top_p`, `seed`, `stop`, `presence_penalty`, `frequency_penalty`).
- **Storage and service tier (implemented 2026-09-30).** Every request sends `store: false` and `service_tier: "default"`, so neither depends on account or project settings. The provider config may state `"store": false` and `"service_tier": "default"`, and refuses any other value. A real price table must declare `"service_tier": "default"`, the tier its rates apply to. Storage for attempt lookup (`store: true` plus metadata) exists in the adapter only, and this CLI refuses it until an explicit authorization adds it. `store: false` is not zero retention: OpenAI keeps abuse-monitoring logs for up to 30 days unless the organization has an approved Zero Data Retention or Modified Abuse Monitoring control. It also does not settle whether the dataset terms allow sending the documents.
- **Run kind and manifest provenance (implemented 2026-09-30).** The identity records `run_kind` and a `pilot_manifest` block: the parent pilot, the SHA-256 of the frozen `results/pilots/<pilot_id>/runtime_manifest.json`, the SHA-256 of the manifest the run read, `canonical` (true only when those two byte hashes are equal), the question count and a digest of the question ids in manifest order. Fake runs are always `engineering_check`. A `development_pilot` on a non-canonical manifest is refused before any directory is created. The summary, status and score summary repeat the kind and the canonical flag, and `registry.py check` refuses a record whose kind differs from its `run_config.json`.
- Importing the modules, `--help`, `dry-run`, `status` and `export` build no provider client and make no outbound connection. A third-party import binds a loopback socket, which is not a connection.
- **Freeze the code before the first paid request.** Code identity covers every `src/faar/*.py` file and the CLI script. Any later edit, even an unrelated one, makes a partly paid run refuse to resume, so it would need a new run.

**Identity record.** `run_config.json` records the run identity, and a resume recomputes it and refuses any difference before dispatching. The identity covers:

- the runtime manifest and noisy-text hashes, and the retrieval description;
- the prompt-template ID and SHA-256;
- provider, requested model, endpoint, parameters, output and input limits, timeout, and `token_limit_param`;
- the price table and the retry parameters;
- the scientific budget (null for this baseline), code identity and contract version.

The safety ceiling is not identity. The requested and the returned model are recorded separately for every response.

**Reproducibility.** A seed or `temperature` 0 does not make a model reproducible, and several current models accept neither. The run keeps every raw response and never regenerates a saved answer. Reproducibility therefore rests on the saved responses and the recorded method identity. The proposed repeat probe (10 questions sent 3 times each) still measures how often answers differ.

### 15.5 Answer-model options

Sources were read on 2026-09-29. Prices are US dollars per 1M tokens. "Snapshot" means whether the documentation says the ID names fixed weights.

| | A. `gpt-4o-2024-11-20` (OpenAI) | B. `claude-sonnet-5-5` (Anthropic) | C. `gpt-6-luna` (OpenAI) |
| --- | --- | --- | --- |
| Text and image input | Yes, text output | Yes, text output | Yes, text output |
| Context, max output | 128,000; 16,384 | 1M; 128K | 1,050,000; 128,000 |
| Input, output price | $2.50; $10.00 (cached input $1.25) | $2; $10 (cache read $0.20) | $0.10; $0.50 (cached input $0.01) |
| Determinism controls | `temperature`, `seed` (best effort, marked deprecated), `system_fingerprint` (marked deprecated) | None. A non-default `temperature`, `top_p` or `top_k` returns a 400 error. Adaptive thinking is on | `temperature` and `top_p` only when `reasoning_effort` is `none`. `seed` appears only in the generic Chat Completions reference |
| Snapshot stability | Dated ID. Listed as a GPT-4o snapshot. Not in the deprecations table. The sibling `gpt-4o-2024-05-13` shuts down 2026-10-23 | Each model ID is a pinned version. Retirement not sooner than 2027-09-28 | The model page lists only the undated ID `gpt-6-luna`. No dated snapshot is documented there |
| Sources | [model page](https://developers.openai.com/api/docs/models/gpt-4o), [deprecations](https://developers.openai.com/api/docs/deprecations), [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create) | [model page](https://platform.claude.com/docs/en/models/sonnet-5-5/overview), [model IDs](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions), [deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations) | [model page](https://developers.openai.com/api/docs/models/gpt-6-luna), [GPT-6 guide](https://developers.openai.com/api/docs/guides/latest-model) |

Notes on the table.

- On 2026-09-29 the OpenAI pricing page did not show text rates for `gpt-4o`, so the rates in column A came from its model page. They match `openai_cost_rates()` in `src/faar/api_logging.py:93-99`. On 2026-09-30 the pricing page listed a Standard `gpt-4o` row with the same rates ($2.50 input, $1.25 cached input, $10.00 output) and a Fast row at 1.7 times those rates. It has no separate row for the dated snapshot. Record the rates with the run, with their source, date and service tier.
- Option A is not marked legacy or deprecated on its model page. The documentation does not show whether the lead's account can call it. Only a call shows that, and no call was made.
- The repository's Anthropic path names `claude-sonnet-4-5` (`settings.py`). Anthropic lists `claude-sonnet-4-5-20250929` with a tentative retirement of not sooner than 2026-09-29. Do not use that path for the baseline.

**Recommendation (Proposed default, not approved): option A.** It is the only option that takes `temperature` 0 and a seed with no other setting, on a dated snapshot, so repeatability can be measured. It is already pinned in the code (`settings.py:226`, `settings.py:306-308`) and in the brief, so nothing changes in the record. The design-time estimate in section 15.6 is about $0.40 for the run. The dry run on 2026-09-30 gives a lower estimate of about $0.24 (docs/reports/live-baseline-readiness-2026-09-30.md, section 5). Judgment: it is an old model, and OpenAI is already retiring other GPT-4o snapshots. A retirement of `gpt-4o-2024-11-20` would force every later condition onto a successor and a rerun of the baseline. Option C is the cheap modern alternative, with `reasoning_effort` `none` so that `temperature` 0 is accepted, but no dated snapshot is documented. Option B documents a fixed version and a retirement date a year away, but it accepts no sampling control, and its thinking tokens bill as output. The choice matters less than using one model for every condition and recording it.

### 15.6 Cost estimate

For one run over `N` questions:

`cost = Σ over requests (t_in × r_in + t_out × r_out) / 1,000,000`

`expected cost ≈ N_model × (1 + ρ) × (T_in × r_in + T_out × r_out) / 1,000,000`

`ceiling = N_model × A_max × (T_in,max × r_in + M_out × r_out) / 1,000,000`

`N_model` is the number of questions that reach the model. It is 63 in r3, because the 7 `no_evidence` questions make no call. `ρ` is the mean number of billed retries per question. `A_max` is the attempts allowed per question. `T_in` and `T_out` are mean input and output tokens per request, `T_in,max` is the largest prompt, `M_out` is the output limit, and `r_in`, `r_out` are the recorded rates.

**Worked example (an estimate, not a measurement), option A.**

| Assumption | Value | Basis |
| --- | --- | --- |
| `N_model` | 70 | Overstates: r3 sent 63 questions to the answer step |
| Evidence tokens | about 1,700 mean, 7,500 largest | Measured on 2026-09-29 as characters in the 5 retrieved chunks per question with the r3 chunker and retriever (mean 3,961 characters, maximum 8,545), converted at 4 characters per token for non-Han text and 1 token per Han character. The conversion is a rough assumption. With the r4 multilingual policy the figures are mean 3,190 and maximum 6,252, so the estimate below errs high |
| Instructions and question | 300 tokens | Assumed prompt length |
| `T_in`, `T_in,max` | 2,000 and 7,800 | Sum of the two rows above, rounded |
| `T_out`, `M_out` | 20 and 128 | Reference answers have a median of 2 words (non-Han), 13 of 55 have 8 or more |
| `ρ` and `A_max` | 0.1 and 3 | Assumption. Timeouts may be billed |
| Rates | $2.50 and $10.00 | Section 15.5 |

- Expected: 70 × 1.1 × (2,000 × 2.50 + 20 × 10) / 1,000,000 = 77 × 0.0052 = about $0.40.
- Ceiling: 70 × 3 × (7,800 × 2.50 + 128 × 10) / 1,000,000 = 210 × 0.02078 = about $4.36.
- The same inputs at option C's rates give about $0.02 expected. At option B's rates they give about $0.32 before thinking tokens. Thinking tokens bill as output, so B's cost grows with the effort setting.

Two things change these numbers. The Han-script text (21 of 63 questions had Han in the retrieved text) may cost more than one token per character, and the r4 retrieval change shortened the evidence (see the evidence-token row). Recompute with a dry run before any spending (section 15.15). The implemented dry run reports a conservative upper bound, not an estimate. With the r4 retrieval (dry run of 2026-09-30 at code `884e388`), the largest prompt bound is 7,920 tokens, the one-attempt bound over the 63 requests is $0.949909, and the worst case over 3 attempts each is $2.849727 at option A's rates. An earlier dry run on 2026-09-29, before the r4 retrieval, gave 8,237 tokens and $2.94. A provider-tokenizer count, which is closer to the real cost, needs that tokenizer offline and is not implemented.

### 15.7 Prompt, answer format and failure handling

**Prompt structure (Lead decision on the exact text; the structure is a Proposed default).**

- One system message holds the rules. One user message holds the question and the evidence as numbered blocks, each headed by its page number and document-free chunk label, in rank order. No gold field, document name or answer form appears.
- Rules: use only the evidence; reply with the shortest answer that a reader can check, copied from the evidence where possible; for a yes/no question reply `Yes` or `No`; for a list, give the items separated by commas; if the evidence does not contain the answer, reply with the abstention token `NO_ANSWER` and nothing else.
- Language: copy the answer in the language and script of the evidence and never translate. This handles Chinese: the pilot has 22 Han-script questions but 15 Han-script references, so the question language does not predict the answer language.
- Extraction: `answer` is the reply after removing surrounding whitespace and one leading `Answer:` label. The raw reply is stored unchanged. An exact reply of `NO_ANSWER` becomes `answer` = `""` with `abstained` = true and status `answered`, which the scorer reports under `abstained` and not under `no_evidence`. Any other text is the answer, even when it looks like a refusal.
- Short answers suit most pilot references (median 2 words, 33 of 55 non-Han references have 3 words or fewer) and cost some F1 on the 13 with 8 or more words. That trade-off is part of what the development run shows. Prompt changes make a new run (section 9).

**Limits, timeouts and retries (Proposed default; values reuse the existing recovery code).**

| Setting | Value |
| --- | --- |
| Output limit | 128 tokens for option A. A reasoning model needs a larger limit because thinking tokens count |
| Input limit | 12,000, compared with the conservative UTF-8 byte bound of the prompt (rule below), not with a tokenizer count. Han-heavy prompts reach the limit at about a third of their real token count. The largest pilot bound is 7,920 (dry run of 2026-09-30), so no prompt is skipped. Whether to keep this rule is a **Lead decision** |
| Request timeout | 60 seconds (`FAAR_VLM_TIMEOUT_SECONDS` default) |
| Attempts per question | 3, with a base backoff of 2 seconds that doubles, at most 60 seconds, plus deterministic jitter (`faar.retry_policy`) |
| Retried | only failures with evidence that the request was not processed (the two HTTP cases still accept a small residual risk of a second charge, since neither source speaks to billing and an intermediary can also send a 408): connect errors and connect or pool timeouts (the request never left the machine), HTTP 408 (RFC 9110 section 15.5.9: the server did not receive a complete request) and HTTP 503 with code `server_is_overloaded` (OpenAI error-codes page: the model "does not have enough capacity to process your request", read 2026-09-29) |
| Not retried | every failure whose outcome is unknown, which waits for reconciliation: read or write timeouts, a connection lost after sending, HTTP 3xx (redirects are not followed), 429 (rate limit), 409, 500, 502, other 503, 504, 522, 524 and other 5xx, a 200 reply without an answer message, an unparsable reply. A provider's advice to retry is not evidence that the request was not processed or billed. Also not retried: other 4xx errors, which are rejected and fail the question |

**Outcomes.**

- A prompt over the input limit is not truncated and not sent. It is recorded as `execution_failed` with reason `prompt_over_limit`.
- A non-retryable error on a question is `execution_failed` with the error type. The first authentication, unknown-model or quota error stops the invocation (`stopped`), since every later request would fail too. Unsent questions stay pending, and a later `run` resumes them after the cause is fixed. A stopped run is incomplete and is not a baseline.
- **Unknown outcome.** A request whose dispatch started but whose result is unknown is never sent again automatically. The provider payload of such an attempt, if any, is kept in the `outcome_unknown` event as billing evidence. That covers a read timeout, a connection lost after sending, a crash before the response was saved, and the HTTP statuses listed as not retried above (for example a 502, where a gateway got an invalid reply from the provider, RFC 9110 section 15.6.3). OpenAI documents no idempotency key for Chat Completions, so the provider cannot deduplicate a resend. The question waits in `needs_reconciliation`, and the other questions continue. A person resolves it with `reconcile ATTEMPT_ID --resolution allow_new_attempt` or `mark_failed`, with a note. The unknown attempt's cost stays counted at its upper bound either way. The path gives safe local resume. It gives no exactly-once guarantee on the provider's side.
- **Circuit breaker.** Three consecutive counted attempts stop the invocation (end reason `circuit_breaker`). A counted attempt is an unknown outcome, a failure that ended its question (rejected, or never sent after its retries, as in a network outage) or an unexpected exception. Only a saved response resets the count. The run state is `needs_reconciliation` when a counted attempt has an unknown outcome, and `stopped` otherwise. This stops a systematic fault from spending the ceiling on reserved cost. The threshold is part of the run identity.
- **Reopening a failed question.** After fixing the cause of an `execution_failed` question, `reopen --run-dir DIR QUESTION_ID --note TEXT` makes it pending again with up to `max_attempts` further attempts. Earlier attempts and their costs stay in the log. Answered, unserved and unknown-outcome questions cannot be reopened, since they resume, or need `reconcile`, instead.
- **Rate limits.** Because OpenAI does not document that a rate-limited request was not processed, a 429 is an unknown outcome. Each one needs `reconcile`, and three in a row trip the circuit breaker. Treating 429 as retryable is a one-line change in `faar.answer_providers`, and it is a **Lead decision**, since it accepts a possible second charge on an undocumented assumption.
- **Safety stop.** Before every dispatch the driver checks the saved records for a violated assumption:
  - reported input tokens above the request's upper bound;
  - a returned model that differs from the configured one;
  - a returned service tier that is not `default`, including a missing or unrecognised value (`returned_service_tier_unverified`);
  - any anomaly in the safety ledger.

  On a violation it keeps the offending response and sends nothing more. It ends the invocation as `safety_stop`, the run state becomes `safety_stopped`, and it exits with code 6. Unserved questions are exported as `execution_failed` with `unserved: true` and failure type `safety_stop`, never as abstentions. A restart rebuilds the violation from the records and exits 6 before building a provider. If a crash left a response file without its event, the restart appends the recovered `response_saved` event and writes the exports, so the records name the stop. A resume, a ceiling raise with a note, `reconcile` and `reopen` do not clear it. Continuing needs a successor run with corrected code or configuration, and the identity checks require that anyway. Commands refuse a run whose `run_config.json` identity block no longer matches its hash, or whose `response_saved` events disagree with the hash-pinned response files. The event log has no hash chain, so this protects against the code's own behaviour and honest operation, not against edits that keep every cross-check consistent. The returned-model check is an exact string match, so configure the dated snapshot that the provider returns. An alias such as an undated model name would stop the run after its first response. `score` refuses a safety-stopped run, because the run is not `complete`. Stopping cannot undo a charge already made. The ceiling holds only as far as the provider reports usage and the bounds are valid.
- **Scoring is final.** `score` runs only on a `complete` run. After scoring, `run`, `reopen` and `reconcile` refuse, and `export` only confirms identical files.
- An empty reply, a reply cut by the output limit (`finish_reason` `length`) or a refusal is an answer, scored as returned, with `output_status` set to `empty`, `truncated` or `refusal`. It is not an API failure.
- No question is dropped. Every question ends in `answered`, `no_evidence` or `execution_failed`. In end-to-end accuracy `execution_failed` counts as incorrect and its rate is reported (sections 7 and 8).
- The maximum execution-failure rate for a valid baseline is a **Lead decision**. Suggestion: any `execution_failed` question is retried as a new attempt after its cause is fixed, and a run with more than 3 of 70 (about 4%) still failing is `failed`. Failures in the shared initial answers propagate to every later policy.

**Two kinds of budget.** They are kept apart.

- **Safety ceiling.** It is an operating limit that protects against runaway spending. It is not part of the method, so it stays out of the run identity. Each invocation records it in `attempts.jsonl` (`invocation_started.safety_ceiling`). The value is a **Lead decision**, and a suggested first value is $2.00 for the main run. This task authorises no monetary value.
- **Scientific budget.** It is a policy budget that the method uses, for example the visual-request cap in section 5 or a per-question repair budget. It belongs to the method configuration and the run identity, and changing it makes a new run. The no-recovery baseline has none (`scientific_budget: null`).

**Safety accounting (implemented).**

1. Before each attempt the runner computes an upper bound for its cost. The input bound is the UTF-8 byte count of the messages plus a small overhead. That bounds any byte-level BPE tokenizer, and the provider config must declare `tokenizer_bound: utf8-bytes`. The bound prices every input token at the higher of the input and cached rates, and adds the full output limit at the output rate. The runner refuses to dispatch when a price, a limit or the bound is missing.
2. The ledger adds measured cost to reserved cost. Measured cost uses usage the provider reported, priced with the recorded rates and their source date. Reserved cost counts each dispatched attempt without measured cost at its upper bound: unknown outcomes, rejected attempts and responses with missing usage. An attempt that provably never left the machine counts 0. No usage is invented for a failure, and no cache discount is assumed until cached tokens are reported. Amounts are kept in integer micro-units.
3. An attempt is dispatched only if measured + reserved + its own bound stays within the ceiling. Otherwise the invocation sends nothing more and ends as `budget_limited`.
3a. **Validity flag.** `run_summary.json` sets `valid_baseline` only for a complete live run of kind `development_pilot` on the canonical frozen manifest, with no `execution_failed` question, no returned-model mismatch (exact match with the requested model; name the dated snapshot), no unverified service tier, no input-bound exceedance and no ledger anomaly. It lists the blockers. An `engineering_check` is never eligible. The flag checks mechanical completeness and eligibility. It does not mean that the prompt, model, retrieval or budget are approved.
3b. **Unverified tier cost.** The price table applies only to the Standard tier. A response whose returned tier is not verified keeps its usage in the response file, the event and the prediction, but its `measured_cost` is null and it is left out of `cost.measured`. `cost.unverified_tier` reports its attempts, usage and a Standard-rate figure that is labelled as not an actual-cost claim. Its reservation stays at the Standard-rate upper bound, which is not an upper bound on the real charge when the tier bills more (the pricing page lists Fast `gpt-4o` at 1.7 times Standard). Stopping cannot undo a charge already incurred. An unusable reply (for example a 200 with no answer) that reports a tier other than `default` also stops the run. Its cost stays reserved as an unknown outcome at the Standard-rate bound. A crash between the provider's reply and the saved response file loses the returned tier together with the response, and that attempt waits for reconciliation.
4. **Budget exhaustion.** Completed answers are kept. Each unserved question is exported as `execution_failed` with `unserved: true` and `failure.type: budget_exhausted`. It is never an abstention, and the scorer counts it as 0. The run summary says `budget_limited` and `valid_baseline: false`.
5. **Authorised continuation.** A later `run --raise-safety-ceiling AMOUNT --authorization-note TEXT` records the change (from, to, note) in its `invocation_started` event and resumes the same run. The earlier budget-limited invocations stay in the log. A raise without a note is refused. The note is free text that nothing verifies, so it records an approval and does not replace one.

### 15.8 Records needed for later fair cost comparison

**Implemented (2026-09-29) for fake-provider runs.** Visual-request counts do not match total cost (section 5). A run directory holds these files:

| File | Content |
| --- | --- |
| `run_config.json` | Identity and its SHA-256, the hash of `requests.jsonl`, provenance (commit, dirty paths, package versions, command), mode and kind. Written once |
| `requests.jsonl` | One prepared record per question in manifest order: send or skip and why, evidence chunk IDs, pages and hashes, prompt and evidence SHA-256, exact messages, input bound, output limit, cost bound and `request_id`. It holds the full prompts, so it stays local (git-ignored) and is regenerated byte for byte |
| `attempts.jsonl` | Append-only events (`invocation_started`, `dispatch_started`, `response_saved`, `attempt_failed`, `outcome_unknown`, `reconciled`, `question_reopened`, `invocation_ended`). A safety stop records its `safety_violations` in `invocation_ended`. Failed and unknown attempts keep the provider's request ID and a bounded error body (`provider_raw`). Each line is fsynced before the next step. A dispatch is logged before it is sent |
| `responses/<attempt_id>.json` | The raw provider payload and the parsed answer, abstention flag, output status, usage as reported (missing fields stay null), measured cost, returned model, response ID, finish reason and latency. Written atomically before its event |
| `predictions.jsonl`, `run_summary.json` | Rebuilt from the files above. One record per question with its attempt IDs. The summary has counts, run state and the ledger (measured, reserved, committed upper bound; currency and a `simulated` flag) |
| `scores.jsonl`, `score_summary.json` | The unchanged official scorer. A scored run refuses further `run` |

A lock (`run.lock`, an exclusive kernel lock held for the process lifetime) stops a second invocation from dispatching the same run.

Per question and per policy, report requests, retries, tokens, dollars and latency. Retrieval time and index-build time are one-time local costs, so they are reported separately. A later policy adds its own attempts to the shared initial cost. Report both the total and the incremental cost, and compare policies on totals. Claim matched total cost only under the conditions in section 5.

### 15.9 The later comparison: simple repair and diagnosis-selected repair

**Proposed default.** After the baseline is accepted, the comparison of a simple repair policy (gate-triggered visual recovery or a single fixed text repair) with the diagnosis-selected repair follows section 5:

- The baseline run supplies the shared retrieval and initial answer. Later runs cite its `run_id` and the hash of its `predictions.jsonl`, and reuse the records. If the prompt, the model or the retrieval setting changes, the initial answers are regenerated in a new baseline run and every policy restarts from it.
- The gate runs once on the saved retrieval, and every policy acts on the same flagged set. The gate threshold is chosen on development data only (section 9).
- Report both views: all eligible questions, and the flagged set with repairs, damage and cost. Cost matching follows section 5.
- This section does not approve the gate, the diagnosis or any repair. Section 6 says when the all-repairs diagnostic may run.

### 15.10 Role of the 70-question pilot

**Proposed default.** The baseline is a development run. It can show:

- that the whole path runs with a real model, including record formats, abstentions and failure handling;
- the answer-format parse rate, the failure rate, tokens, dollars and latency per question;
- a first EM and F1 with a wide interval. At EM 0.30 on 70 independent questions the 95% half-width is about 0.11, and it is wider because questions share documents (section 8);
- whether the format choices in section 15.7 change scores. Each change is a new run.

It cannot show:

- significance, or a difference of a few points between two policies;
- behaviour on unseen documents or publications (three source families also occur in validation or test);
- how often failures occur in the benchmark (the pilot is a train-exclusive draw);
- anything about diagnosis, repair, chart questions or long documents.

### 15.11 Train, validation and test

**Proposed default, consistent with the Agreed no-tuning rule (section 2).**

- Development (train) data, including `ohr_dev_v1`, is where prompts, retrieval settings, thresholds and repairs are developed. Log each change (section 9).
- Validation stays unused until the lead freezes the method (section 9, freeze procedure). It then gets one confirmation run. No threshold or setting is chosen on validation outcomes.
- Test is run once, after the freeze, and never informs a choice.
- The baseline prompt, model, retrieval settings and cap are part of the frozen method.

### 15.12 Limits of a mostly single-page pilot

- 23 of 30 documents are single pages, so choosing the right page is trivial for them. The baseline cannot say how well retrieval finds the right page in a long document.
- With chunks of 180 whitespace words and five hits, a short page often reaches the model whole, so a baseline error there is mostly an answer-step or OCR error. Wrong-page retrieval and the retry repair get little coverage. A longer-document development sample is a separate **Lead decision** (section 9).
- The pilot has no chart questions, 6 questions whose evidence page has empty text, and 22 Han-script questions. Report these groups as their own rows.
- All figures depend on the MinerU version, which is unknown upstream.

### 15.13 Decision list for the lead

1. Choose the answer model. Proposed: option A (`gpt-4o-2024-11-20`). Confirm that the account can call it, since documentation cannot show that.
2. Set the hard spending cap. Suggested: $2.00 for the main run, and $0.50 for the 10-question repeat probe (expected cost about $0.16).
3. Approve the prompt structure and abstention token in section 15.7, after a dry run prints the real prompts for a few questions.
4. Approve the retrieval settings for the first real run. The r4 engineering run gives every question query tokens, but whether the retrieved chunks hold the evidence is not measured.
5. Agree that the first baseline is text-only, so the image budget and page-selection decision wait for the visual repair.
6. Set the maximum execution-failure rate. Suggested: at most 3 of 70 questions still failed after retries.
7. Inspect and label the 20 cases in section 15.14. Decide whether a second person labels them blind.
8. Decide whether to create the versioned annotation form in section 15.14 with an `evidence_impact` column.
9. Say whether GPU calibration stays on hold. It is not needed for this baseline.
10. Say whether to define the longer-document development sample now or after the baseline.
11. Approve or change the draft prompt `faar-answer-draft-v1` after reading the local preview (section 15.15). The draft adds two lines beyond section 15.7: "Reply with the answer only. Do not explain." and a closing reminder. It also treats a `content_filter` finish without a refusal message as ordinary text.
12. Keep or replace the input-limit rule (12,000 against the byte bound).
13. Decide whether provider-side storage for attempt lookup (`store=true` plus metadata at OpenAI, which allows looking up a lost response) may ever be switched on. It stores prompts and replies with the provider, so it is a data-handling choice. Since 2026-09-30 every request sends `store=false`, and this CLI refuses the storage option. Separately, decide whether sending OHR-Bench document text to OpenAI fits the dataset terms, given the 30-day abuse-monitoring copy that `store=false` does not remove.

### 15.14 Human inspection: what you check and label

**Packet check on 2026-09-29.** Read-only, in the main checkout. `results/pilots/ohr_dev_v1/inspection/index.html` and `annotations.csv` are byte-identical to the copies committed at `8479309`.

- `index.html` has 20 cases and 24 `<img>` references. All 24 files exist under `pages/`, they cover exactly the 17 files in `render_record.json`, and every PNG matches its recorded SHA-256. The page has no script and no remote reference.
- `annotations.csv` has 24 rows (one per case and page) and 20 distinct cases. Its question IDs match `inspection_cases.json`. Its columns are `case`, `question_id`, `doc_id`, `page_idx`, `word_corruption`, `lost_or_misleading_structure`, `missing_content`, `apparently_adequate_ocr`, `uncertain_or_mixed`, `uncertainty_note`, `notes`, `annotator`. They match section 10, and no cell in the observation columns holds a value.
- `pages/` is git-ignored. A fresh checkout shows broken images until `scripts/data/build_pilot.py` regenerates them from `data/ohr_bench_raw/pdfs.zip`. Open the packet from the main checkout.
- Each case shows the question first, so the question is visible while you judge the text. The page's clean reference text and the gold answer sit in two collapsed sections. Keep both closed for step 1 of section 10.

**What to inspect.** For each row, mark the defects you see on the page, then judge evidence impact once per case. Write `evidence: <survives|damaged|lost|uncertain>` at the start of `notes` on the first row of the case only.

| Cases | Why selected | Rows | Notes |
| --- | --- | --- | --- |
| 1 to 3 | Empty MinerU text (GNHK handwriting, one notes page) | 3 | Do not infer a cause from absent text alone. Say whether the image holds text |
| 4, 5 | Evidence on two pages of one document | 4 | Read both pages together |
| 6, 7 | Two questions on the same two pages | 4 | The same two images appear twice. Label each case separately |
| 8, 9, 13 | Han-script questions (13 also tests reading order) | 3 | Judge whether characters and order survive |
| 10 | Table (Han) | 1 | Judge lost structure |
| 11, 12 | Formula | 2 | Judge whether the formula survives as text |
| 14 to 20 | Ordinary text cases | 7 | Include `apparently_adequate_ocr` where the text is fine |

Also note wherever 150 DPI is too coarse to read the page (`pilot_readiness.md`). The eventual diagnosis study needs at least two independent labellers (section 10). One person's labels here are development inspection only. Do not fill any field from a model's suggestion.

**Proposed versioned form (not created).** Section 10 step 3 stores evidence impact in free text. A new file `annotations_v2.csv` in a new directory (for example `results/pilots/ohr_dev_v1_inspection_v2/`, with its own version record) could add one case-level row per case with `evidence_impact` (survives, damaged, lost, uncertain), `evidence_impact_note`, `annotator` and `blind_to_reference` (yes or no), and keep the page-level defect rows as they are. The frozen `annotations.csv` and its schema stay unchanged. Creating the file is a **Lead decision**.

### 15.15 Dry-run inspection before any spending

**Procedure (implemented; the run itself is not approved).**

1. Run `python scripts/experiments/run_pilot_live.py dry-run --out .local/work/prompt-preview`. It reads only the runtime manifest and the MinerU text, calls no provider, needs no credential, and writes `prepared_requests.jsonl`, `prompt_preview.md` and `dry_run_summary.json`. Keep the output local. It holds full document text.
2. Open `prompt_preview.md`. It has a summary table, an index of English, Chinese, mixed-language, skipped and longest-evidence questions, and one section per question. Each section gives the question, the evidence chunk IDs and pages, the exact system and user messages, the evidence size, a heuristic token estimate (labelled as such), the input and cost bounds, and the prompt and evidence hashes.
3. Check that no gold answer, reference text or label appears. Check that the evidence belongs to the named document and that instructions inside the evidence are fenced as document text.
4. On 2026-09-29 the dry run gave 70 questions: 63 to send and 7 skipped (6 `no_text_chunks`, 1 `no_text_content`). Without a provider config it uses option A's unapproved values, and `dry_run_summary.json` says so.

**Limits that still apply.** Query tokens do not show that the evidence is relevant. 23 of 30 pilot documents are single pages. The macOS OpenMP limitation is documented, not fixed. The scorer's redistribution question is open. The human inspection of section 15.14 has not been done.

