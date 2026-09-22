# Cluster handoff

The step-by-step supervisor handoff (clone, environment, preflight, bounded
calibration, resume, sizing, sharded preparation) is in
`docs/operations/runbook.md`. This file documents the mechanics behind each
command.

## Environment

Create the pinned CPython 3.12 environment and verify it:

```bash
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
mkdir -p results/environment
.venv-aaai/bin/python -m pip freeze > results/environment/pip-freeze.txt
```

A paper run is invalid if `pip check` fails or the freeze file is missing; see
`docs/experiments/aaai-reproducibility.md`. The first command on the shared server is a
read-only hardware preflight. It does not download the benchmark, load FAAR,
load a model, or use an API key. On a login node, skip CUDA:

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Send `cluster/preflight.json` back with the GPU model and available VRAM, CPU
RAM or scheduler limit, free disk, CUDA version, and scheduler name. Do not
send `.env` files, API keys, or model caches. The preflight reads only an
explicit whitelist of environment variables and redacts secret-shaped strings
before printing or writing the report; the `--out` write is atomic.

The report includes:

- parsed Slurm metadata (job id/name, partition, QOS, CPUs, memory, walltime,
  and requested/on-node GPU counts) and PBS metadata (job id/name, queue, CPU
  count, memory, GPU count, walltime);
- cgroup v1/v2 current, peak, and limit memory;
- `ulimit` soft/hard limits;
- scratch (`TMPDIR`, `TMP`, `SCRATCH`, ...) and cache (`HF_HOME`,
  `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`) paths with
  existence, writability, and free-space checks.

## Dry run

Before any billed or GPU time, print the exact ordered B0-B4 commands with a
one-example dry run:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench --split test \
  --max-examples 1 --dry-run
```

`--dry-run` only prints each stage's launcher command; it writes nothing, needs
no API key, and does not require the B2 gate lock.

## Ordered baseline runner

`cluster/run_baselines.py` runs B0, B1, B2, B3, B4 in order, launching each
stage as a fresh subprocess through `cluster/launcher.py` and validating every
written result before the next stage:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench --split test \
  --resume
```

Stages and boundaries:

- B0 `--gate off --recovery off`: text-only RAG; runs without an OpenAI key.
- B1 `--recovery always_vlm`: first `OPENAI_API_KEY` boundary; the launcher
  refuses to start without the key required by `VLM_BACKEND`.
- B2 `--gate on --recovery random_type`: requires
  `config/gate_threshold.json` locked from validation-only results
  (`source_split` `val`, precision >= 0.75, recall >= 0.70); run
  `scripts/experiments/tune_gate.py` on validation results first. The gate is never tuned on test
  data.
- B3 `--mode colpali`: visual retrieval baseline.
- B4 `--mode visrag`: visual retrieval baseline.

The runner prints `[run-baselines] launch <stage>` before each stage and
`[run-baselines] done <stage>` after validation; `--dry-run` shows the same
commands without running them.

Resume behavior: without `--resume` the runner refuses to start if any
`results/baselines/<dataset>/<split>/{b0,b1,b2,b3,b4}.json` exists. With
`--resume`, completed stages are re-validated and skipped, and each missing
stage runs. `--resume` is forwarded to `scripts/experiments/run.py` for B0..B4, so all stages
reuse completed per-example checkpoints.

Output validation: each stage result must carry that stage's profile and
label, a summary with `EM`, `F1`, `vlm_rate`, `harm_rate`, API requests,
tokens, `cost_usd`, and `runtime_sec`, full `run_spec` provenance (dataset,
split, seed,
`max_examples`, embedding, reranker, OCR, model revisions, unsharded), rows
that exactly match B0's example ids, and summary values recomputed from the
rows. Any mismatch stops the run. Because each stage runs in a fresh
subprocess, per-stage memory is released before the next stage starts.

## Validation B0 and gate tuning

The gate threshold may be tuned only on the validation split. Before the
lock exists, the ordered runner can produce the validation B0 in isolation —
this is the only ordered-runner mode that does not require
`config/gate_threshold.json`:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench \
  --split val \
  --stop-after b0 \
  --resume
```

The output is `results/baselines/ohrbench/val/b0.json`. Do not use the
default `test` split for gate tuning; scheduler jobs that prepare validation
assets must request the `val` split explicitly (`FAAR_SPLIT=val`).

Once the validation B0 exists, tune and lock the gate threshold:

```bash
.venv-aaai/bin/python scripts/experiments/tune_gate.py \
  --baseline results/baselines/ohrbench/val/b0.json \
  --out results/gate_threshold_search.json \
  --lock config/gate_threshold.json
```

This writes `config/gate_threshold.json` (locked at precision >= 0.75 and
recall >= 0.70). It refuses non-validation sources and refuses to lock a
search that did not pass.

After the lock exists, resume the complete B0-B4 sequence with the exact
runner command on validation first:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench --split val \
  --resume
```

This re-validates and skips the existing validation B0, then runs validation
B1, B2, B3, and B4. Freeze the accepted gate configuration and validation
artifacts before starting test. Only then run the test sequence:

```bash
.venv-aaai/bin/python cluster/run_baselines.py \
  --project-root "$PWD" \
  --dataset ohrbench --split test \
  --resume
```

B2 cannot be skipped: any run that reaches or prepares for B2 requires a
compatible gate lock, and the runner validates the lock's model provenance
against the run's embedding/reranker configuration before launching B0 or
paid B1. `--dry-run` requires no lock and performs no execution; it only
prints the commands.

## Launcher

`cluster/launcher.py` is the generic entry point for one bounded repository
job, spawned per stage by `run_baselines.py`. It is pure-stdlib, never installs
packages, never copies datasets, and never overrides `CUDA_VISIBLE_DEVICES`.

```bash
.venv-aaai/bin/python cluster/launcher.py \
  --project-root "$PWD" \
  --gate off --recovery off --split test \
  --out results/b0.json
```

It runs preflight, then:

1. Requires exactly one logical GPU unless `--cpu-only` is passed; a
   multi-GPU or GPU-less allocation is rejected with exit code 2.
2. Requires an explicit positive `FAAR_GPU_BUDGET_GB` chosen after the
   preflight shows the GPU model and free VRAM. The budget is never inferred
   from VRAM headroom or model size. For backward compatibility an explicitly
   supplied `FAAR_MAX_GPU_MEMORY_FRACTION` in `(0, 1]` is accepted instead and
   translated to an equivalent budget; with neither variable set the launch
   fails closed. Setting both variables is rejected as ambiguous.
3. Rejects a budget above the GPU's total VRAM and any non-finite or
   non-positive value; the requested budget is never silently clamped.
4. Requires the initial free VRAM to cover the process budget plus the
   co-tenant reserve (`FAAR_MIN_GPU_FREE_GB`, a 20% reserve by default).
5. Derives guards only when absent:
   - `FAAR_MAX_RSS_GB` (90% of the scheduler/cgroup CPU-RAM budget, else 50%
     of physical memory),
   - `FAAR_MIN_GPU_FREE_GB` (a 20% co-tenant reserve),
   - `FAAR_MAX_GPU_MEMORY_FRACTION` (`FAAR_GPU_BUDGET_GB` / total VRAM),
   - `OMP_NUM_THREADS` / `MKL_NUM_THREADS` (half the allocated CPUs).
6. Validates the Hugging Face cache and, only for a run that can invoke a VLM,
   a non-empty key required by `VLM_BACKEND` (`OPENAI_API_KEY` for OpenAI or
   `ANTHROPIC_API_KEY` for Claude). B0 does not require either key. Key values
   are never printed.
7. Spawns the selected repository-local Python entry point with a plain argv
   list (no shell interpolation), rejects entry points outside the project,
   and forwards `SIGTERM`/`SIGINT` to the child, propagating its exit code.

Every launch writes a redacted, job-namespaced report under
`results/environment/`; override this with `--preflight-out`.
For deliberate reuse of an already-recorded preflight, set
`FAAR_PREFLIGHT_JSON` to the report path; the launcher will not re-probe the
host. `FAAR_PYTHON` overrides the interpreter used for the selected entry
point.

## Shared-server calibration before data transfer

Do not transfer or expand a 200 GB dataset first. Use the existing OHR archive
to run one complete 108-page document through PDF extraction, Docling, page
rendering, and pinned GOT-OCR. This is calibration evidence, not a paper
result. Run the preflight in the actual scheduler allocation, choose the
process budget from its reported GPU and free VRAM, and place outputs on the
lab scratch filesystem:

```bash
.venv-aaai/bin/python cluster/preflight.py --out cluster/preflight.json

export FAAR_GPU_BUDGET_GB=<approved-process-vram-gib>
export HF_HOME="${SCRATCH:-$PWD/results/calibration}/huggingface"
mkdir -p "$HF_HOME"

.venv-aaai/bin/python cluster/launcher.py \
  --project-root "$PWD" \
  --entrypoint scripts/data/prepare_benchmark_assets.py \
  --preflight-out results/environment/preflight_calibration.json \
  --dataset ohrbench \
  --out-root "${SCRATCH:-$PWD/results/calibration}/faar-ohr-108" \
  --smoke-doc manual/User_Manual_1500S_Classic_EN \
  --require-calibration-108 \
  --execute
```

The launcher requires one logical GPU, keeps the explicit per-process VRAM
cap, reserves 20% of visible VRAM for co-tenants by default, derives an RSS
limit from the scheduler/cgroup allocation, and caps CPU library threads.
Asset preparation checks those limits before and after PDF extraction,
Docling, rendering, and every GOT-OCR page. Exceeding a limit aborts the
process instead of continuing under memory pressure. Partial files are
removed, and outputs from an earlier completed checkpoint remain resumable.

Return these two files to the project owner after the run:

- `results/environment/preflight_calibration.json`
- `${SCRATCH:-$PWD/results/calibration}/faar-ohr-108/prepare_checkpoint.json`

The checkpoint records measured stage runtime, peak RSS, per-page OCR time,
and PNG/OCR storage. Use those measurements to choose scheduler RAM,
walltime, and scratch capacity and to extrapolate the validation footprint.
Do not copy the full corpus until that estimate fits the lab quota with
headroom.

## Scheduler templates

Editable one-GPU Slurm templates under `cluster/templates/`:

- `slurm_preflight.sbatch`: allocated-GPU preflight (`preflight.py --check`)
  with the job exit code mirroring the preflight exit code (0 ready, 1
  blocking failure, 2 warnings). The login-node-safe variant runs directly:
  `cluster/preflight.py --check --no-cuda --project-root "$PWD"`.
- `slurm_calibration_108.sbatch`: the bounded 108-page calibration through the
  launcher with `--signal=TERM@120`, scratch-scoped `HF_HOME`, and an explicit
  `FAAR_GPU_BUDGET_GB` placeholder. Resubmission resumes completed stages.
- `slurm_prepare_val_shard.sbatch`: Slurm array job for the sharded validation
  asset preparation (`cluster/prepare_assets.py --split val --resume`), one
  shard per array task. Shards resume independently; merge/register follows
  after all shards complete.
- `slurm_pilot_baseline.sbatch`: 50-100 example baseline pilot
  (`run_baselines.py --max-examples 100 --resume`); engineering check only,
  never to be launched before the calibration report is approved.

`slurm_one_gpu.sbatch` runs the full ordered baseline runner and
`pbs_one_gpu.pbs` is its PBS equivalent:

- `slurm_one_gpu.sbatch`: generic full B0-B4 runner, not the 108-page
  calibration. `--gpus=1`, bounded `--mem` and `--time`, `--signal=TERM@120`,
  and `<EDIT_PARTITION>` / `<EDIT_QOS>` / `<EDIT_ACCOUNT>` placeholders that
  refuse to start Python until they are replaced.
- `pbs_one_gpu.pbs`: `ngpus=1` and bounded `ncpus`/`mem`/
  `walltime`; its optional TORQUE `softwalltime` directive is disabled until
  the target cluster confirms support.

All templates allocate exactly one GPU, never set `CUDA_VISIBLE_DEVICES`
(Slurm and PBS manage the allocation), keep resource boundaries and scheduler
directives as placeholders to edit once the lab allocation is known, and
require `FAAR_GPU_BUDGET_GB` set from the preflight report (the launcher fails
closed without it). Credentials are read from the submit environment or a
`.env` at the repository root; never write keys into a template file.

## Staged execution

1. Run the preflight and the bounded 108-page complete-document calibration
   above. This produces throughput and memory evidence, not paper results.
2. Run an end-to-end pilot on complete documents covering roughly 50--100
   questions. This is still an engineering check.
3. Prepare the full OHR validation assets: 1,274 questions, 549 documents, and
   7,037 pages. Use this split for real B0 and gate tuning.
4. Run B0, B1, B2, B3, and B4 sequentially, saving each result before the next.
5. Run the full OHR test split: 1,276 questions, 567 documents, and 6,849
   pages.
6. Run ArXivQA only after all 500 page mappings are human-confirmed. MP-DocVQA
   is a separate job after its official validation data is acquired.

The dataset is staged once and reused. Do not make a separate copy of the
page images for each baseline. The repository's locked split and model
revisions remain the source of truth.

## Multi-GPU sharding (text baselines B0-B2)

The recommended scaling strategy is one process per GPU with deterministic
shards; DDP/DataParallel add no value for this embarrassingly parallel
retrieval workload. Each shard writes per-example checkpoints and its own
shard result file, so shards can be resumed independently and merged later.

```bash
# one job per GPU, e.g. 8 GPUs:
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i .venv-aaai/bin/python cluster/launcher.py \
    --project-root "$PWD" \
    --preflight-out "results/environment/preflight_b0_shard${i}.json" \
    --gate off --recovery off --dataset ohrbench --split test \
    --shard-index $i --num-shards 8 \
    --out results/b0.json > results/b0_shard$i.log 2>&1 &
done
wait

# merge the shards into the single result the analysis pipeline expects:
.venv-aaai/bin/python cluster/merge_shards.py \
  --out results/b0.json \
  --manifest data/benchmark_assets/ohrbench/test.json \
  results/b0_shard*of8.json
```

Rules:

- Every shard must be launched with the same `--seed`, `--num-shards`,
  dataset, split, and model flags; `merge_shards.py` rejects shards whose
  run_specs differ or whose example ids overlap. Publication requires every
  shard index exactly once.
- B0 merge requires the exact locked asset manifest. Its SHA-256 must match
  every shard's `run_spec`, and the merged example ids must equal the manifest
  selection after `--max-examples`; a complete set of partial workers is not
  publishable.
- The merged file resets `shard_index`/`num_shards` to null and recomputes the
  summary (EM, F1, vlm_rate, API totals) and `harm_rate` from the merged rows,
  so it passes the normal stage validation on later `--resume` runs.
- B1/B2 shards need the matching B0 shard as `--baseline` while running
  (run_spec comparison includes the shard index). Merge B0 first; when merging
  B1 or B2, pass the merged B0 as `--baseline` so the merged IDs and harm rate
  are validated before publication.
- B3/B4 (ColPali/VisRAG) are single-GPU per job: `scripts/experiments/run.py` rejects shard flags
  for visual modes because corpus embeddings are built once per process. The
  corpus-embedding cache (below) keeps a resumed visual run from re-encoding
  the corpus.

## Shard manifest merge

After all preparation shards finish, validate the shard set and merge it into
one operator-facing record before registering the benchmark assets:

```bash
.venv-aaai/bin/python cluster/merge_prep_shards.py \
  --project-root "$PWD" --split val \
  --out results/validation/merged_assets.json \
  results/validation/faar-ohr-val/shard_manifest_shard*.json
```

The merge verifies the split checksum lock, requires every shard index in
`[0, num_shards)` exactly once, rejects overlapping or uncompleted documents,
and requires the combined document set to match the locked split exactly. The
merged record is published atomically. `prepare_assets.py` and
`scripts/data/prepare_benchmark_assets.py` read `FAAR_PDF_ROOT`, `FAAR_PDF_ZIP`, and
`FAAR_DOCUMENT_INVENTORY` from the environment as defaults for their CLI
flags, and preflight validates those configured paths before any job.

## Persistent corpus-embedding cache

Both the text retriever and the visual retrievers persist corpus embeddings
under `<cache_dir>/` so that resumes and shards skip the encode step:

- text: `<cache_dir>/text_embeddings/<key>.npz` + `<key>.meta.json`
- visual: `<cache_dir>/visual_embeddings/<key>.npz` + `<key>.meta.json`

The key covers the model, revision, dtype, and the exact corpus identity
(chunk ids + texts, or the image path list). Any mismatch recomputes; a hit
reuses bitwise-identical embeddings. `cache_dir` defaults to
`<project_root>/cache` and can be moved with `FAAR_CACHE_DIR`. Concurrent
writers produce identical content, so atomic last-writer-wins is safe.

## Runtime configuration

All of the following are environment variables with validated values; the
launcher derives the memory/GPU ones when unset:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FAAR_GPU_BUDGET_GB` | required for GPU runs | explicit per-process VRAM budget in GiB |
| `FAAR_MAX_GPU_MEMORY_FRACTION` | derived from budget | per-process CUDA memory fraction in `(0,1]` |
| `FAAR_MIN_GPU_FREE_GB` | 20% of GPU VRAM | co-tenant free-VRAM reserve |
| `FAAR_MAX_RSS_GB` | 90% of scheduler/cgroup RAM | in-process RSS fail-fast guard |
| `FAAR_EMBED_BATCH_SIZE` | 2 | corpus/query embedding batch size |
| `FAAR_VISUAL_BATCH_SIZE` | 1 | ColPali/VisRAG corpus-encode batch size |
| `FAAR_VISUAL_SCORE_BATCH_SIZE` | 8 | ColPali score batch size |
| `FAAR_VLM_MAX_RETRIES` | 3 | total attempts per VLM call; retries only on 429/5xx/timeouts |
| `FAAR_VLM_RETRY_BACKOFF_SECONDS` | 2.0 | exponential backoff base with jitter |
| `FAAR_VLM_TIMEOUT_SECONDS` | 60 | per-request VLM timeout |
| `FAAR_CACHE_DIR` | `<project_root>/cache` | corpus-embedding cache root |
| `FAAR_PDF_ROOT` | `data/ohr_bench_raw/pdfs.zip` | filesystem PDF source root for asset preparation |
| `FAAR_PDF_ZIP` | `data/ohr_bench_raw/pdfs.zip` | PDF archive path for asset preparation |
| `FAAR_DOCUMENT_INVENTORY` | `OHR-Bench/data/retrieval_base/gt` | per-document page inventory directory |
| `FAAR_LOG_INTERVAL_SECONDS` | 60 | progress-line interval |
| `FAAR_LOG_INTERVAL_ITEMS` | 25 | progress-line item interval |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | half the allocated CPUs | thread caps |

CUDA out-of-memory during corpus encoding or scoring halves the batch size
automatically (floor of 1) and retries the same slice after releasing the
CUDA cache; the event is printed and the reduced size is kept for the rest of
the process. A batch of 1 that still OOMs fails the job rather than hiding the
error.

A job interrupted by SIGTERM/SIGINT stops at the next example boundary,
leaves all completed per-example checkpoints intact, and exits 130/143 without
publishing a partial result file. Per-example failures (bad data, exhausted
VLM retries) are recorded in a failed checkpoint with the error message and
the run continues; the run then fails at the end with the exact list of failed
example ids, so nothing is silently skipped and a single bad example cannot
waste a multi-hour job.
