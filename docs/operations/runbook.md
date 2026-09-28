# FAAR shared-cluster runbook

This is the copy-paste procedure. The short supervisor page is
[SUPERVISOR_HANDOFF.md](../../SUPERVISOR_HANDOFF.md). Submit only the allocated-GPU
preflight and the bounded 108-page calibration until that calibration report is
approved. Do not submit validation, pilot, or B0-B4 jobs before then.

---

## 1. Clone and checkout the exact branch

```bash
git clone https://github.com/haseebraza715/FailSafeRAG.git faar
cd faar
git checkout main
git rev-parse HEAD          # record this SHA; it goes into the calibration report
```

## 2. Create the environment

```bash
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
mkdir -p results/environment
.venv-aaai/bin/python -m pip freeze > results/environment/pip-freeze.txt
```

A paper run is invalid if `pip check` fails or the freeze file is missing.

## 3. Configure paths

Copy `.env.example` to `.env` and fill only the names you need. The command
line below shows the same variables. **Never commit `.env` and never paste
keys into scheduler templates.**

```bash
export FAAR_PROJECT_ROOT="$PWD"
export FAAR_SCRATCH="${FAAR_SCRATCH:-$PWD/results/calibration}"      # fast lab scratch
export FAAR_OUT_ROOT="$FAAR_SCRATCH"                                  # asset output root
export HF_HOME="$FAAR_SCRATCH/huggingface"                            # model cache on scratch
mkdir -p "$HF_HOME"
# Optional overrides if the OHR archive is not at the default location:
# export FAAR_PDF_ROOT=/path/to/pdf/dir
# export FAAR_PDF_ZIP=/path/to/pdfs.zip
# export FAAR_DOCUMENT_INVENTORY=/path/to/gt
```

Required local inputs (see section 12): `config/datasets/ohr_split.json`, `config/split_checksums.json`,
`OHR-Bench/data/qas_v2.json`, `OHR-Bench/data/retrieval_base/gt/`, and the PDF
archive at `data/ohr_bench_raw/pdfs.zip` (or `FAAR_PDF_ROOT`). The
preparation CLIs and preflight honor `FAAR_PDF_ROOT`, `FAAR_PDF_ZIP`, and
`FAAR_DOCUMENT_INVENTORY` from the environment, so the same `.env` values drive
both the checks and the jobs; preflight fails if a configured path does not
exist.

## 4. Login-node-safe preflight (no CUDA)

```bash
.venv-aaai/bin/python cluster/preflight.py --check --no-cuda --project-root "$PWD"
```

Exit codes: **0** ready, **1** blocking failure (fix before anything else),
**2** warnings only. Run this from a login node; it never touches CUDA, never
writes anything, and never prints secrets. This is the command to run first.

## 5. Allocated-GPU preflight

On a short interactive allocation (or via `sbatch cluster/templates/slurm_preflight.sbatch`):

```bash
srun --gpus=1 --cpus-per-task=8 --mem=32G --time=00:30:00 --pty bash -c '
  .venv-aaai/bin/python cluster/preflight.py --check --project-root "$PWD" \
    --out "cluster/preflight_${SLURM_JOB_ID}.json"'
```

Send `cluster/preflight_<jobid>.json` back with the report. From the GPU model
and free VRAM, choose a positive `FAAR_GPU_BUDGET_GB` (GiB) for the process;
the launcher fails closed without it. The cluster's own GPU count/type, CPU and
RAM bounds, and partition/account go into the templates under
`cluster/templates/` (edit the `<EDIT: ...>` placeholders).

## 6. Submit ONLY the bounded 108-page calibration first

Edit `cluster/templates/slurm_calibration_108.sbatch`: partition, qos, account,
and `FAAR_GPU_BUDGET_GB`. Then:

```bash
sbatch cluster/templates/slurm_calibration_108.sbatch
```

This runs one complete 108-page document
(`manual/User_Manual_1500S_Classic_EN`, pages 0–107) through PDF extraction, Docling, page
rendering, and pinned GOT-OCR. The job refuses to start if the document or
inventory is not that exact 108-page set. **Calibration evidence only — never a paper
result.** Submit nothing else until it is approved. Do not use
`cluster/templates/slurm_one_gpu.sbatch` for this step.

## 7. Monitor logs and disk

```bash
squeue -u "$USER"
tail -f logs/calibration_*.out
du -sh results/calibration/*            # assets grow as pages complete
df -h "$FAAR_SCRATCH"
```

Progress lines appear as `[faar] stage=... progress=N/M ...`. A job interrupted
by the 120 s SIGTERM warning exits 143 after saving its checkpoint; completed
stages are never re-done. The interrupted attempt is closed with its measured
elapsed time. A SIGKILL / hard timeout cannot run that cleanup: the next resume
marks the still-running attempt as an unclean shutdown and sets
`timing_complete=false`. Do not treat that incomplete sum as the exact cluster
wall time.

## 8. Resume a preempted job

Re-submit the same script; it resumes from the checkpoint:

```bash
sbatch cluster/templates/slurm_calibration_108.sbatch    # calibration
sbatch cluster/templates/slurm_prepare_val_shard.sbatch  # sharded prep (per shard)
```

Re-running a finished job is idempotent: completed documents are re-validated
and skipped, and the split checksum is verified before anything starts.

If calibration was hard-killed (SIGKILL / node failure) the checkpoint keeps a
running attempt. Resume records it as `unclean`. Hard-kill timing correction
happens only against the calibration checkpoint. Do not guess elapsed time from
the start timestamp. Supply the scheduler-reported wall time for that attempt:

```bash
.venv-aaai/bin/python scripts/data/prepare_benchmark_assets.py \
  --project-root "$PWD" \
  --dataset ohrbench \
  --out-root results/calibration/faar-ohr-108 \
  --smoke-doc manual/User_Manual_1500S_Classic_EN \
  --require-calibration-108 \
  --scheduler-elapsed-sec <seconds from sacct/qstat>
```

Every unclean attempt needs its own scheduler-reported duration. If several
attempts are unresolved, repeat that checkpoint command once per attempt, or
rerun calibration. Then rebuild the summary and project. The projection command
does not accept scheduler timing.

Use a new `--checkpoint` path if the smoke document, page set, source PDF, or
GOT-OCR/Docling lock changed. The runner refuses to mix those identities.

## 9. Build the calibration report and send it back

```bash
.venv-aaai/bin/python cluster/calibration_report.py build \
  --checkpoint results/calibration/faar-ohr-108/prepare_checkpoint.json \
  --preflight results/environment/preflight_calibration.json \
  --project-root "$PWD" \
  --out results/calibration/calibration_summary.json

.venv-aaai/bin/python cluster/calibration_report.py project \
  --summary results/calibration/calibration_summary.json \
  --out results/calibration/preparation_projection.json
```

The summary contains the commit SHA, GPU model, peak RSS, per-stage and total
runtimes, per-page OCR throughput distributions, storage, cache growth, resume
flag, timing_complete, and split/manifest hashes. `total_wall_sec` is present
only when every attempt has measured or scheduler-supplied elapsed time. If
`timing_complete=false`, correct the checkpoint first as in section 8, rebuild
this summary, then run projection. Projection refuses incomplete timing and
does not take a scheduler duration of its own. The projection sizes validation
(549 docs / 7,037 pages) and test (567 docs / 6,849 pages) separately for
Docling (document-level), rendering, and OCR (page-level), with a configurable
headroom, suggested shard counts, and explicit assumptions.

**Return to the research team:** `calibration_summary.json`,
`preparation_projection.json`, `cluster/preflight_<jobid>.json`, and the
calibration `prepare_checkpoint.json`. Wait for approval before proceeding.

## 10. After approval: sharded validation preparation

Size the shard count from the projection, then edit
`cluster/templates/slurm_prepare_val_shard.sbatch` (`--array` and `SHARDS`
must match) and submit:

```bash
sbatch cluster/templates/slurm_prepare_val_shard.sbatch
```

Each shard writes `checkpoint_shardNofM.json` and
`shard_manifest_shardNofM.json` under the out root. When all shards are
finished, validate the shard set (complete, non-overlapping, every index in
range, no uncompleted documents, exact match with the locked split) and merge
it into one record:

```bash
.venv-aaai/bin/python cluster/merge_prep_shards.py \
  --project-root "$PWD" \
  --split val \
  --out "$FAAR_SCRATCH/faar-ohr-val/merged_assets.json" \
  "$FAAR_SCRATCH/faar-ohr-val"/shard_manifest_shard*.json
```

Then build the locked benchmark manifest and validate coverage:

```bash
.venv-aaai/bin/python scripts/data/register_benchmark_assets.py \
  --dataset ohrbench --split val \
  --ocr-dir "$FAAR_SCRATCH/faar-ohr-val/ocr" \
  --image-dir "$FAAR_SCRATCH/faar-ohr-val/images" \
  --out data/benchmark_assets/ohrbench/val.json
```

The manifest build refuses to publish until every inventory page has OCR text
and an image; the fixed split is enforced. The full validation run (B0-B4,
gate tuning) happens only after the manifest is registered and reviewed.

A small 50-100 question pilot (`cluster/templates/slurm_pilot_baseline.sbatch`)
is the recommended engineering check before full baselines.

## 11. Common failure recovery

| Symptom | Action |
| --- | --- |
| preflight exit 1 | Read the failing check; fix paths/limits/checksums, re-run |
| `Immutable split integrity check failed` | The split/QA files changed or the clone is wrong; restore them, never edit |
| `FAAR_GPU_BUDGET_GB` missing | Choose it from the preflight GPU report |
| Job killed at walltime (143) | Re-submit; checkpoints resume automatically |
| Hard kill / SIGKILL / node death | Resume marks the running attempt unclean and `timing_complete=false`. Apply `scripts/data/prepare_benchmark_assets.py --scheduler-elapsed-sec` from sacct/qstat to the checkpoint, rebuild the summary, then project. Repeat once per unresolved unclean attempt, or re-run calibration. Projection does not accept scheduler timing. |
| Checkpoint identity mismatch | The smoke document, pages, source PDF, or model lock changed. Use a new `--checkpoint` path. |
| `CUDA out of memory` / `MemoryError` | Job aborts safely with checkpoint; reduce budget or re-check the preflight |
| Recoverable document errors | Listed in the shard manifest; the job exits non-zero; re-submit with `--resume` to retry failed documents |
| Missing HF model | Check `hf_model_access` in preflight; export `HF_TOKEN` for gated models |
| Corrupt checkpoint | The runner fails closed and tells you; restore from job logs, do not hand-edit it |

## 12. Required inputs and credentials by stage

| Stage | Hugging Face | OpenAI/Anthropic key |
| --- | --- | --- |
| Preflight | checks model access (no downloads, honors `HF_TOKEN`) | none |
| 108-page calibration / asset preparation | GOT-OCR-2.0 (locked revision), Docling models snapshot (`docling-project/docling-models` at immutable commit `2bdc831f…`, pinned in `config/model_revisions.json`) | none |
| B0 baseline | NV-Embed-v2, bge-reranker-v2-m3 | none |
| B1/B2/B4 | same + VisRAG/ColPali | `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` if `VLM_BACKEND=claude*`) |
| Gate tuning | none | none |

Keys are read from the environment or `$PWD/.env`; never printed, never in
templates.

## 13. Handoff checklist

- [ ] Cloned on branch `main`; commit SHA recorded
- [ ] Environment created; `pip check` clean; `pip-freeze.txt` saved
- [ ] `.env` configured (names only; `.env` never committed)
- [ ] Login-node preflight `--no-cuda` passes (exit 0/2)
- [ ] Allocated-GPU preflight report sent back
- [ ] `FAAR_GPU_BUDGET_GB` chosen; templates edited for partition/account
- [ ] Calibration job submitted; logs and disk monitored
- [ ] Calibration summary + projection built and returned
- [ ] **Approval received before any further job**
- [ ] Sharded validation prep sized, submitted, merged, manifest registered
- [ ] Pilot (50-100 questions) approved before full baselines

## 14. What to send the supervisor

1. `cluster/preflight_<jobid>.json` (allocated-GPU preflight)
2. `results/calibration/calibration_summary.json`
3. `results/calibration/preparation_projection.json`
4. `results/calibration/faar-ohr-108/prepare_checkpoint.json`

Never send `.env`, keys, model caches, or the raw dataset.

## 15. Dataset warning

**Inventory the source archive and measure generated storage before transferring
or duplicating the corpus.** The compressed archive is 1,516,951,813 bytes
(`config/ohr_pdf_source_lock.json`). The earlier ~200 GB figure is unmeasured and must
not be used as a dataset size or hardware requirement. The calibration needs only the OHR
archive, the fixed split, the QA source, and the per-document page inventory
(section 3). Validation assets are produced on the cluster scratch and only
the final manifest plus measurements are returned. Confirm quota and scratch
policy with the cluster administrator before staging the full corpus.
