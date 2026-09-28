# AAAI Reproducibility Contract

This document defines the software and model identity checks required before a
FAAR result can be reported. It complements `config/environment/constraints-aaai.txt`; it does not
replace the experiment ordering or acceptance criteria in
`docs/research/aaai-plan.md`.

## Python environment

Use CPython 3.12 in a fresh environment. Install the project plus the tools that
are intentionally not declared in `pyproject.toml` through the shared
constraints file:

```bash
python3.12 -m venv .venv-aaai
.venv-aaai/bin/python -m pip install --upgrade pip
.venv-aaai/bin/python -m pip install -c config/environment/constraints-aaai.txt -e '.[aaai]'
.venv-aaai/bin/python -m pip check
mkdir -p results/environment
.venv-aaai/bin/python -m pip freeze > results/environment/pip-freeze.txt
```

All entries in `config/environment/constraints-aaai.txt` are exact pins. The set was dependency
resolved together for CPython 3.12 on macOS arm64. Platform-specific wheels must
still be resolved and checked on the machine used for reported runs. In
particular, do not replace the pinned PyTorch build with a CUDA build without
recording its exact version, wheel source, CUDA version, GPU model, and driver.

Label Studio 1.17.0 is intentional. Label Studio 1.13.1 fixes
`jsonschema==3.2.0`, which conflicts with Docling 1.20.0's
`jsonschema>=4.16.0` requirement. Version 1.17.0 remains within the plan's
required Label Studio 1.x line and resolves with Docling 1.20.0. Typer 0.12.5 is
also intentional because Docling 1.20.0 requires Typer `>=0.12.5,<0.13.0`.

The constraints file pins direct and compatibility-critical packages. The
captured `pip-freeze.txt` is the immutable record of every transitive package in
the actual run environment. A paper run is invalid if `pip check` fails or the
freeze file is missing.

## Model identities

Repository IDs and immutable revisions are locked in
`config/model_revisions.json`. They were resolved from the model host with
`HfApi.model_info(repo_id=..., revision="main").sha`; the runtime passes those
40-character commits to every model loader and records them in run provenance.
Never report a run loaded from a branch name such as `main`.

```bash
export EMBED_MODEL_REPO=nvidia/NV-Embed-v2
export EMBED_MODEL_REVISION=3fa59658547db50a1e8e3346cf057fd0c77ed6ef

export RERANKER_MODEL_REPO=BAAI/bge-reranker-v2-m3
export RERANKER_MODEL_REVISION=953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e

export GOT_OCR_MODEL_REPO=stepfun-ai/GOT-OCR-2.0-hf
export GOT_OCR_MODEL_REVISION=d3017ef2c2c1395888c8d635c5e0508bcb0ac78d

export COLPALI_MODEL_REPO=vidore/colpali-v1.2-hf
export COLPALI_MODEL_REVISION=56b5c732913abf3501ba208562df7b8f4edd5861

export VISRAG_MODEL_REPO=openbmb/VisRAG-Ret
export VISRAG_MODEL_REVISION=95ef596df871b606167cb7e4b7215caf1bfdf761

export BYT5_MODEL_REPO=google/byt5-small
export BYT5_MODEL_REVISION=68377bdc18a2ffec8a0533fef03b1c513a4dd49d
```

Resolve a Hugging Face revision at the time the experimental environment is
frozen, while authenticated for gated repositories:

```python
from huggingface_hub import HfApi

repo_id = "nvidia/NV-Embed-v2"
sha = HfApi().model_info(repo_id=repo_id, revision="main").sha
assert sha is not None and len(sha) == 40
print(sha)
```

Repeat this resolution before the final environment freeze to verify that the
locked commits remain accessible. Any intentional revision update must change
`config/model_revisions.json`, the paper appendix, and the run provenance
together. `trust_remote_code=True` does not pin remote code by itself; it is
acceptable only together with the immutable revision.

Docling may download additional layout, OCR, and table-model artifacts. Before
reported runs, enumerate those repositories from the populated Hugging Face
cache, resolve each to an immutable commit SHA, and add them to the run manifest.
The `docling==1.20.0` package pin alone does not pin those weights.

## Hosted VLM identity

The user selected the plan's OpenAI Alternate A. Use a dated GPT-4o snapshot
for every VLM-dependent baseline, FAAR run, and ablation:

```bash
export VLM_BACKEND=openai
export OPENAI_MODEL=gpt-4o-2024-11-20
```

Store the exact request model, API response model identifier, and run timestamp
in every result's provenance. Do not replace the pinned snapshot with a mutable
model alias during reported runs.

## Required run manifest

Each reported result must retain, at minimum:

- Git commit and branch.
- Python version, OS, CPU, GPU, accelerator backend, and driver/CUDA versions.
- `pip-freeze.txt` and successful `pip check` output.
- Every model repository ID and resolved 40-character commit SHA.
- The hosted VLM request model, response model identifier, and UTC timestamp.
- Dataset name, dataset revision/checksum, split name, and `config/datasets/ohr_split.json` checksum.
- Gate threshold and evidence that it was selected on validation only.
- Random seeds, command line, environment-variable names, and output JSON path.
- API call count, token usage, cost, runtime, and W&B run ID.

Secrets are never part of the manifest. Record only whether the required
credential was present; never print or persist its value.

## Pre-run block

A reported run must stop before inference when any of these checks fails:

1. A revision still contains `REPLACE_WITH` or is not a 40-character commit SHA.
2. A loader omits its configured `revision` argument.
3. `pip check` fails or the environment freeze cannot be written.
4. A paid backend lacks its required API key.
5. A dataset or split checksum differs from the locked manifest.

These checks prevent a successful command from producing scientifically
unattributable numbers.

The AAAI runner enforces these model revision checks through
`AppSettings.validate_model_revisions()` and records repository/revision pairs
under `run_spec.model_provenance`. The OpenAI backend similarly rejects any
model other than the dated `gpt-4o-2024-11-20` snapshot.
