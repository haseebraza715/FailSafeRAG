# Workspace migration, 2026-09-28

Record of the structure change on branch `chore/research-workspace-structure`.
Starting commit `22be5e29be5e75191c7979b9414dc9f041b72987` (`hardening/review-pass`).
The existing audit and pilot work was committed first, unchanged, as
`e38d1fa` and `78f15e7`.

Paths written inside historical documents, result payloads and manifests are
evidence and were not rewritten. Use the relocation map below to resolve them.

## Relocation map (repository)

Apply the longest matching prefix. Everything not listed stayed where it was.

| Old path | New path | Reason |
| --- | --- | --- |
| `docs/experiments/` | `docs/research/` | Research protocol and study brief; frees "experiments" for the run registry |
| `docs/phases/` | `docs/history/phases/` | Prototype phase records |
| `docs/repo_handbook/` | `docs/history/repo_handbook/` | Superseded handbook |
| `docs/repo_architecture/` | `docs/history/repo_architecture/` | Superseded Diataxis layout |
| `docs/archives/` | `docs/history/archives/` | Indexes of committed prototype logs and artifacts |
| `docs/reports/phase1_report.md` … `phase4_report.md` | `docs/history/reports/` | Prototype phase reports (40-example mock evidence) |
| `docs/reports/supervisor_progress_report.md` | `docs/history/reports/` | Superseded progress report |
| `docs/development/agent-notes.md` | `docs/history/agent-notes.md` | Superseded git rules ("work on main") |
| `docs/paper/faar_aaai_findings.tex` | `docs/history/status-report-2026-08/faar_aaai_findings.tex` | Dated status report, not the paper |
| `docs/output/pdf/faar_aaai_findings.pdf` | `docs/history/status-report-2026-08/faar_aaai_findings.pdf` | Its compiled PDF (built before the 2026-09-28 one-sentence edit) |
| `docs/index.md` | removed; `docs/README.md` | Duplicate documentation home |

New homes: `paper/` (single current paper source), `experiments/` (run
registry and record format).

Unchanged on purpose:

- `artifacts/`, `logs/`, `data/phase0/`: prototype evidence that `src/faar/settings.py`, the demo and about 25 tests use as default paths. Moving them would change behaviour. `docs/evidence/evidence-manifest.tsv` indexes them.
- `OHR-Bench/`: vendored upstream benchmark with its README and copyright statement; the QA lock and audit pin it by path.
- `results/`, `config/`, `data/`: the frozen pilot and audit record these paths inside their digests. Moving them would break `build_pilot.py` validation.
- `cluster/`, `scripts/`, `src/`, `tests/`, `annotation/`, `examples/`, `assets/`.

## Relocation map (parent workspace, outside Git)

| Old path | New path | Reason |
| --- | --- | --- |
| `paperss/` | `archive/paper-drafts/` | Four earlier external draft PDFs |
| `agent_review_context/` | `archive/review-bundle-2026-06-01/agent_review_context/` | Extracted review bundle |
| `ZIPs/agent_review_context.zip` | `archive/review-bundle-2026-06-01/agent_review_context.zip` | Same bundle, zipped |
| `ZIPs/literature-view.zip` | `archive/literature-view.zip` | Nested Notion export of a literature-review page |
| `ZIPs/` | removed (empty) | |

`Literature/` and `literature_review_output/` keep their names.

Moves inside the repository used `git mv`, so `git log --follow` traces each
file. Markdown link targets inside the moved historical documents were
re-pointed with the same prefix map so the history stays browsable. Body text,
backticked paths and link labels were left as written, except the four
"current protocol is …" banner labels, which now name `docs/research/`.

## Cleanup ledger

### Deleted (all git-ignored, none tracked, 491 files)

No active process used them (checked with `ps` and `lsof`).

| Item | Files | Why disposable | Rebuild |
| --- | ---: | --- | --- |
| `__pycache__/` outside the virtual environments (root, `src/`, `scripts/`, `cluster/`, `tests/`, `OHR-Bench/`) | 400 (incl. `build/`) | Python bytecode | automatic |
| `.pytest_cache/`, `.ruff_cache/` | 30 | Tool caches | automatic |
| `build/` | 108 | setuptools build tree of `src/faar` | `pip wheel .` |
| `dist/faar-0.1.0-py3-none-any.whl` | 1 | Local wheel from 2026-08-18; sha256 `77449ad0e17e718667aa045be4f72cb7e7668dc41b03539b0d2e943329efb7de` | `python -m build` |
| `dist/faar-0.1.0.tar.gz` | 1 | Local sdist; sha256 `5abee15d8ce2d3b03c3fec462d6c8d3bf3062bf2fbe74120e04caf2c5b8060da` | `python -m build` |
| `src/faar.egg-info/` | 6 | Stale setuptools metadata; both venvs use PEP 660 editable installs with their own `dist-info` | `pip install -e .` |
| `docs/output/pdf/faar_aaai_findings.{aux,fdb_latexmk,fls,log,out}` | 5 | LaTeX byproducts of the status report | recompile the `.tex` |
| `docs/tmp/faar_findings/page-{1..4}.png`, `docs/tmp/pdfs/faar_findings-{1..4}.png` | 8 | Page renders of the status-report PDF | `pdftoppm` on the PDF |
| `tmp/` | 0 | Empty directory | — |

Hashes of the non-bytecode deletions:

```
db25d7f1ee0a17de5a58e2b394671363d8dc083a31c9252d62c86a877cbac5bf  docs/output/pdf/faar_aaai_findings.aux
950a43c00d474c1a81aa284187ab350c5cd475aacbe9bcb76c3a8ce819269199  docs/output/pdf/faar_aaai_findings.fdb_latexmk
14d5ea7d35f7534c91c511780ae466e408c41e3e58816508e2a0be9acea6af66  docs/output/pdf/faar_aaai_findings.fls
2226da3b54640633a89f4e66b18250a22e9244e2977718b683ef619f34bc5f7d  docs/output/pdf/faar_aaai_findings.log
6a484a3e5489ef449273ca43ef17bdc1fe391af328165948336e6732471c6b79  docs/output/pdf/faar_aaai_findings.out
94053a9ee83a569748ded1ed84f6616a2df6e606ee27a9b55e995032fc134bcf  docs/tmp/faar_findings/page-1.png
521f49a13329f6b219e883053161008be79f8d044560a4f90e1af2c60b468464  docs/tmp/faar_findings/page-2.png
889ec527fea2171462be82a3572a88efbde347479bb7750bbcf4c562686874d6  docs/tmp/faar_findings/page-3.png
9c3c54552d4661abbbc3c3865d56bddd0cba36b266cc2b060ae7263455579901  docs/tmp/faar_findings/page-4.png
e380328c8d48e4f26d9068020570ea404f1f599b2ff46a34f9c7b5aff85c153c  docs/tmp/pdfs/faar_findings-1.png
3760311d4fbde9eefecea9cb9490479933b385c6133174d54a2af6816acef256  docs/tmp/pdfs/faar_findings-2.png
2373498f95698ac67808a94e0c9a12b5f8bf64b2a0ad8c33b13387d0427ee625  docs/tmp/pdfs/faar_findings-3.png
0f943d0ae4843ee41e5cd2ab44c66940375184cc4e41343b32593cf685318b3b  docs/tmp/pdfs/faar_findings-4.png
```

Tracked deletion: `docs/index.md` (a three-line pointer to `docs/README.md`),
recoverable from Git.

### Archived (parent workspace, content unchanged)

`paperss/`, `agent_review_context/` and both ZIPs moved into `archive/` as in the
map above. All 85 files keep the same sha256 before and after the move.
Verified before moving:

- The 48 files of `agent_review_context/` and the 48 members of `agent_review_context.zip` are the same set of contents.
- Its `Literature/` and `literature_review_output/` copies equal the top-level folders byte for byte.
- Each of its 16 `Code/` files matches a blob reachable from the local branches `main`, `faar-aaai-experiments` and `hardening/review-pass` (`git hash-object` against `git rev-list --objects`).
- `literature-view.zip` contains one nested ZIP with a Notion Markdown export (6,802 bytes) that exists nowhere else, so it is kept.

Nothing was deleted from the parent workspace. The extracted review bundle is an
exact duplicate of its ZIP; it was kept because deletion outside Git cannot be
undone except by unzipping.

### Kept on purpose

- `.venv/` (1.2 GB, prototype environment) and `.venv-aaai/` (2.6 GB, pinned environment): environments are not general cleanup; both have editable installs pointing at `Code/src`.
- `.env` (git-ignored, never staged), `AGENT.MD` (git-ignored local agent notes), `.DS_Store` files.
- `data/` in full, including `data/ohr_bench_raw/pdfs.zip`; `OHR-Bench/`; `artifacts/`; `logs/`; `results/`; `assets/demo/`.
- Two git stashes on `main` (`backup-untracked-after-reset`, `autostash`): not inspected for deletion and left untouched.
- Prototype page images already tracked under `artifacts/phase0/page_images/` (up to 11 MB each): history was not rewritten.

### Unresolved

- `results/pilots/ohr_dev_v1/inspection/pages/` (17 PNGs, 31.6 MB) and `data/ohr_bench_raw/pdfs.zip` exist only in this checkout. Hashes are recorded; there is no backup.
- ArXivQA bulk assets named by `data/benchmark_prep/arxivqa/paper_inventory.json` are absent here; whether a copy exists elsewhere is unknown.
- `.venv/` may be superseded by `.venv-aaai/`; not verified, so kept.
