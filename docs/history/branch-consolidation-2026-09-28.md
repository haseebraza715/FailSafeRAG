# Branch consolidation, 2026-09-28

This record describes how the reviewed work reached `main` on
`origin` (github.com/haseebraza715/FailSafeRAG). Remote state comes from
`git fetch --prune origin` and `git ls-remote --heads origin`, run at the start.

## Starting state

| Ref | Tip |
| --- | --- |
| local `main` | `c65efe3a` (38 commits behind `origin/main`) |
| `origin/main` | `253b255d8ee8a7fb016a240c6dc60c9b11acdbec` |
| `faar-aaai-experiments`, `origin/faar-aaai-experiments` | `25dcae2da72b351df60b1ee999e621364329d97d` |
| `hardening/review-pass` (local only) | `22be5e29be5e75191c7979b9414dc9f041b72987` |
| `chore/research-workspace-structure` (local only) | `79b14fb`, 129 commits ahead of `origin/main` |
| `origin/laptop-recovery/2026-09-22/failsaferag-source-only` | `8e0a76410de0e63674c32fd4ec16e8186b4ff4dc` |

One worktree (this checkout), no open pull requests (PR 1 is merged), and no
branch protection on `main`. No other git or pipeline process was running.

Stashes, kept unchanged:

- `stash@{0}` `30403f83772f710c6e9b4ff49cd3a62e1aa21182` "On main: backup-untracked-after-reset"
- `stash@{1}` `f7871f03d7becd7c9a94e343256b46583cb21537` "On main: autostash"

## Branches

"Unique" counts commits not reachable from the publication candidate
(`chore/research-workspace-structure` plus this record's commit).

| Branch | Unique | Status | Worktree or PR | Decision |
| --- | ---: | --- | --- | --- |
| `origin/main` | 0 | Ancestor of the candidate | none | Fast-forward to the candidate |
| local `main` | 0 | Ancestor of `origin/main` | none | Fast-forward to the candidate |
| `faar-aaai-experiments` (local and remote) | 0 | Included | none | Delete both after publication |
| `hardening/review-pass` | 0 | Included | none | Delete after publication |
| `chore/research-workspace-structure` | 0 | Is the candidate | current checkout | Delete after `main` points at it |
| `origin/laptop-recovery/2026-09-22/failsaferag-source-only` | 1 | Redundant snapshot | none | Tag `recovery/laptop-2026-09-22-source-only`, push the tag, then delete the branch |

## The recovery commit

`8e0a764` "Preserve source state without local data artifacts" (2026-09-22)
is one commit on top of `origin/main`. It changes 136 files. It is a snapshot,
not new work:

- Every one of its 3,741 blobs already exists in the candidate's history. It holds no file content that the candidate lacks.
- Its tree matches no single candidate commit. The closest is `22be5e2` (51 files differ). The differences are older documentation from before the reorganisations, and the absence of `scripts/data/*.py`, `results/environment/pip-freeze.txt` and `tests/test_corpus_cache.py`. That pattern fits a copy made while data paths were excluded.
- Merging it would delete current scripts and research records without adding anything. It is not merged.

The tag keeps the snapshot reachable on `origin` after the branch is deleted.

## Documentation changes made for the merge

- `README.md`, `SUPERVISOR_HANDOFF.md` and `docs/operations/runbook.md` now check out `main` instead of `faar-aaai-experiments`. The README no longer says to avoid running from `main`.
- `README.md` and `docs/research/aaai-plan.md` state that the study brief takes precedence over the earlier protocol where they conflict.

Historical documents keep their original branch names.
