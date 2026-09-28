# Phase 4 Acceptance Checklist

This checklist defines completion criteria for Phase 4 claim refinement.

## Requirement Matrix

| Requirement | Evidence | Status | Notes |
| --- | --- | --- | --- |
| Phase 4 analysis logic implemented in code | `src/faar/phase4_analysis.py` | Complete | Includes summaries, claim checks, baseline deltas, and case studies. |
| CLI command available to run full Phase 4 | `src/faar/cli.py` (`run-phase4`) | Complete | Produces machine-readable artifacts under `artifacts/phase4/`. |
| New logic has dedicated unit and CLI tests | `tests/test_phase4_analysis.py`, `tests/test_phase4_cli.py` | Complete | Includes malformed input and edge-case coverage. |
| Full suite passes after Phase 4 changes | local test run: `./.venv/bin/python -m pytest -q` | Complete | 18 tests passed. |
| Claim outcomes documented with explicit supported/not-supported status | `docs/phases/phase4/claim_refinement.md` | Complete | Prevents overstatement of results. |
| Phase and report indexes updated | `docs/phases/index.md`, `docs/reports/index.md`, `docs/archives/index.md` | Complete | Added Phase 4 navigation entries. |

## Exit Criteria

Phase 4 is complete when:

- all requirement matrix rows are marked `Complete`
- artifacts are generated and linkable
- claim wording is aligned with measured evidence
