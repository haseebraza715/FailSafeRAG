# Instruction audit, 2026-09-28

Audit of the repository's agent instructions and writing skills after they were
created. Line numbers refer to the files as committed with this report.

## How the audit ran

- `/doctor prompt-audit` did not run. It is a terminal command, and this session ran in the Claude desktop Code tab, where `/doctor` is not available.
- The audit ran through the bundled `claude-api` skill with the `prompt-audit` subcommand. That skill loaded from a bundled-skills directory labelled `2.1.281`, while the installed CLI reports `2.1.283 (Claude Code)`. The procedure followed is its `shared/prompt-audit.md`.
- **Scope:** `AGENTS.md`, `CLAUDE.md`, `.claude/skills/technical-writing/SKILL.md`, `.claude/skills/unslop/SKILL.md`. Personal and global instruction files were read only to detect name clashes, and were not audited or changed.
- **Target model:** Claude Opus 5.5 (`claude-opus-5-5`), as the task named.
- No behavioural probe ran against another model instance. The audit spawned no subagents or model calls.

## Inventory

| File | Lines | Loaded |
| --- | ---: | --- |
| `CLAUDE.md` | 7 | At session start, when the session starts in this repository |
| `AGENTS.md` | 71 | Through the `@AGENTS.md` import in `CLAUDE.md` |
| `.claude/skills/technical-writing/SKILL.md` | 139 | Description at start; body when the skill is invoked |
| `.claude/skills/unslop/SKILL.md` | 80 | Description at start; body when invoked, or read by path |

## Summary

Five findings were applied: three in `AGENTS.md` and two in the skills. Three
more are recorded as flags. The biggest risk was ambiguity rather than dated
prompting. The original research rule "Do not change an experiment and re-run
it until it produces a positive result" could be read as permission to wait for
a positive result. The files contain no pressure language, reasoning scaffolds,
update suppressors or anti-formatting rules. Their prohibitions are research,
data or git constraints with stated reasons, which the procedure keeps.

## Findings

| # | Location | Evidence | Pattern | Why | Confidence | Action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `AGENTS.md:27` | "Do not change an experiment and re-run it until it produces a positive result." | Conflicting or ambiguous instruction | Current models follow instructions literally. "until" can attach to "re-run" or to "Do not", and the second reading inverts the safeguard. | High | Applied: "Do not keep changing and re-running an experiment in search of a positive result." |
| 2 | `.claude/skills/technical-writing/SKILL.md:104` | "four spaces for Python, as `ruff` enforces" | Volatile factual claim (Group 2) | `pyproject.toml` selects ruff rules E4, E7, E9, F and I only. Indentation is not checked, so the claim is false. | High | Applied: removed "as `ruff` enforces". |
| 3 | `.claude/skills/unslop/SKILL.md:50` | "Use periods or commas only (no parentheses, …)" | Conflicting instructions across files | Read literally, it bans all parentheses, which conflicts with the Global English rule on full-unit parentheses in `technical-writing/SKILL.md:92` and with repository docs. Its own next sentence shows the rule targets dash substitutes. | Medium | Applied: now bans parentheses, en dashes and spaced hyphens used as dash substitutes. |
| 4 | `AGENTS.md:23` | "Use the kinds in `experiments/README.md`: `engineering_check`, … and data preparation is a fourth kind." | Mid-sentence colon, inconsistent naming | The kind was named in prose instead of by its record value `data_preparation`, so one thing had two names. | Medium | Applied: lists the four record values. |
| 5 | `AGENTS.md:53` | "revalidates the frozen pilot" with a command fixed to `ohr_dev_v1` | Volatile specifics | The command is verified, but naming one pilot as "the" pilot goes stale when a second pilot is frozen. | Medium | Applied: says the command checks one pilot and should run for each config in `config/pilots/`. |
| 6 | `CLAUDE.md:7`; `~/.claude/skills/unslop/` | Project skill `unslop` shares its name with the owner's personal skill | Skill that cannot trigger as intended | Claude Code gives a personal skill precedence over a project skill with the same name. For this owner, `/unslop` and automatic loading resolve to the personal copy, which lacks the project's adaptations. | Medium | Flag. `CLAUDE.md` and `technical-writing` now direct Claude to read the project file by path. Renaming the project skill (for example `faar-unslop`) would remove the clash, but the task asked for `unslop`. |
| 7 | `CLAUDE.md:5` | "Claude Code reads `AGENTS.md` by itself only when no `CLAUDE.md` exists, so this file imports it." | Possible explanation the model does not need | It explains the import to people editing the file and prevents its accidental removal. Keeping it costs one line. | Low | Flag, no edit. |
| 8 | `AGENTS.md:61`, `technical-writing/SKILL.md:104`, checklist item 8 | Counts and paths must be true at the commit | Duplicate rule | The copies agree. The task asked for the rule in `AGENTS.md`, and the procedure keeps redundancy that agrees. | Low | Flag, no edit. |

## Diff applied

```diff
--- a/AGENTS.md
+++ b/AGENTS.md
-- Label every run by what it measured. Use the kinds in `experiments/README.md`: `engineering_check`, `development_pilot` and `scientific_evaluation` are different claims, and data preparation is a fourth kind. Mock-backend and smoke runs are engineering checks even when they report EM or F1.
+- Label every run by what it measured, using the kinds in `experiments/README.md` (`data_preparation`, `engineering_check`, `development_pilot` and `scientific_evaluation`). Each kind supports a different claim. Mock-backend and smoke runs are engineering checks even when they report EM or F1.
-- Do not tune on test outcomes. Do not change an experiment and re-run it until it produces a positive result. A null or negative result is evidence.
+- Do not tune on test outcomes. Do not keep changing and re-running an experiment in search of a positive result. A null or negative result is evidence.
-The last command revalidates the frozen pilot and writes nothing.
+The last command revalidates one frozen pilot and writes nothing. Run it for each config in `config/pilots/`.
--- a/.claude/skills/technical-writing/SKILL.md
+++ b/.claude/skills/technical-writing/SKILL.md
-... four spaces for Python, as `ruff` enforces. Write real paths ...
+... four spaces for Python. Write real paths ...
--- a/.claude/skills/unslop/SKILL.md
+++ b/.claude/skills/unslop/SKILL.md
-13. **Em dash overuse.** Avoid em dashes entirely. Use periods or commas only (no parentheses, no en dashes, no hyphen-as-dash substitutes). ...
+13. **Em dash overuse.** Avoid em dashes entirely. Use a period or a comma instead. Don't swap in parentheses, an en dash, or a spaced hyphen as a dash substitute. ...
```

## Proposed, not applied

- Finding 6: rename the project skill directory and `name` to `faar-unslop`, then update `technical-writing/SKILL.md`, `CLAUDE.md` and `AGENTS.md`. The owner decides, because it changes the requested path.

## Other checks

- Frontmatter parses as YAML. Each `name` matches its directory, and each description is third person, 340 to 358 characters, with no XML. Neither skill sets `disable-model-invocation`, so both can load automatically.
- `CLAUDE.md` holds the only `@` import (`@AGENTS.md`). `AGENTS.md` imports nothing, so there is no cycle. The memory docs state that keeping this import never loads `AGENTS.md` twice.
- The skills contain no machine-specific paths. They reference each other as `../unslop/SKILL.md` and `../technical-writing/SKILL.md`.
- Every relative Markdown link in the repository resolves, including the historical documents.
- Every command in `AGENTS.md` ran successfully: `ruff check .`, `registry.py check`, `build_pilot.py --no-render` for each pilot config, and one focused `pytest` file. `latexmk` is installed.
- Remaining legacy instruction files: `docs/history/agent-notes.md` and `docs/history/legacy-agent-instructions.md`. Both are labelled superseded, and `AGENTS.md` tells agents not to follow `docs/history/`.

## Not verified

- Whether Claude Code actually loads `CLAUDE.md`, `AGENTS.md` and the two skills in a new session. `/memory`, `/skills` and `/context` are interactive commands that did not run here. Check them in the next session that starts in `Code/`.
- A session started in the parent workspace folder loads `Code/CLAUDE.md` only when it reads files under `Code/`. It loads nested project skills only after it edits a file there.
- Whether the skills improve writing consistently. Three hand-worked examples are recorded below as illustrations, not as evidence.

## Writing-skill examples

These are before-and-after rewrites made by applying `technical-writing` and
`unslop` to three short samples. Every number in each "before" appears
unchanged in its "after".

**README instruction.** Before: "In order to validate the frozen pilot, it is important to note that the builder script should simply be run with the --no-render flag, which ensures that nothing is overwritten — a crucial safeguard for the integrity of our selection." After: "To validate the frozen pilot without overwriting anything, run the builder script with `--no-render`."

**Experiment status.** Before: "The data audit has been successfully completed, showcasing the robustness of our pipeline. MinerU text could potentially cover most questions: 8,256 of 8,498 questions have non-empty evidence pages, while 237 have empty pages and 5 are missing. This pivotal milestone sets the stage for the upcoming calibration." After: "The data audit is complete. MinerU text is non-empty on the evidence pages of 8,256 of 8,498 questions. 237 questions have an empty evidence page and 5 have a missing one. The calibration comes next."

**Paper paragraph.** Before: "Our typed recovery approach significantly improves answer accuracy over random recovery, leveraging a quality gate to facilitate targeted repairs across the 70 pilot questions, while also reducing visual calls in a budget-matched comparison." After: "We will compare typed recovery with random recovery on paired questions, holding the retriever, the answer model and the evidence budget fixed. A budget-matched comparison will give both policies the same number of visual calls. The 70 questions of the development pilot serve mechanism development only. No accuracy has been measured on them, and results are pending." The claim changes on purpose: the "before" states an unmeasured improvement as a result, and the manuscript rules require it to read as planned work. The number 70 is unchanged.
