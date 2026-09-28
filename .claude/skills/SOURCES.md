# Skill sources

The skills in this directory are project copies of the repository owner's
personal writing skills. They were copied on 2026-09-28 from the owner's Codex
skill directory (`technical-writing/SKILL.md`, 130 lines, and `unslop/SKILL.md`,
80 lines) and then adapted as listed below. Nothing here reads those original
files at run time.

The substantive rules are unchanged. The owner's Claude Code copy of `unslop`
differs from the Codex copy in one line (step 3 limits "add voice" to essays,
posts and personal writing). This project copy adopts that change.

## technical-writing

- Frontmatter: removed `disable-model-invocation: true` so the skill can load automatically, and rewrote the description in the third person with the tasks it covers.
- References to `unslop` point at `../unslop/SKILL.md`.
- Code indentation follows the repository's Python convention (four spaces, `ruff`) instead of tabs.
- Commit messages: added the repository's one-line conventional format.
- Added "Scientific manuscript prose" for `paper/main.tex`: manuscript structure instead of Diátaxis modes, and the separation of verified facts, interpretations, proposals and pending results.

## unslop

- Frontmatter: replaced "Must always apply." with a third-person description of what the skill does and when it applies.
- Step 3 and "Adding soul": voice applies to essays, posts and personal writing, not to documentation, reports, PR text, commit messages or the paper. The step 3 wording comes from the owner's Claude Code copy.
- Rule 24: keep uncertainty the evidence requires. Cut stacked qualifiers only.

Later edits made after the instruction audit are listed in
`docs/reports/instruction-audit-2026-09-28.md`.
