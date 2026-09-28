@AGENTS.md

## Claude Code

Claude Code reads `AGENTS.md` by itself only when no `CLAUDE.md` exists, so this file imports it. The shared rules live there. Put only Claude-specific notes here.

The writing skills are project skills in `.claude/skills/`. `technical-writing` can load automatically. A personal skill with the same name overrides a project skill, and some users have a personal `unslop`. When `technical-writing` asks for unslop, read `.claude/skills/unslop/SKILL.md` by its path so the project version applies.
