# Antiphon changelog

Adds support for the Antiphon transducer to BabelBrain. Device-specific geometry and drive data are not part of this repository; they are loaded at run time from a private file.

This file lists every file the Antiphon work adds, modifies, deletes or renames relative to upstream `ProteusMRIgHIFU/BabelBrain`. Paths are relative to the repository root.

- Each commit that touches files adds a dated entry under **Log** naming every one of those paths, in backticks.
- The **Files touched** table is the cumulative list: one row per path, holding its current status against the upstream base.
- A rename is listed as both its old and new path.

Upstream base: `upstream/main` at `29b1026` (7 Oct 2026).

## Files touched

| Path | Status vs upstream | First touched | Purpose |
|---|---|---|---|
| `.gitignore` | modified | 2026-10-13 | Ignore local Claude Code files: `.claude/` and `CLAUDE.local.md` |
| `ANTIPHON_CHANGELOG.md` | added | 2026-10-13 | This changelog |

## Log

### 2026-10-13: Establish the Antiphon branch

- modified `.gitignore`: appended three ignore rules (`.claude/`, `CLAUDE.local.md`, `**/CLAUDE.local.md`) so local Claude Code settings, memory and notes are never tracked. Upstream's own `CLAUDE.md` is untouched.
- added `ANTIPHON_CHANGELOG.md`: this file.
