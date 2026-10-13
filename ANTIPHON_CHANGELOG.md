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
| `containers/README.md` | added | 2026-10-13 | Recipe: build and run the MSI Apptainer image; switch code refs without rebuild |
| `containers/babelbrain.def` | added | 2026-10-13 | Apptainer definition: CUDA 12.9, Python 3.11, BabelViscoFDTD 1.2.7; no repo code baked in |
| `containers/bb-entry.sh` | added | 2026-10-13 | Container entrypoint: headless env, provenance, dependency check, THP disabled by default |
| `containers/bb-run.sh` | added | 2026-10-13 | Wrapper: fetch the fork and run at BB_REF (branch/tag/SHA) or BB_SRC |
| `containers/build.sbatch` | added | 2026-10-13 | Slurm CPU job that builds the image |
| `containers/config.example.sh` | added | 2026-10-13 | Template for site settings (real values live in an ignored config.sh) |
| `containers/depcheck.py` | added | 2026-10-13 | Flags when a ref needs dependencies the image lacks |
| `containers/deps_allow.txt` | added | 2026-10-13 | Allow-list for depcheck.py |
| `containers/requirements.in` | added | 2026-10-13 | Top-level container dependencies |
| `containers/requirements.lock.txt` | added | 2026-10-13 | Hash-pinned container dependency lock (cp311 manylinux) |
| `containers/smoke_test.py` | added | 2026-10-13 | GPU smoke test: imports, CUDA backend, tiny water FDTD |
| `containers/smoke_test.sbatch` | added | 2026-10-13 | Slurm GPU job for the smoke test |

## Log

### 2026-10-13: Establish the Antiphon branch

- modified `.gitignore`: appended three ignore rules (`.claude/`, `CLAUDE.local.md`, `**/CLAUDE.local.md`) so local Claude Code settings, memory and notes are never tracked. Upstream's own `CLAUDE.md` is untouched.
- added `ANTIPHON_CHANGELOG.md`: this file.

### 2026-10-13: MSI container recipe

This adds an Apptainer recipe for running BabelBrain without the GUI on MSI GPU nodes: CUDA 12.9, Python 3.11, BabelViscoFDTD 1.2.7, CuPy 13.6.0 and NumPy 1.26.4. The image holds only the environment. `bb-run.sh` fetches the fork at run time by `BB_REF` (branch, tag or SHA), so new code needs no rebuild, and `depcheck.py` reports when a ref needs packages the image lacks. Site settings go in an untracked `config.sh`.
The local development environment isn't tracked: its recipe lives in the git-ignored `env/` and the uv venv in the git-ignored `venv/`.
- added `containers/README.md`
- added `containers/babelbrain.def`
- added `containers/bb-entry.sh`
- added `containers/bb-run.sh`
- added `containers/build.sbatch`
- added `containers/config.example.sh`
- added `containers/depcheck.py`
- added `containers/deps_allow.txt`
- added `containers/requirements.in`
- added `containers/requirements.lock.txt`
- added `containers/smoke_test.py`
- added `containers/smoke_test.sbatch`
- modified `ANTIPHON_CHANGELOG.md`: added the table rows and this entry.

### 2026-10-13: Disable transparent hugepages in MSI runs

Agate nodes run transparent hugepages `always` (defrag `madvise`) and hold a large page cache. Jobs that allocate multi-GB host arrays have stalled for an hour or more in direct reclaim and compaction, as seen in the prod9 elastic runs. The cause is likely, not yet measured.
- modified `containers/bb-entry.sh`: `BB_THP_DISABLE=1` is the default.
  - It exports `NUMPY_MADVISE_HUGEPAGE=0` and starts the final command under `prctl(PR_SET_THP_DISABLE)`, which children inherit.
  - It logs the kernel's `THP_enabled` flag and records `thp_disabled` in `provenance.json`.
  - `BB_THP_DISABLE=0` opts out.
  - Tested locally: with it on, the command sees `THP_enabled: 0`; with it off, it sees 1.
- modified `ANTIPHON_CHANGELOG.md`: updated the table row and added this entry.
