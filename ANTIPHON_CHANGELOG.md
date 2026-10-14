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
| `containers/babelbrain.def` | added | 2026-10-13 | Apptainer definition: CUDA 12.9, Python 3.12, BabelViscoFDTD 1.2.7; no repo code baked in |
| `containers/bb-entry.sh` | added | 2026-10-13 | Container entrypoint: headless env, provenance, dependency check, THP disabled by default |
| `containers/bb-run.sh` | added | 2026-10-13 | Wrapper: fetch the fork and run at BB_REF (branch/tag/SHA) or BB_SRC |
| `containers/build.sbatch` | added | 2026-10-13 | Slurm CPU job that builds the image |
| `containers/config.example.sh` | added | 2026-10-13 | Template for site settings (real values live in an ignored config.sh) |
| `containers/depcheck.py` | added | 2026-10-13 | Flags when a ref needs dependencies the image lacks |
| `containers/deps_allow.txt` | added | 2026-10-13 | Allow-list for depcheck.py |
| `containers/requirements.in` | added | 2026-10-13 | Top-level container dependencies |
| `containers/requirements.lock.txt` | added | 2026-10-13 | Hash-pinned container dependency lock (cp312 manylinux) |
| `containers/smoke_test.py` | added | 2026-10-13 | GPU smoke test: imports, CUDA backend, tiny water FDTD |
| `containers/smoke_test.sbatch` | added | 2026-10-13 | Slurm GPU job for the smoke test |
| `TranscranialModeling/BabelIntegrationELEMENT_ARRAY.py` | added | 2026-10-14 | Generic element-array transducer: per-element complex drive from a definition file; volumetric per-element sources; box domain around all elements |
| `TranscranialModeling/BabelIntegrationBASE.py` | modified | 2026-10-14 | Region hook `DomeRegionMap` in dome mode (default behaviour unchanged) |
| `BabelBrain/CalculateFieldProcess.py` | modified | 2026-10-14 | Dispatch entry for `TxSystem == 'ElementArray'` |
| `OfflineBatchExamples/ElementArray/run_element_array.py` | added | 2026-10-14 | Headless driver for ElementArray: water or step-1 mask; writes field metrics (peak, half-max zone, envelope, calibration scale) |
| `containers/run_element_array.sbatch` | added | 2026-10-14 | Slurm GPU job running the driver in the container |

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

### 2026-10-14: MSI image on Python 3.12; smoke-test fixes

The first image's GPU smoke test (MSI job 3604094) failed for all three refs. One failure was real: upstream `main` uses Python 3.12 f-string syntax, so Python 3.11 can't import `BabelBrain.py` or `CalculateTemperatureEffects.py`. The other three were bugs in the smoke test itself; the image's CUDA works.
- modified `containers/requirements.in`, `containers/requirements.lock.txt`: Python 3.12, pinned to upstream main's `environment_win-314.yml` (NumPy 2.3.5, CuPy 13.6.0, SciPy 1.17.1, pandas 3.0.2, VTK 9.6.1) plus PySide6 6.9.3 from `environment_mac_*-314.yml`. BabelViscoFDTD stays at 1.2.7. 83 packages, all cp312 manylinux wheels. Not 3.14: CuPy 13.6.0 has wheels for cp39-cp313 only.
- modified `containers/babelbrain.def`: Python 3.12.13; labels for the new basis, including `bb.env_basis_sha256`.
- modified `containers/depcheck.py`: compares against `environment_win-314.yml` first, then the older files. It now compares any conda pin that names an installed package, and `deps_allow.txt` can excuse the Python version.
- modified `containers/deps_allow.txt`: Python 3.12 vs 3.14; BabelViscoFDTD 1.2.7 vs main's 1.2.6 and 1.2.4.
- modified `containers/smoke_test.py`, fixing three test bugs:
  - CuPy: a raw NVRTC kernel checked exactly, plus a float64 sum, instead of an exact float32 sum comparison.
  - BabelViscoFDTD backends: read from the module's attributes, because import messages print only once.
  - Water case: sensors every 2 voxels (80° per step). At 6 voxels (240°), `np.unwrap` folded the phase and reported 2987.6 m/s. The fixed case measures 1503.1 m/s on the local CPU.
- modified `containers/README.md`: versions, basis, and depcheck behaviour.
- modified `ANTIPHON_CHANGELOG.md`: table rows and this entry.
- The rebuild (MSI job 3668317) and smoke test (3668318) are queued; results go in the next entry.

### 2026-10-14: Generic element-array transducer

A new transducer family for arrays defined entirely by a file, including arrays with elements on several sides of the target. Each element is voxelised into the grid as its own stress source, following DomeTx's approach. The differences from DomeTx:
- each element's complex drive sets both its amplitude and its phase;
- the element footprint is normalised by voxel count;
- the domain is a box around all elements on every side of the focus;
- sensors cover the whole interior;
- there is no Rayleigh pre-step, built-in steering or refocusing, because the drive file carries them;
- the source ramps up over 4 cycles, as BabelBrain's other families do.

The definition-file format (`element_array_v1`) is documented in the module docstring.
- added `TranscranialModeling/BabelIntegrationELEMENT_ARRAY.py`
- modified `TranscranialModeling/BabelIntegrationBASE.py`: in `UpdateConditions` with `DomeType=True`, a subclass may define `DomeRegionMap(xfield, yfield, zfield)` returning normalised coordinates and the region that must fit in the domain. Without it, the existing hemisphere is used unchanged.
- modified `BabelBrain/CalculateFieldProcess.py`: `TxSystem == 'ElementArray'` imports the new module; `kargs` must include `ElementArrayFile`.
- Tested locally on the CPU through `RUN_SIM.RunCases`, in water at PPW 3, with two opposing 8×8 panels of 2.4 mm square elements on a 30 mm sphere. The domain grew on both sides; all 128 elements got their own source voxels; the pressure peak landed on the target voxel. GPU runs follow on MSI.
- modified `ANTIPHON_CHANGELOG.md`: table rows and this entry.

### 2026-10-14: Headless driver for ElementArray, and its MSI job

- added `OfflineBatchExamples/ElementArray/run_element_array.py`: runs BabelBrain's acoustic step (`RUN_SIM.RunCases`) for a definition file without the GUI. It runs either a synthetic water case (a cubic mask with the target at its centre) or a head case on a step-1 mask. It writes `metrics.json`: the peak and the connected half-maximum zone in voxels and in the Tx frame, |p| at the target, and the calibration scale for a target pressure. With `--envelope-axis`, the same metrics are computed on the standing-wave envelope (|p|² averaged over λ/2 along that axis).
  - BabelBrain stores `TargetLocation` in the simulation's z orientation but saves the volumes flipped along z. The driver maps it and cross-checks it against the coordinate vectors.
  - Tested locally on the CPU with a steered synthetic drive: the envelope half-maximum centroid landed at x = 1.99 mm for 2.0 mm requested.
- added `containers/run_element_array.sbatch`: a GPU job (48 GB, 1 h by default) that runs the driver at `BB_REF` in the container.
- modified `ANTIPHON_CHANGELOG.md`: table rows and this entry.

### 2026-10-14: Centred envelope window in the ElementArray driver

- modified `OfflineBatchExamples/ElementArray/run_element_array.py`: the envelope moving-average window is now odd. An even window shifts the envelope by half a voxel; on the first MSI water run that read as a 0.141 mm axial offset that wasn't in the field, whose raw half-max centroid was exactly on the target.
- modified `ANTIPHON_CHANGELOG.md`: this entry.

### 2026-10-14: Off-target analysis and box water mask in the ElementArray driver

- modified `OfflineBatchExamples/ElementArray/run_element_array.py`:
  - `--water-mask-mm` takes one value (a cube) or three (a box).
  - `--offtarget` adds field-derived off-target metrics in the Tx frame:
    - focal = max |p| within `--focal-radius-mm` of the target;
    - envelope = an odd max filter at least one standing-wave fringe wide;
    - main lobe = connected envelope ≥ 0.5 focal containing the target, with extents per axis;
    - off-target = max |p| outside the main lobe dilated by `--main-dilate-mm`, inside `--region-half-mm` along `--region-axis`;
    - the envelope at points `--face-offsets-mm` in front of each element face.
  - Tested locally on the CPU (synthetic panels).
- modified `ANTIPHON_CHANGELOG.md`: this entry.
