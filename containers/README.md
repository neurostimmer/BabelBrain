# BabelBrain on an HPC GPU cluster (Apptainer, headless)

This is an environment-only Apptainer image for running BabelBrain without a display on NVIDIA A100/H100 nodes. It was written for MSI's agate cluster, but no site values are hard-coded.

- The image holds Python, CUDA user-space libraries and the pinned packages. It holds **no BabelBrain code**.
- `bb-run.sh` checks out the requested ref of the fork (`BB_REF`: a branch, tag or SHA) into scratch, or uses a checkout you supply (`BB_SRC`). It then runs your command in the image with that code on `PYTHONPATH`.
- A new code version is a `git fetch` plus a new ref, not a rebuild.

| File | Role |
|---|---|
| `babelbrain.def` | Apptainer definition: `nvidia/cuda:12.9.1-base-ubuntu22.04`, CUDA 12.9 libraries, Mesa/Qt/X client libraries, Python 3.11 venv from `requirements.lock.txt` |
| `requirements.in` / `requirements.lock.txt` | Pins: the upstream env file mapped to PyPI wheels; the lock is hash-pinned, wheels only |
| `deps_allow.txt` | Known, deliberate differences from a ref's own env file (read by `depcheck.py`) |
| `bb-entry.sh` | In-image entrypoint (`/opt/bb/bin/bb-entry`, the `%runscript`). See below. |
| `depcheck.py` | Compares the checked-out ref's env pins with the image |
| `bb-run.sh` | Host-side wrapper: mirror, fetch, resolve, checkout, `apptainer run --nv` |
| `build.sbatch` | CPU build job |
| `smoke_test.sbatch` / `smoke_test.py` | GPU smoke test: imports, GPU backends, BabelBrain CUDA GPU functions, and a tiny water-only BabelViscoFDTD case |
| `config.example.sh` | Site settings template. Copy it to `config.sh`, which is never committed. |

`bb-entry.sh` does the following, in order:

1. sets `PYTHONPATH=$BB_SRC:$BB_SRC/BabelBrain` and the headless variables;
2. prints the commit SHA, image, versions and GPU, and writes `provenance.json` when `BB_OUT_DIR` is set;
3. runs `depcheck.py`;
4. execs the command.

## Versions

| | Image | Notes |
|---|---|---|
| CUDA | 12.9 (cudart, NVRTC, cuBLAS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, CCCL, headers) | Needs driver ≥ 575. Agate's A100 nodes run 580.x (CUDA ≤ 13.0). |
| Python | 3.11 (python-build-standalone via uv; 3.11.14 requested, as upstream) | |
| BabelViscoFDTD | 1.2.7 | |
| CuPy | 13.6.0 (`cupy-cuda12x`) | Same release as upstream's CUDA 13 env |
| NumPy / SciPy / PySide6 / VTK | 1.26.4 / 1.16.3 / 6.4.3 / 9.5.2 | Upstream pins |

**Basis.** The basis is upstream `environment_linux_cuda_13_driver_580.yml` on branch `UserTxServerMode` @ `f973f5d` (sha256 `c838524…`). That file is identical on `UserTx`. Its pip section is used verbatim. Its conda packages map to PyPI wheels:

- numpy, scipy, numba, llvmlite, fastrlock and cupy;
- CUDA comes from the NVIDIA apt repo in the base image.

**Deliberate differences**, also marked `DIFF` in `requirements.in`:

- **CUDA 12.9 instead of conda CUDA 13.1.**
  - Driver 580 officially supports CUDA ≤ 13.0, while upstream's file name ("driver_580") relies on minor-version compatibility.
  - CuPy 14, which can use pip-installed CUDA, needs NumPy 2, which upstream's pins exclude.
- **No MKL / `mkl_fft`.** BabelBrain falls back to `numpy.fft` (`try: import mkl_fft`).
- **Added:** `jinja2`, which `UserTx*` imports in `CreateTransducers/transducer_creator.py` but no env file lists; `xmltodict`, which the non-CUDA Linux file lists; `pytest` and `pytest-html`.

**BabelViscoFDTD 1.2.7 vs 1.2.4:**

- Upstream `main` pins 1.2.4. `UserTx`, `UserTxServerMode` and `dualTx` pin 1.2.7.
- 1.2.4 has no Linux wheels (sdist only). 1.2.7 has cp39-cp314 manylinux wheels.
- The 1.2.5-1.2.7 changes are wheel packaging, macOS build fixes, BHTE (thermal) matrix preparation, and choosing the exact GPU by full name. No FDTD solver changes are listed.
- The Antiphon elastic MSI environment also uses 1.2.7, on Python 3.12 + CuPy 14. This image uses Python 3.11, where `distutils` still exists, so the 3.12 import workaround is not needed.

## Upstream branches covered

| Branch | Dependency change vs `main` | Covered |
|---|---|---|
| `UserTx` (custom transducers) | babelviscofdtd 1.2.4 → 1.2.7, + pyvistaqt 0.12.0; imports jinja2 | yes, exact match |
| `UserTxServerMode` (`UserTx` + `--serve`, `--headless --execute`) | Same env file as `UserTx`; server uses stdlib `http.server` only | yes, exact match |
| `dualTx` | babelviscofdtd 1.2.7 | yes |
| `main` | Pins 1.2.4 | yes; 1.2.7 runs, listed in `deps_allow.txt` |

`UserTxServerMode` adds a git submodule (`BabelBrain/ExternalBin/PlanTUS/PlanTUS`). Checkouts do not fetch it. PlanTUS is a GUI planning tool that headless runs don't use.

The SimNIBS mesh conversion (`ExternalBin/SimbNIBSMesh/run_linux.sh`) activates `<simnibs dir>/simnibs_env`. SimNIBS is not in this image, so bind a SimNIBS install with `BB_BINDS` if a run needs it. The elastix binaries ship inside the BabelBrain repository.

## Setup (once per site)

```bash
# login node, in a copy of this directory on the cluster
cp config.example.sh config.sh && $EDITOR config.sh          # BB_SCRATCH, BB_ACCOUNT, ...
source config.sh && mkdir -p "$BB_SCRATCH"/{logs,images}
./bb-run.sh fetch                                            # bare mirror of the fork in $BB_SCRATCH/src
sbatch --test-only -A "$BB_ACCOUNT" build.sbatch             # check the ETA
sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" build.sbatch
```

**What the build job does:**

- First it checks outbound internet: Docker Hub, the NVIDIA apt repo, PyPI and astral.sh. It stops with `FAIL: no outbound internet` if any of them can't be reached.
- It builds with `apptainer build --fakeroot` on node-local `$TMPDIR`. This works without a `/etc/subuid` entry, because Apptainer ≥ 1.1 falls back to its fakeroot-command mode.
- It runs `%test`, then copies the image to `$BB_SCRATCH/images/babelbrain_env_<date>_<lock sha8>_<def sha8>.sif` and repoints the `babelbrain_env.sif` symlink.
- Older images are kept, so running jobs and rollbacks are unaffected.
- `*.build.json`, `*.build_info.json` and `*.installed.txt` sit next to each image.

**Resources and size:**

- The build uses 8 cores, 32 GB and up to 2 h.
- The expected image is about 2-3 GB, mostly PySide6, VTK and the CUDA libraries.

## Running

```bash
./bb-run.sh fetch                                          # login node; compute nodes then need no internet
REF=$(./bb-run.sh resolve antiphon)                          # pin a SHA for reproducible batches
BB_REF=$REF BB_OUT_DIR=$BB_SCRATCH/runs/x ./bb-run.sh python my_case.py
BB_REF=UserTxServerMode ./bb-run.sh python BabelBrain/BabelBrain.py --headless --execute script.py
BB_SRC=$HOME/BabelBrain ./bb-run.sh python my_case.py      # your own (bind-mounted) checkout
./bb-run.sh shell                                          # debug shell with the code set up
sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" --export=ALL,BB_REFS="main UserTxServerMode" smoke_test.sbatch
```

**Switching code.**

- Change `BB_REF`. Each SHA is checked out once, as a detached git worktree under `$BB_SCRATCH/src/trees/<sha>`, and reused by later jobs.
- Fetching and checkout are serialised with `flock`, so array tasks can share them.
- With `BB_FETCH=auto` (the default), a SHA that is already present needs no network. A branch name triggers a fetch, and if the fetch fails the wrapper falls back to the mirror.
- If code writes into its own tree, use `BB_TREE=private`, which copies the tree to job-local `$TMPDIR`.
- Unpushed work: rsync a checkout to scratch and use `BB_SRC`. It is logged with its `HEAD` SHA and the count of modified tracked files.

**Provenance.** Every run prints `[bb] BabelBrain commit : <sha>` to the job log and exports `BB_COMMIT` to the process, so scripts should store it with their outputs. With `BB_OUT_DIR`:

- it writes `provenance.json`, the latest run;
- it appends to `provenance.jsonl`;
- both hold the ref, SHA, image path and labels, versions, host, Slurm IDs and the command.

**Headless.**

- `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg`, `PYVISTA_OFF_SCREEN=true` and `BABEL_NO_PLOTS=1` are set.
- BabelBrain's telemetry is switched off with `MYAPP_DISABLE_TELEMETRY=1`.
- Mesa (llvmpipe/OSMesa/EGL) supplies software OpenGL for VTK.
- `BB_XVFB=1` runs the command under `xvfb-run` instead, for GUI-driving scripts that need real GL windows. It is off by default.

**Environment hygiene.**

- The container runs with `--cleanenv`, so host modules (`LD_LIBRARY_PATH`, `PYTHONPATH`) don't leak in.
- `CUDA_VISIBLE_DEVICES`, `TMPDIR`, `OMP_NUM_THREADS`, the Slurm IDs and the `BB_*` settings are passed through. Add other variables with `BB_PASS_ENV="VAR1 VAR2"`.
- Kernel caches (CuPy, PyOpenCL, Numba) go to `$BB_SCRATCH/cache`.

## Dependency changes (when a rebuild is needed)

Code is fetched at run time, but dependencies are baked into the image. A ref that changes its env file therefore needs a rebuild. Each run detects this as follows:

1. The image labels record `bb.env_basis_sha256` (the upstream env file the image was derived from), `bb.requirements_in_sha256` and `bb.requirements_lock_sha256`. To read them, run `apptainer inspect --labels <sif>` or `./bb-run.sh info`.
2. Before every command, `depcheck.py` reads the checked-out ref's `environment_linux_cuda_13_driver_580.yml`. It reports whether the file's hash equals the image's basis, then compares every pip pin and the mapped conda pins with the installed versions:

   ```
   [depcheck] environment_linux_cuda_13_driver_580.yml sha256 d0d1aa51e019 (DIFFERS from the image's basis c83852477c3a); 66 pins match, 1 known differences, 0 unexpected
   ```

3. An unexpected mismatch prints `MISMATCH` lines. With `BB_DEPCHECK=strict` it also stops the run (exit 3), which is the recommended setting for production batches.

When it fires, do this:

1. Update `requirements.in`.
2. Regenerate the lock with `uv pip compile requirements.in --python-version 3.11 --python-platform x86_64-manylinux_2_28 --no-build --generate-hashes -o requirements.lock.txt`. This reads metadata only and runs anywhere.
3. Rebuild, then update the `bb.env_basis*` labels in `babelbrain.def` if the basis moved.

Use `deps_allow.txt` for differences you accept deliberately.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `image not found` | Build first. `BB_IMAGE` defaults to `$BB_SCRATCH/images/babelbrain_env.sif`. |
| `cannot resolve ref` | Run `./bb-run.sh fetch` on a login node. Check that the branch exists on the fork, not just locally. |
| `FAIL: no outbound internet` in the build | Ask the site how to build images: a different partition, a proxy, or building elsewhere and copying the `.sif`. |
| `[depcheck] MISMATCH` | See "Dependency changes". |
| CuPy `CUDA driver version is insufficient` | You landed on an old-driver node. Use `--constraint="a100\|h100"`. |
| Qt `could not load the Qt platform plugin` | Missing X/GL client library. Run `apptainer exec <sif> bash -c 'ldd /opt/bb/venv/lib/python3.11/site-packages/PySide6/Qt/plugins/platforms/*.so \| grep "not found"'` and add the package to `%post`. |
