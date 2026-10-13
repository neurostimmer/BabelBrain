# Site configuration for bb-run.sh and the sbatch scripts. Copy to config.sh (git-ignored)
# or anywhere else and point BB_CONFIG at it. Sourced by bash; keep it to plain assignments.
# Nothing here is needed inside the image; only bb-run.sh and the sbatch scripts read it.

# Working area on the cluster (images, code mirror, checkouts, caches, logs).
BB_SCRATCH="${BB_SCRATCH:-/scratch.global/<x500>/babelbrain_mod}"

# Slurm account. Use it on the sbatch command line: sbatch -A "$BB_ACCOUNT" ...
BB_ACCOUNT="${BB_ACCOUNT:-<slurm-account>}"

# Image built by build.sbatch.
BB_IMAGE="${BB_IMAGE:-$BB_SCRATCH/images/babelbrain_env.sif}"

# Code: fork URL and the default ref (branch, tag or SHA). BB_SRC=<path> uses an existing
# checkout instead (bind-mounted as is; no fetch).
BB_REPO_URL="${BB_REPO_URL:-https://github.com/neurostimmer/BabelBrain.git}"
BB_REF="${BB_REF:-main}"

# Extra bind mounts for data, comma-separated "host[:container]" (default: same path inside).
# Example: BB_BINDS="/scratch.global/<x500>/data,/home/<group>/<x500>/subjects:/data/subjects"
BB_BINDS="${BB_BINDS:-}"

# Optional: kernel caches (CuPy, PyOpenCL, Numba, Matplotlib). Default: $BB_SCRATCH/cache
# BB_CACHE_DIR="$BB_SCRATCH/cache"
