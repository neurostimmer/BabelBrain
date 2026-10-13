#!/usr/bin/env bash
#SBATCH --job-name=bb_smoke
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=bb_smoke_%j.out
# GPU smoke test of the image at one or more BabelBrain refs: imports, GPU backends,
# BabelBrain's CUDA GPU functions and a tiny water-only BabelViscoFDTD case (smoke_test.py).
# Submit from the recipe directory, after `bb-run.sh fetch` on a login node:
#
#   sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" \
#          --export=ALL,BB_REFS="main UserTx UserTxServerMode" smoke_test.sbatch
#
# Use SHAs in BB_REFS to pin the code (bb-run.sh resolve <ref>). Results:
# $BB_SCRATCH/runs/smoke/<job>/<ref>/{smoke_test.json,provenance.json}.
set -uo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    if [ -n "$f" ] && [ -f "$f" ]; then source "$f"; break; fi
done
: "${BB_SCRATCH:?set BB_SCRATCH}"
export BB_SCRATCH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
REFS="${BB_REFS:-${BB_REF:-main}}"
FAILS=0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
for ref in $REFS; do
    echo "################ ref $ref"
    out="$BB_SCRATCH/runs/smoke/${SLURM_JOB_ID:-local}/${ref//\//_}"
    if BB_REF="$ref" BB_OUT_DIR="$out" "$RECIPE/bb-run.sh" run python /opt/bb/bin/smoke_test.py; then
        echo "ref $ref: SMOKE PASS"
    else
        echo "ref $ref: SMOKE FAIL"; FAILS=$((FAILS + 1))
    fi
done
[ "$FAILS" -eq 0 ] && echo "ALL SMOKE PASS ($REFS)" || { echo "SMOKE FAILURES: $FAILS of $(echo $REFS | wc -w) refs"; exit 1; }
