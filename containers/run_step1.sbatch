#!/bin/bash
#SBATCH --job-name=bb_step1
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=bb_step1_%j.out
# BabelBrain step 1 only (the simulation domain: head model from the segmentation and CT/pseudo-CT), headless,
# in the container. Its outputs, <out>/<ID>/<prefix>BabelViscoInput.nii.gz with CT.nii.gz, CT-cal.npz and
# AirRegions.nii.gz on the CT path, can be reused by any number of step-2 runs
# (run_element_array.sbatch with EA_ARGS="--mask <...BabelViscoInput.nii.gz> --use-ct").
# Submit from the recipe directory with:
#   sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" \
#     --export=ALL,BB_REF=<ref>,S1_ARGS="<run_step1.py args>",S1_OUT=<provenance dir> run_step1.sbatch
# The limits above are defaults; size each run from measured ones (command-line options override them).
set -euo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    [ -n "$f" ] && [ -f "$f" ] && { . "$f"; break; }
done
: "${S1_ARGS:?S1_ARGS (run_step1.py arguments) is required}"
[ -n "${S1_OUT:-}" ] && export BB_OUT_DIR="$S1_OUT"
echo "[step1] $S1_ARGS"
# shellcheck disable=SC2016,SC2086
"$RECIPE/bb-run.sh" run bash -c 'exec python "$BB_SRC/OfflineBatchExamples/ElementArray/run_step1.py" "$@"' _ \
    $S1_ARGS --backend cuda
