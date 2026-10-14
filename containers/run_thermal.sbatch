#!/bin/bash
#SBATCH --job-name=bb_thermal
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=bb_thermal_%j.out
# BabelBrain step 3 (thermal) on a step-2 head result, headless, in the container
# (OfflineBatchExamples/ElementArray/run_thermal.py). The step-2 directory must hold the head result and its
# water-only companion, which run_element_array.py writes for head runs.
# Submit from the recipe directory with:
#   sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" \
#     --export=ALL,BB_REF=<ref>,TH_STEP2=<step-2 out dir>,TH_PROFILE=<profile .yaml>,TH_OUT=<output dir>,\
#              TH_ARGS="<extra run_thermal.py args, e.g. --isppa drive>" run_thermal.sbatch
# The limits above are defaults; size each run from measured ones (command-line options override them).
set -euo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    [ -n "$f" ] && [ -f "$f" ] && { . "$f"; break; }
done
: "${TH_STEP2:?TH_STEP2 is required}" "${TH_PROFILE:?TH_PROFILE is required}" "${TH_OUT:?TH_OUT is required}"
mkdir -p "$TH_OUT"
export BB_OUT_DIR="$TH_OUT"
echo "[thermal] step 2 $TH_STEP2; profile $TH_PROFILE"
# shellcheck disable=SC2016,SC2086
"$RECIPE/bb-run.sh" run bash -c 'exec python "$BB_SRC/OfflineBatchExamples/ElementArray/run_thermal.py" "$@"' _ \
    --step2 "$TH_STEP2" --profile "$TH_PROFILE" --out "$TH_OUT" --backend cuda ${TH_ARGS:-}
