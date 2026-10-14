#!/bin/bash
#SBATCH --job-name=bb_head
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=bb_head_%j.out
# Head run for an element-array transducer, headless, in the container:
#   1. BabelBrain step 1 (run_step1.py), unless S1_ARGS is empty (reuse an existing mask);
#   2. BabelBrain step 2 (run_element_array.py) on the step-1 mask (CT/pseudo-CT path).
# Submit from the recipe directory with:
#   sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" \
#     --export=ALL,BB_REF=<ref>,S1_ARGS="<run_step1.py args>",EA_MASK=<...BabelViscoInput.nii.gz>,\
#              EA_ARRAY=<definition .npz>,EA_OUT=<output dir>,EA_ARGS="<extra run_element_array.py args>" \
#     run_head_element_array.sbatch
# Head runs need a time limit of at least 3 h (host copy-back can stall on busy nodes).
set -euo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    [ -n "$f" ] && [ -f "$f" ] && { . "$f"; break; }
done
: "${EA_MASK:?EA_MASK is required}" "${EA_ARRAY:?EA_ARRAY is required}" "${EA_OUT:?EA_OUT is required}"
mkdir -p "$EA_OUT"
export BB_OUT_DIR="$EA_OUT"
if [ -n "${S1_ARGS:-}" ]; then
    echo "[head] step 1: $S1_ARGS"
    # shellcheck disable=SC2016,SC2086
    "$RECIPE/bb-run.sh" run bash -c 'exec python "$BB_SRC/OfflineBatchExamples/ElementArray/run_step1.py" "$@"' _ \
        $S1_ARGS --backend cuda
fi
[ -f "$EA_MASK" ] || { echo "[head] step-1 mask not found: $EA_MASK" >&2; exit 2; }
echo "[head] step 2 on $EA_MASK"
# shellcheck disable=SC2016,SC2086
"$RECIPE/bb-run.sh" run bash -c 'exec python "$BB_SRC/OfflineBatchExamples/ElementArray/run_element_array.py" "$@"' _ \
    --array "$EA_ARRAY" --mask "$EA_MASK" --use-ct --out "$EA_OUT" --backend cuda ${EA_ARGS:-}
