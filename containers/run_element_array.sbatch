#!/bin/bash
#SBATCH --job-name=bb_elarr
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=bb_elarr_%j.out
# Run BabelBrain's acoustic step for an element-array definition file, headless, in the container.
# Submit from the recipe directory with:
#   sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" \
#          --export=ALL,BB_REF=<ref>,EA_ARRAY=<definition .npz>,EA_OUT=<output dir>,EA_ARGS="<extra args>" \
#          run_element_array.sbatch
# EA_ARGS is passed to OfflineBatchExamples/ElementArray/run_element_array.py, e.g.
#   "--water-mask-mm 40 --envelope-axis z --calibrate-pa 1e6"   (water)
#   "--mask <step-1 mask> --use-ct"                                (head)
# Head runs at full resolution need --time of at least 3 h (host copy-back can stall).
set -euo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    [ -n "$f" ] && [ -f "$f" ] && { . "$f"; break; }
done
: "${EA_ARRAY:?EA_ARRAY (definition .npz) is required}" "${EA_OUT:?EA_OUT (output dir) is required}"
mkdir -p "$EA_OUT"
export BB_OUT_DIR="$EA_OUT"
# The checkout path is only known inside the container ($BB_SRC), so a shell there expands it.
# shellcheck disable=SC2016,SC2086
"$RECIPE/bb-run.sh" run bash -c 'exec python "$BB_SRC/OfflineBatchExamples/ElementArray/run_element_array.py" "$@"' _ \
    --array "$EA_ARRAY" --out "$EA_OUT" --backend cuda ${EA_ARGS:-}
