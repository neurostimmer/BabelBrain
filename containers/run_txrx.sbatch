#!/bin/bash
#SBATCH --job-name=bb_txrx
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=bb_txrx_%j.out
# Transmit-receive runs of BabelBrain's ElementArray transducer (OfflineBatchExamples/ElementArray/run_txrx.py),
# headless, in the container. TX_ARGS is one run_txrx.py invocation (domain built once, then one FDTD run per
# transmit set and parameter set); TX_ARGS2, if set, is a second invocation run after it in the same job (for
# example the water reference on the same mask with --water-only). Submit from the recipe directory with:
#   sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" \
#     --export=ALL,BB_REF=<ref>,TX_ARGS="<args>"[,TX_ARGS2="<args>"] run_txrx.sbatch
# The limits above are defaults; size each run from measured ones (command-line options override them).
set -euo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    [ -n "$f" ] && [ -f "$f" ] && { . "$f"; break; }
done
: "${TX_ARGS:?TX_ARGS (run_txrx.py arguments) is required}"
for a in "$TX_ARGS" "${TX_ARGS2:-}"; do
    [ -z "$a" ] && continue
    echo "[txrx] $a"
    # shellcheck disable=SC2016,SC2086
    "$RECIPE/bb-run.sh" run bash -c 'exec python "$BB_SRC/OfflineBatchExamples/ElementArray/run_txrx.py" "$@"' _ \
        $a --backend cuda
done
