#!/usr/bin/env bash
#SBATCH --job-name=bb_build
#SBATCH --partition=msismall
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=60G
#SBATCH --time=02:00:00
#SBATCH --output=bb_build_%j.out
# Build the BabelBrain environment image (CPU job; needs outbound internet for Docker Hub,
# the NVIDIA apt repository, PyPI and astral.sh). Submit from the recipe directory:
#
#   cd <recipe dir> && sbatch -A "$BB_ACCOUNT" -o "$BB_SCRATCH/logs/%x_%j.out" build.sbatch
#
# Writes $BB_SCRATCH/images/babelbrain_env_<date>_<lock sha>.sif, points the
# babelbrain_env.sif symlink at it (only after a successful build and test), and writes a
# .build.json record next to it. Settings: config.sh (see config.example.sh) or environment.
set -euo pipefail
RECIPE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$RECIPE"
for f in "${BB_CONFIG:-}" "$RECIPE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    if [ -n "$f" ] && [ -f "$f" ]; then source "$f"; break; fi
done
: "${BB_SCRATCH:?set BB_SCRATCH}"
for f in babelbrain.def requirements.in requirements.lock.txt deps_allow.txt bb-entry.sh depcheck.py smoke_test.py; do
    [ -f "$f" ] || { echo "missing $f in $RECIPE"; exit 2; }
done

echo "[build] $(date -u +%FT%TZ) host $(hostname) job ${SLURM_JOB_ID:-none} recipe $RECIPE"
apptainer --version
for url in https://registry-1.docker.io/v2/ https://developer.download.nvidia.com/ https://pypi.org/simple/ https://astral.sh/; do
    code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$url" || true)
    echo "[build] reach $url -> HTTP ${code:-none}"
    [ "${code:-000}" != 000 ] || { echo "[build] FAIL: no outbound internet from $(hostname); this build cannot run here"; exit 3; }
done

LOCK_SHA=$(sha256sum requirements.lock.txt | cut -c1-8)
DEF_SHA=$(sha256sum babelbrain.def | cut -c1-8)
STAMP=$(date -u +%Y%m%d)
mkdir -p "$BB_SCRATCH/images" "$BB_SCRATCH/cache/apptainer"
OUT="$BB_SCRATCH/images/babelbrain_env_${STAMP}_${LOCK_SHA}_${DEF_SHA}.sif"
LOCAL="${TMPDIR:-/tmp}/bb_build_${SLURM_JOB_ID:-$$}"
mkdir -p "$LOCAL"
export APPTAINER_TMPDIR="$LOCAL" APPTAINER_CACHEDIR="$BB_SCRATCH/cache/apptainer"
trap 'rm -rf "$LOCAL"' EXIT
df -h "$LOCAL" | tail -1 | sed 's/^/[build] local tmp: /'

t0=$(date +%s)
# Build to node-local disk, then copy to scratch (fakeroot builds want a local filesystem).
apptainer build --fakeroot "$LOCAL/img.sif" babelbrain.def
echo "[build] build took $(( $(date +%s) - t0 )) s"
apptainer test "$LOCAL/img.sif"
cp "$LOCAL/img.sif" "$OUT.partial" && mv "$OUT.partial" "$OUT"
ln -sfn "$(basename "$OUT")" "$BB_SCRATCH/images/babelbrain_env.sif"

SIZE=$(du -h "$OUT" | cut -f1)
SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
apptainer inspect --labels "$OUT"
apptainer exec "$OUT" cat /opt/bb/BUILD_INFO.json | tee "$OUT.build_info.json"
cat > "$OUT.build.json" <<EOF
{"image": "$OUT", "size": "$SIZE", "sha256": "$SHA", "job": "${SLURM_JOB_ID:-}",
 "host": "$(hostname)", "built_utc": "$(date -u +%FT%TZ)", "build_seconds": $(( $(date +%s) - t0 )),
 "requirements_lock_sha256": "$(sha256sum requirements.lock.txt | cut -d' ' -f1)",
 "def_sha256": "$(sha256sum babelbrain.def | cut -d' ' -f1)", "apptainer": "$(apptainer --version)"}
EOF
apptainer exec "$OUT" cat /opt/bb/env/installed.txt > "$OUT.installed.txt"
echo "[build] image $OUT ($SIZE, sha256 $SHA)"
echo "[build] symlink $BB_SCRATCH/images/babelbrain_env.sif -> $(basename "$OUT")"
echo "BUILD PASS"
