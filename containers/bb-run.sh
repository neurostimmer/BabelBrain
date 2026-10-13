#!/usr/bin/env bash
# Run BabelBrain code at a chosen ref inside the environment-only Apptainer image.
#
#   bb-run.sh [run] <command ...>   check out BB_REF (or use BB_SRC), then run the command in the image
#   bb-run.sh fetch                 create or update the bare mirror of BB_REPO_URL (needs internet)
#   bb-run.sh resolve [REF]         print the commit SHA that REF resolves to (fetches first)
#   bb-run.sh checkout [REF]        create the shared checkout for REF and print its path
#   bb-run.sh shell                 interactive shell in the image with the code set up
#   bb-run.sh info                  image labels and build record
#
# Examples
#   BB_REF=UserTxServerMode bb-run.sh python /opt/bb/bin/smoke_test.py
#   BB_REF=$(bb-run.sh resolve antiphon) sbatch ...        # pin a SHA before submitting an array
#   BB_SRC=$HOME/BabelBrain bb-run.sh python my_case.py  # a checkout you manage yourself
#   bb-run.sh python BabelBrain/BabelBrain.py --headless --execute script.py
#
# Settings come from the environment or from a config file (first found): $BB_CONFIG,
# <this dir>/config.sh, ~/.config/babelbrain-msi/config.sh. See config.example.sh. Also:
#   BB_FETCH      auto (default: fetch unless BB_REF is a SHA already present) | 1 | 0
#   BB_TREE       shared (default: one checkout per SHA under $BB_SCRATCH/src/trees) |
#                 private (copy it to job-local $TMPDIR first, for code that writes into its tree)
#   BB_OUT_DIR    run output directory: provenance.json(l) is written there
#   BB_PASS_ENV   extra variable names to pass into the container (space-separated)
#   BB_DEPCHECK   warn | strict | off;   BB_XVFB=1: run under xvfb-run;   BB_NV=0: no --nv
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in "${BB_CONFIG:-}" "$HERE/config.sh" "$HOME/.config/babelbrain-msi/config.sh"; do
    if [ -n "$f" ] && [ -f "$f" ]; then
        # shellcheck disable=SC1090
        source "$f"; BB_CONFIG="$f"; break
    fi
done
: "${BB_SCRATCH:?set BB_SCRATCH (config.sh or environment), e.g. /scratch.global/<x500>/babelbrain_mod}"
BB_IMAGE="${BB_IMAGE:-$BB_SCRATCH/images/babelbrain_env.sif}"
BB_REPO_URL="${BB_REPO_URL:-https://github.com/neurostimmer/BabelBrain.git}"
BB_REF="${BB_REF:-main}"
BB_CACHE_DIR="${BB_CACHE_DIR:-$BB_SCRATCH/cache}"
MIRROR="$BB_SCRATCH/src/BabelBrain.git"
TREES="$BB_SCRATCH/src/trees"
LOCK="$BB_SCRATCH/src/.lock"

log() { echo "[bb-run] $*" >&2; }
git_m() { git -C "$MIRROR" "$@"; }

ensure_mirror() {
    mkdir -p "$BB_SCRATCH/src" "$TREES"
    if [ ! -d "$MIRROR" ]; then
        log "creating mirror of $BB_REPO_URL"
        git init -q --bare "$MIRROR"
        git_m remote add origin "$BB_REPO_URL"
        git_m config remote.origin.fetch '+refs/heads/*:refs/heads/*'
        git_m config --add remote.origin.fetch '+refs/tags/*:refs/tags/*'
        git_m config gc.auto 0
    fi
    if [ "$(git_m remote get-url origin)" != "$BB_REPO_URL" ]; then
        log "mirror remote is $(git_m remote get-url origin), not $BB_REPO_URL; set BB_REPO_URL or BB_SCRATCH"; exit 2
    fi
}

do_fetch() {
    ensure_mirror
    log "fetching $BB_REPO_URL"
    git_m fetch -q --prune origin || { log "git fetch failed (no internet on this node? fetch on a login node: bb-run.sh fetch)"; return 1; }
}

is_sha() { [[ "$1" =~ ^[0-9a-f]{7,40}$ ]]; }

resolve() {   # REF -> full SHA; fetches unless told not to
    local ref="$1" sha=""
    ensure_mirror
    case "${BB_FETCH:-auto}" in
        0) ;;
        1) do_fetch ;;
        auto) if ! (is_sha "$ref" && git_m cat-file -e "$ref^{commit}" 2>/dev/null); then do_fetch || true; fi ;;
    esac
    for cand in "refs/heads/$ref" "refs/tags/$ref" "$ref"; do
        if sha="$(git_m rev-parse -q --verify "$cand^{commit}" 2>/dev/null)"; then echo "$sha"; return 0; fi
    done
    log "cannot resolve ref '$ref' in $BB_REPO_URL"; return 1
}

checkout() {  # SHA -> path of a shared, detached worktree
    local sha="$1" dir="$TREES/$1"
    if [ ! -f "$dir/.bb_ready" ]; then
        rm -rf "$dir"
        git_m worktree prune
        git_m worktree add -q --detach "$dir" "$sha"
        touch "$dir/.bb_ready"
        log "checked out $sha -> $dir"
    fi
    echo "$dir"
}

with_lock() { mkdir -p "$BB_SCRATCH/src"; ( flock -w 900 9 || { log "timed out waiting for $LOCK"; exit 1; }; "$@" ) 9>"$LOCK"; }

prepare_code() {   # sets SRC, COMMIT
    if [ -n "${BB_SRC:-}" ]; then
        SRC="$(cd "$BB_SRC" && pwd)"
        COMMIT="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
        BB_REF="${BB_REF_LABEL:-BB_SRC}"
        log "using checkout $SRC at $COMMIT"
    else
        COMMIT="$(with_lock resolve "$BB_REF")"
        SRC="$(with_lock checkout "$COMMIT")"
        log "ref $BB_REF -> $COMMIT"
    fi
    if [ "${BB_TREE:-shared}" = private ]; then
        local priv="${TMPDIR:-/tmp}/bb_src_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
        rm -rf "$priv"; cp -a "$SRC" "$priv"; SRC="$priv"
        log "private copy of the code: $SRC"
    fi
}

container() {   # run "$@" inside the image with code, binds and environment set up
    [ -f "$BB_IMAGE" ] || { log "image not found: $BB_IMAGE (build it with build.sbatch)"; exit 2; }
    command -v apptainer >/dev/null || { log "apptainer not on PATH"; exit 2; }
    mkdir -p "$BB_CACHE_DIR"
    local binds="$BB_SCRATCH,$SRC" b
    [ -n "${BB_OUT_DIR:-}" ] && { mkdir -p "$BB_OUT_DIR"; binds="$binds,$BB_OUT_DIR"; }
    [ -n "${BB_SRC:-}" ] && [ -e "$SRC/.git" ] && [ -f "$SRC/.git" ] && \
        binds="$binds,$(sed -n 's/^gitdir: \(.*\)\/worktrees\/.*/\1/p' "$SRC/.git")"
    if [ -n "${BB_BINDS:-}" ]; then binds="$binds,$BB_BINDS"; fi
    [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ] && binds="$binds,$TMPDIR"
    # Pass settings as APPTAINERENV_* (survives --cleanenv; values may contain commas).
    export APPTAINERENV_BB_SRC="$SRC" APPTAINERENV_BB_COMMIT="$COMMIT" APPTAINERENV_BB_REF="$BB_REF" \
           APPTAINERENV_BB_REPO_URL="$BB_REPO_URL" APPTAINERENV_BB_CACHE_DIR="$BB_CACHE_DIR"
    for b in BB_OUT_DIR BB_DEPCHECK BB_DEPS_ALLOW BB_XVFB BB_QUIET TMPDIR CUDA_VISIBLE_DEVICES OMP_NUM_THREADS \
             SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURM_CPUS_PER_TASK SLURM_JOB_NAME \
             ${BB_PASS_ENV:-}; do
        if [ -n "${!b:-}" ]; then export "APPTAINERENV_$b=${!b}"; fi
    done
    local args=(--cleanenv --bind "$binds")
    [ "${BB_NV:-1}" = 1 ] && args+=(--nv)
    log "image $BB_IMAGE"
    exec apptainer "$1" "${args[@]}" "$BB_IMAGE" "${@:2}"
}

cmd="${1:-run}"
case "$cmd" in
    fetch)    with_lock do_fetch ;;
    resolve)  with_lock resolve "${2:-$BB_REF}" ;;
    checkout) sha="$(with_lock resolve "${2:-$BB_REF}")"; with_lock checkout "$sha" ;;
    info)     apptainer inspect --labels "$BB_IMAGE"; apptainer exec "$BB_IMAGE" cat /opt/bb/BUILD_INFO.json ;;
    shell)    prepare_code; container exec /opt/bb/bin/bb-entry bash ;;
    run)      shift || true; prepare_code; container run "$@" ;;
    *)        prepare_code; container run "$@" ;;
esac
