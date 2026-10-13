#!/usr/bin/env bash
# In-image entrypoint (installed as /opt/bb/bin/bb-entry; the image's %runscript).
# Puts a BabelBrain checkout on PYTHONPATH, prints provenance, checks the checkout's
# dependency pins against this image, then execs the command.
#
#   BB_SRC         BabelBrain checkout inside the container (required; bb-run.sh sets it)
#   BB_COMMIT      resolved commit SHA (bb-run.sh sets it; else read from BB_SRC/.git)
#   BB_REF         ref requested (informational)
#   BB_OUT_DIR     if set, provenance.json is written there
#   BB_DEPCHECK    warn (default) | strict (exit 3 on unexpected pin mismatch) | off
#   BB_XVFB        1 = run the command under xvfb-run (virtual X server, Mesa GL) instead of
#                  Qt's offscreen platform; only for GUI-driving code that needs real GL windows
#   BB_QUIET       1 = shorter banner
set -euo pipefail

: "${BB_SRC:?BB_SRC is not set (path of the BabelBrain checkout inside the container)}"
[ -d "$BB_SRC/BabelBrain" ] || { echo "bb-entry: $BB_SRC does not look like a BabelBrain checkout" >&2; exit 2; }

export PYTHONPATH="$BB_SRC:$BB_SRC/BabelBrain${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/xdg-$(id -u)}"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
# Agate runs transparent hugepages "always" (defrag "madvise"), and its nodes hold a large page cache.
# Fresh multi-GB host arrays can then stall for an hour or more in direct reclaim and compaction.
# BB_THP_DISABLE=1 (default) turns THP off for the run: NumPy's madvise is skipped here, and the
# final command runs under prctl(PR_SET_THP_DISABLE), which children inherit. Set 0 to opt out.
export BB_THP_DISABLE="${BB_THP_DISABLE:-1}"
[ "$BB_THP_DISABLE" = 1 ] && export NUMPY_MADVISE_HUGEPAGE=0
if [ -n "${BB_CACHE_DIR:-}" ]; then
    export CUPY_CACHE_DIR="${CUPY_CACHE_DIR:-$BB_CACHE_DIR/cupy}"
    export PYOPENCL_CACHE_DIR="${PYOPENCL_CACHE_DIR:-$BB_CACHE_DIR/pyopencl}"
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$BB_CACHE_DIR/numba}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$BB_CACHE_DIR/matplotlib}"
fi

if [ -z "${BB_COMMIT:-}" ]; then
    BB_COMMIT="$(git -c safe.directory='*' -C "$BB_SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
DIRTY="$(git -c safe.directory='*' -C "$BB_SRC" status --porcelain --untracked-files=no 2>/dev/null | wc -l || echo '?')"
export BB_COMMIT BB_DIRTY="$DIRTY"

label() { sed -n "s/.*\"$1\": *\"\\([^\"]*\\)\".*/\\1/p" /.singularity.d/labels.json 2>/dev/null | head -1; }
echo "[bb] ================================================================"
echo "[bb] BabelBrain commit : $BB_COMMIT (ref ${BB_REF:-?}; modified tracked files: $DIRTY)"
echo "[bb] source          : $BB_SRC"
echo "[bb] image           : ${APPTAINER_CONTAINER:-?} (built $(cat /opt/bb/BUILD_DATE 2>/dev/null || echo ?), lock $(label bb.requirements_lock_sha256 | cut -c1-12))"
if [ "${BB_QUIET:-0}" != 1 ]; then
    python - <<'PY' || true
import importlib.metadata as md, platform
v = lambda d: md.version(d)
print(f"[bb] python {platform.python_version()}, BabelViscoFDTD {v('babelviscofdtd')}, cupy {v('cupy-cuda12x')}, "
      f"numpy {v('numpy')}, PySide6 {v('pyside6')}, vtk {v('vtk')}")
PY
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
        | sed 's/^/[bb] gpu /' || echo "[bb] gpu: nvidia-smi not available (no --nv or no GPU)"
fi
echo "[bb] host $(hostname), job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}, $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -n "${BB_OUT_DIR:-}" ]; then
    mkdir -p "$BB_OUT_DIR"
    python - "$BB_OUT_DIR/provenance.json" "$@" <<'PY' || echo "[bb] could not write provenance.json" >&2
import json, os, sys, platform, importlib.metadata as md, datetime
out, cmd = sys.argv[1], sys.argv[2:]
try:
    labels = json.load(open("/.singularity.d/labels.json"))
except Exception:
    labels = {}
rec = dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           bb_ref=os.environ.get("BB_REF"), bb_commit=os.environ.get("BB_COMMIT"),
           bb_dirty_tracked_files=os.environ.get("BB_DIRTY"), bb_src=os.environ.get("BB_SRC"),
           repo_url=os.environ.get("BB_REPO_URL"), image=os.environ.get("APPTAINER_CONTAINER"),
           image_labels={k: v for k, v in labels.items() if k.startswith("bb.")},
           python=platform.python_version(), babelviscofdtd=md.version("babelviscofdtd"),
           cupy=md.version("cupy-cuda12x"), host=platform.node(),
           slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
           thp_disabled=os.environ.get("BB_THP_DISABLE") == "1", command=cmd)
# One record per run: append to a JSON-lines log, and keep the latest as provenance.json.
with open(out + "l", "a") as f:
    f.write(json.dumps(rec) + "\n")
json.dump(rec, open(out, "w"), indent=1)
print(f"[bb] provenance -> {out}")
PY
fi

if [ "${BB_DEPCHECK:-warn}" != off ]; then
    set +e
    python /opt/bb/bin/depcheck.py "$BB_SRC"
    rc=$?
    set -e
    if [ $rc -ne 0 ] && [ "${BB_DEPCHECK:-warn}" = strict ]; then
        echo "[bb] BB_DEPCHECK=strict: stopping (see above)" >&2; exit 3
    fi
fi
echo "[bb] ================================================================"

[ $# -gt 0 ] || set -- python
if [ "${BB_XVFB:-0}" = 1 ]; then
    unset QT_QPA_PLATFORM
    set -- xvfb-run -a -s "-screen 0 1920x1080x24" "$@"
fi
if [ "$BB_THP_DISABLE" = 1 ]; then
    # PR_SET_THP_DISABLE = 41. The setting survives execve, so it covers the command and its children.
    exec python -c '
import ctypes, os, sys
rc = ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)
state = next((l.split()[1] for l in open("/proc/self/status") if l.startswith("THP_enabled")), "?")
print(f"[bb] THP disabled for this run: prctl rc={rc}, THP_enabled={state}", file=sys.stderr, flush=True)
os.execvp(sys.argv[1], sys.argv[1:])' "$@"
fi
exec "$@"
