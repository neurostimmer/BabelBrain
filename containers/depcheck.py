#!/usr/bin/env python3
"""Compare a BabelBrain checkout's pinned dependencies with what this image has installed.

Code is not baked into the image, so a branch that changes its dependencies would silently
run against the wrong environment. This reads the checkout's
environment_linux_cuda_13_driver_580.yml (falls back to environment_linux.yml), takes its
pip pins plus the conda pins that have a PyPI equivalent here, and compares them with the
installed distributions. Known, deliberate differences are listed in
/opt/bb/env/deps_allow.txt (from deps_allow.txt in the recipe).

Exit 0: everything matches or is a known difference. Exit 1: a pin differs or a package is
missing, so review it and rebuild if the change matters. Usage: depcheck.py <checkout>
"""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import sys

ENV_FILES = ["environment_linux_cuda_13_driver_580.yml", "environment_linux.yml"]
# conda package name -> installed distribution name (conda-only packages are not compared)
CONDA_MAP = {"numpy": "numpy", "scipy": "scipy", "numba": "numba", "llvmlite": "llvmlite",
             "cupy": "cupy-cuda12x", "fastrlock": "fastrlock"}
ALLOW_FILE = os.environ.get("BB_DEPS_ALLOW", "/opt/bb/env/deps_allow.txt")


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_env(path):
    """Return ({dist: (op, version)} for pip pins and mapped conda pins, python version or None)."""
    pins, py = {}, None
    pip_indent = None                       # indent of the "- pip:" line while inside its list
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        item = line.strip()
        if pip_indent is not None and indent <= pip_indent:
            pip_indent = None
        if item.startswith("- pip:"):
            pip_indent = indent
            continue
        if not item.startswith("- ") or indent == 0:
            continue
        spec = item[2:].strip()
        if pip_indent is not None:
            m = re.match(r"([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=)?\s*([^\s;]*)", spec)
            if m:
                pins[norm(m.group(1))] = (m.group(2) or "", m.group(3) or "")
        else:
            parts = spec.split("=")
            name = norm(parts[0])
            if name == "python" and len(parts) > 1:
                py = parts[1]
            elif name in CONDA_MAP and len(parts) > 1 and parts[1]:
                pins.setdefault(norm(CONDA_MAP[name]), ("==", parts[1]))
    return pins, py


def satisfied(have, op, ver):
    if not ver:
        return True
    try:
        from packaging.version import Version
        h, v = Version(have), Version(ver)
    except Exception:
        h, v = have, ver
    return {"==": h == v, ">=": h >= v, "<=": h <= v, "~=": h >= v, "": True}.get(op, h == v)


def load_allow():
    allow = {}
    if os.path.exists(ALLOW_FILE):
        for raw in open(ALLOW_FILE, encoding="utf-8"):
            line = raw.split("#", 1)[0].strip()
            if line:
                name, _, why = line.partition(" ")
                allow[norm(name)] = why.strip() or "listed in deps_allow.txt"
    return allow


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BB_SRC", ".")
    env = next((os.path.join(src, f) for f in ENV_FILES if os.path.exists(os.path.join(src, f))), None)
    if env is None:
        print(f"[depcheck] no environment file in {src}; nothing to compare")
        return 0
    sha = hashlib.sha256(open(env, "rb").read()).hexdigest()
    basis = os.environ.get("BB_ENV_BASIS_SHA256", "")
    if not basis:
        try:
            basis = json.load(open("/.singularity.d/labels.json")).get("bb.env_basis_sha256", "")
        except Exception:
            basis = ""
    pins, py = parse_env(env)
    installed = {norm(d.metadata["Name"]): d.version for d in md.distributions()}
    allow = load_allow()
    ok, known, bad = 0, [], []
    for name, (op, ver) in sorted(pins.items()):
        have = installed.get(name)
        if have is None:
            msg = f"{name}{op}{ver}: MISSING"
        elif satisfied(have, op, ver):
            ok += 1
            continue
        else:
            msg = f"{name}{op}{ver}: image has {have}"
        (known if name in allow else bad).append(msg + (f"  [known: {allow[name]}]" if name in allow else ""))
    pyv = platform.python_version()
    if py and not pyv.startswith(".".join(py.split(".")[:2])):
        bad.append(f"python={py}: image has {pyv}")
    rel = os.path.basename(env)
    same = "same as" if sha == basis else "DIFFERS from"
    print(f"[depcheck] {rel} sha256 {sha[:12]} ({same} the image's basis {basis[:12] or '?'}); "
          f"{ok} pins match, {len(known)} known differences, {len(bad)} unexpected")
    for m in known:
        print(f"[depcheck]   known    {m}")
    for m in bad:
        print(f"[depcheck]   MISMATCH {m}")
    if bad:
        print("[depcheck] This ref pins dependencies the image does not have. If the change matters, update "
              "requirements.in, regenerate requirements.lock.txt and rebuild (README.md, 'Dependency changes'); "
              "if it is deliberate, add it to deps_allow.txt.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
