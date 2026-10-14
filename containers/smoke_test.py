#!/usr/bin/env python3
"""GPU smoke test for the BabelBrain image (run through bb-run.sh on a GPU node).

1. Imports: the environment, then the BabelBrain modules a headless run needs, at the checked-out ref.
2. GPU backends: CuPy/CUDA devices and versions, OpenCL platforms and devices, and which
   BabelViscoFDTD backends load.
3. BabelBrain's own CUDA GPU functions: initialise the five used in step 1 and compare the
   GPU median filter with scipy.
4. A tiny water-only BabelViscoFDTD case on CUDA (and OpenCL): a CW monopole at 650 kHz,
   9 points per wavelength, 96^3 grid. Checks the field against p ~ exp(-ikr)/r: amplitude
   ratio and phase slope (phase speed) between on-axis sensors.

Prints one PASS/FAIL line per check and SMOKE PASS / SMOKE FAIL at the end (exit 0 / 1).
Results also go to $BB_OUT_DIR/smoke_test.json when BB_OUT_DIR is set.
"""
import contextlib
import importlib
import io
import json
import os
import platform
import sys
import time
import traceback

import numpy as np

RESULTS = {"checks": {}, "info": {}}
FAILS = []


def check(name, ok, detail="", fatal=True):
    tag = "PASS" if ok else ("FAIL" if fatal else "WARN")
    print(f"  [{tag}] {name}: {detail}", flush=True)
    RESULTS["checks"][name] = {"ok": bool(ok), "fatal": fatal, "detail": detail}
    if not ok and fatal:
        FAILS.append(name)


def info(key, value):
    print(f"  [info] {key}: {value}", flush=True)
    RESULTS["info"][key] = value


def quiet(fn, *a, **k):
    """Run fn with its stdout captured (BabelViscoFDTD is chatty); return (result, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ---------------------------------------------------------------- 1. imports
print("== 1. imports", flush=True)
import importlib.metadata as md  # noqa: E402

info("python", platform.python_version())
for d in ["babelviscofdtd", "cupy-cuda12x", "numpy", "scipy", "pyopencl", "pyside6", "vtk", "pyvista"]:
    info(d, md.version(d))
info("bb_commit", os.environ.get("BB_COMMIT", "?"))
info("bb_ref", os.environ.get("BB_REF", "?"))

src = os.environ.get("BB_SRC", "")
# Modules a headless run touches. Optional ones exist on some refs only.
MODULES = [
    ("BabelDatasetPreps", True), ("CalculateMaskProcess", True), ("CalculateFieldProcess", True),
    ("FileManager", True), ("ThermalModeling.CalculateTemperatureEffects", True),
    ("GPUFunctions.GPUMedianFilter.MedianFilter", True),
    ("TranscranialModeling.BabelIntegrationBASE", False),                                      # main
    ("TranscranialModeling.babel_integration.integration_templates.babel_integration_base", False),  # UserTx*
    ("TranscranialModeling.tx_geometries", False),                                             # UserTx*
    ("scripting", False), ("server", False),                                                   # UserTxServerMode
    ("BabelBrain.BabelBrain", False),                                                          # GUI module (Qt offscreen)
]
for mod, required in MODULES:
    path_hint = os.path.join(src, "BabelBrain", *mod.split(".")) + ".py"
    path_hint2 = os.path.join(src, *mod.split(".")) + ".py"
    present = os.path.exists(path_hint) or os.path.exists(path_hint2)
    if not present and not required:
        print(f"  [skip] import {mod}: not on this ref", flush=True)
        continue
    try:
        quiet(importlib.import_module, mod)
        check(f"import {mod}", True, "ok")
    except Exception as e:  # noqa: BLE001
        check(f"import {mod}", False, f"{type(e).__name__}: {e}", fatal=required or present)

# ---------------------------------------------------------------- 2. GPU backends
print("== 2. GPU backends", flush=True)
gpu_name = None
try:
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    names = [cp.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in range(n)]
    gpu_name = names[0] if names else None
    drv = cp.cuda.runtime.driverGetVersion()
    rt = cp.cuda.runtime.runtimeGetVersion()
    nvrtc = cp.cuda.nvrtc.getVersion()
    info("cuda_devices", names)
    info("cuda_driver_api", f"{drv // 1000}.{drv % 1000 // 10}")
    info("cuda_runtime", f"{rt // 1000}.{rt % 1000 // 10}")
    info("nvrtc", f"{nvrtc[0]}.{nvrtc[1]}")
    free, total = cp.cuda.runtime.memGetInfo()
    info("gpu0_memory_free_total_GB", [round(free / 1e9, 1), round(total / 1e9, 1)])
    # A raw kernel forces an NVRTC compile. Compare with a tolerance: a float32 reduction on the
    # GPU adds in a different order from NumPy, so its last bits legitimately differ.
    k = cp.RawKernel(r'''extern "C" __global__ void axpb(const float* x, float* y, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x; if (i < n) y[i] = 2.0f * x[i] + 1.0f; }''', "axpb")
    nel = 1 << 20
    x = cp.arange(nel, dtype=cp.float32)
    y = cp.empty_like(x)
    k(((nel + 255) // 256,), (256,), (x, y, cp.int32(nel)))
    exact = bool((cp.asnumpy(y) == 2 * np.arange(nel, dtype=np.float32) + 1).all())
    total = float(cp.sum(x * 2, dtype=cp.float64))
    check("cupy kernel (NVRTC compile + run)", exact and np.isclose(total, nel * (nel - 1.0), rtol=1e-12),
          f"{n} device(s): {', '.join(names)}; raw kernel exact: {exact}; float64 sum {total:.6e}")
except Exception as e:  # noqa: BLE001
    check("cupy CUDA", False, f"{type(e).__name__}: {e}")

try:
    import pyopencl as cl
    devs = [(p.name, d.name) for p in cl.get_platforms() for d in p.get_devices()]
    check("OpenCL devices", len(devs) > 0, str(devs), fatal=False)
except Exception as e:  # noqa: BLE001
    check("OpenCL devices", False, f"{type(e).__name__}: {e}", fatal=False)

# PropagationModel prints which backends it loaded only on first import, and earlier sections
# may already have imported it, so read the module's attributes instead of its messages.
pm, txt = quiet(importlib.import_module, "BabelViscoFDTD.PropagationModel")
loaded = [b for b in ("StaggeredFDTD_3D_CPU", "StaggeredFDTD_3D_CUDA", "StaggeredFDTD_3D_OPENCL",
                      "StaggeredFDTD_3D_METAL", "StaggeredFDTD_3D_MLX") if hasattr(pm, b)]
info("BabelViscoFDTD backends loaded", loaded)
check("BabelViscoFDTD CUDA backend", "StaggeredFDTD_3D_CUDA" in loaded, txt.strip().replace("\n", "; "))

# ---------------------------------------------------------------- 3. BabelBrain GPU functions
print("== 3. BabelBrain GPU functions (CUDA)", flush=True)
if gpu_name:
    try:
        from scipy import ndimage
        mods = {}
        for sub, init in [("GPUMedianFilter.MedianFilter", "InitMedianFilter"), ("GPUVoxelize.Voxelize", "InitVoxelize"),
                          ("GPUMapping.MappingFilter", "InitMapFilter"), ("GPUResample.Resample", "InitResample"),
                          ("GPUBinaryClosing.BinaryClosing", "InitBinaryClosing")]:
            try:
                m = importlib.import_module("GPUFunctions." + sub)
                quiet(getattr(m, init), DeviceName=gpu_name, GPUBackend="CUDA")
                mods[sub] = m
                check(f"BabelBrain {init}(CUDA)", True, "kernels built")
            except Exception as e:  # noqa: BLE001
                check(f"BabelBrain {init}(CUDA)", False, f"{type(e).__name__}: {e}")
        if "GPUMedianFilter.MedianFilter" in mods:
            rng = np.random.default_rng(0)
            vol = rng.integers(0, 255, size=(48, 40, 32), dtype=np.uint8)
            out, _ = quiet(mods["GPUMedianFilter.MedianFilter"].MedianFilter, vol, 3, GPUBackend="CUDA")
            ref = ndimage.median_filter(vol, 3, mode="nearest")
            inner = (slice(1, -1),) * 3
            nbad = int(np.count_nonzero(out[inner] != ref[inner]))
            check("BabelBrain MedianFilter(CUDA) == scipy (interior)", nbad == 0, f"{nbad} differing voxels of {ref[inner].size}")
    except Exception as e:  # noqa: BLE001
        check("BabelBrain GPU functions", False, f"{type(e).__name__}: {e}")
else:
    check("BabelBrain GPU functions", False, "no CUDA device")


# ---------------------------------------------------------------- 4. tiny water-only FDTD case
def water_case(backend, backend_id, dev_name):
    from BabelViscoFDTD.PropagationModel import PropagationModel
    f0, c0, rho = 650e3, 1500.0, 1000.0
    ppw = 9
    h = c0 / f0 / ppw                          # 0.2564 mm, as prod9
    N = 96
    props = np.array([[rho, c0, 0.0, 0.0, 0.0]])
    model = PropagationModel()
    z = np.zeros((4, 4, 4), np.uint32)
    (okp, dt_stable), _ = quiet(model.StaggeredFDTD_3D_with_relaxation, z, props, f0, z, np.zeros((1, 2), np.float32),
                                h, 1e-6, z, CheckOnlyParams=True, TypeSource=2, AlphaCFL=1.0)
    dt = 0.8 * float(dt_stable)
    duration = 30e-6
    nt = int(np.ceil(duration / dt))
    t = np.arange(nt + 2) * dt
    ramp = np.clip(t * f0 / 4.0, 0, 1)         # 4-period linear ramp
    pulse = (ramp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)[None, :]
    mat = np.zeros((N, N, N), np.uint32)
    c = N // 2
    smap = np.zeros((N, N, N), np.uint32); smap[c, c, c] = 1
    ox = np.zeros((N, N, N), np.float32); ox[c, c, c] = 1.0
    # Sensors on +x, in voxels from the source. At 9 points per wavelength, 2 voxels is an 80 deg
    # phase step, safely under the 180 deg np.unwrap can resolve. (A 6-voxel spacing is 240 deg,
    # which unwraps to -120 deg and halves the fitted slope, doubling the apparent phase speed.)
    offs = list(range(8, 27, 2))
    sens = np.zeros((N, N, N), np.uint32)
    for o in offs:
        sens[c + o, c, c] = 1
    period = 1 / (f0 * dt)
    t0 = time.time()
    res, log = quiet(model.StaggeredFDTD_3D_with_relaxation, mat, props, f0, smap, pulse, h, duration, sens,
                     Ox=ox, TypeSource=2, DT=dt, AlphaCFL=1.0, SelRMSorPeak=1, SelMapsRMSPeakList=["Pressure"],
                     SelMapsSensorsList=["Pressure"], SensorSubSampling=1, SensorStart=0,
                     COMPUTING_BACKEND=backend_id, USE_SINGLE=True, SILENT=1,
                     DefaultGPUDeviceName=dev_name, DefaultGPUDeviceNumber=0)
    wall = time.time() - t0
    p = np.asarray(res[0]["Pressure"], np.float64)
    tt = np.asarray(res[0]["time"], np.float64)[: p.shape[1]]
    p = p[:, : len(tt)]
    keep = tt >= tt[-1] - 6 / f0              # last 6 periods: steady state everywhere
    w = 2 * np.pi * f0
    A = np.stack([np.cos(w * tt[keep]), np.sin(w * tt[keep])], 1)
    coef = np.linalg.lstsq(A, p[:, keep].T, rcond=None)[0]   # p = a cos + b sin = Re(P e^{-iwt}), P = a + i b
    P = coef[0] + 1j * coef[1]
    resid = np.linalg.norm(p[:, keep].T - A @ coef, axis=0) / np.linalg.norm(p[:, keep].T, axis=0)
    r = np.array(offs) * h
    amp_r = np.abs(P) * r                     # constant for a monopole
    amp_dev = float(np.max(np.abs(amp_r / amp_r.mean() - 1)))
    ph = np.unwrap(np.angle(P))
    slope = np.polyfit(r, ph, 1)[0]           # rad/m; |slope| = k = w / c
    c_est = w / abs(slope)
    out = dict(backend=backend, grid=[N, N, N], h_mm=h * 1e3, dt_ns=dt * 1e9, steps=nt,
               steps_per_period=period, wall_s=wall, gvox_steps_per_s=N ** 3 * nt / wall / 1e9,
               finite=bool(np.isfinite(p).all()), max_abs=float(np.abs(p).max()),
               amp_times_r_max_dev=amp_dev, phase_speed_m_s=c_est, phasor_resid_max=float(resid.max()))
    return out


print("== 4. tiny water-only BabelViscoFDTD case", flush=True)
for backend, bid, fatal in [("CUDA", 1, True), ("OpenCL", 2, False)]:
    try:
        r = water_case(backend, bid, gpu_name or "")
        RESULTS[f"water_{backend}"] = r
        ok = r["finite"] and r["max_abs"] > 0 and r["amp_times_r_max_dev"] < 0.10 and abs(r["phase_speed_m_s"] / 1500 - 1) < 0.02
        check(f"water case {backend}", ok,
              f"{r['grid'][0]}^3, {r['steps']} steps in {r['wall_s']:.1f} s ({r['gvox_steps_per_s']:.2f} Gvox-steps/s); "
              f"|p|*r spread {r['amp_times_r_max_dev']:.3f} (< 0.10), phase speed {r['phase_speed_m_s']:.1f} m/s "
              f"(1500 +- 2 %), phasor residual {r['phasor_resid_max']:.1e}", fatal=fatal)
    except Exception as e:  # noqa: BLE001
        check(f"water case {backend}", False, f"{type(e).__name__}: {e}", fatal=fatal)
        traceback.print_exc()

RESULTS["fails"] = FAILS
if os.environ.get("BB_OUT_DIR"):
    os.makedirs(os.environ["BB_OUT_DIR"], exist_ok=True)
    with open(os.path.join(os.environ["BB_OUT_DIR"], "smoke_test.json"), "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
print("SMOKE PASS" if not FAILS else f"SMOKE FAIL ({len(FAILS)}): {FAILS}", flush=True)
sys.exit(0 if not FAILS else 1)
