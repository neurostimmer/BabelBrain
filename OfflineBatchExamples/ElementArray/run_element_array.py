#!/usr/bin/env python3
"""Run BabelBrain's acoustic step for an element-array transducer without the GUI, and report field metrics.

Inputs: an `element_array_v1` definition file (see TranscranialModeling/BabelIntegrationELEMENT_ARRAY.py)
and either
  --water-mask-mm L   a synthetic water-only case: an L-mm cube mask with the target at its centre, or
  --mask FILE         a BabelBrain step-1 mask (<target>_<f>kHz_<PPW>PPW_BabelViscoInput.nii.gz) for a head
                      run, with the CT files step 1 wrote next to it (--use-ct).
Outputs in --out: BabelBrain's own result files, and metrics.json with:
  peak position and offset from the target (mm), |p| at the target, the connected half-maximum zone
  around the peak (volume, centroid, centroid offset from the target, extents), and the scale that makes
  |p(target)| equal --calibrate-pa (for a calibration run).

Example (water, BabelBrain's default PPW for 650 kHz):
  python run_element_array.py --array drive.npz --water-mask-mm 40 --out runs/centre --backend cuda
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "BabelBrain")]


def default_ppw(f0):
    """BabelBrain's GUI default: 9 PPW at or below 350 kHz, 6 above (BabelBrain.py)."""
    return 9 if f0 <= 350e3 else 6


def gpu_name(backend):
    if backend == "cuda":
        import cupy as cp
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    if backend == "opencl":
        import pyopencl as cl
        return [d.name for p in cl.get_platforms() for d in p.get_devices()][0]
    return "CPU"


def vox_to_tx(D, ijk, shape):
    """Saved-volume voxel (possibly fractional) -> Tx-frame position in mm (target at the origin).
    The saved volumes are flipped along z relative to z_vec."""
    i, j, k = np.asarray(ijk, float)
    xs, ys, zs = (np.asarray(D[v], float) for v in ("x_vec", "y_vec", "z_vec"))
    kz = shape[2] - 1 - k
    return 1e3 * np.array([np.interp(i, np.arange(xs.size), xs), np.interp(j, np.arange(ys.size), ys),
                           np.interp(kz, np.arange(zs.size), zs)])


def half_max_zone(p, peak, h):
    from scipy import ndimage
    lab, _ = ndimage.label(p >= 0.5 * p[tuple(peak)])
    zone = lab == lab[tuple(peak)]
    idx = np.argwhere(zone)
    return zone, idx.mean(0), float(zone.sum() * h ** 3 * 1e9), (np.ptp(idx, axis=0) + 1) * h * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--water-mask-mm", type=float)
    g.add_argument("--mask")
    ap.add_argument("--use-ct", action="store_true", help="head run on step-1 CT outputs (CT/pseudo-CT path)")
    ap.add_argument("--ppw", type=int, default=None, help="default: BabelBrain's GUI default for the frequency")
    ap.add_argument("--backend", choices=["cuda", "opencl", "cpu"], default="cuda")
    ap.add_argument("--calibrate-pa", type=float, default=None)
    ap.add_argument("--envelope-axis", choices=["x", "y", "z"], default=None,
                    help="also report metrics on the standing-wave envelope along this Tx-frame axis: |p|^2 averaged "
                         "over a half-wavelength window (opposing elements make a standing wave along their axis)")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        import nibabel
        import TranscranialModeling.BabelIntegrationBASE as B
        from TranscranialModeling.BabelIntegrationELEMENT_ARRAY import RUN_SIM, LoadElementArray
        from BabelViscoFDTD.H5pySimple import ReadFromH5py
    ea = LoadElementArray(args.array)
    f0 = ea["f0_hz"]
    ppw = args.ppw or default_ppw(f0)
    h = B.GetSmallestSOS(f0, bShear=True) / f0 / ppw
    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "babelbrain")
    case_dir = os.path.join(work, "case")
    os.makedirs(case_dir, exist_ok=True)
    tag = f"_{int(f0 / 1e3)}kHz_{ppw}PPW_"
    if args.water_mask_mm:
        M = int(np.ceil(args.water_mask_mm * 1e-3 / h)) | 1
        mask = np.zeros((M, M, M), np.uint8)
        mask[M // 2, M // 2, M // 2] = 5
        target = "T"
        nibabel.Nifti1Image(mask, np.diag([h * 1e3] * 3 + [1.0])).to_filename(
            os.path.join(case_dir, target + tag + "BabelViscoInput.nii.gz"))
        water = True
    else:
        name = os.path.basename(args.mask)
        if tag + "BabelViscoInput.nii.gz" not in name:
            raise SystemExit(f"mask name must end with {tag}BabelViscoInput.nii.gz for PPW {ppw}")
        target = name.split(tag)[0]
        src = os.path.dirname(os.path.abspath(args.mask))
        for f in os.listdir(src):
            if f.startswith(target + tag):
                shutil.copy2(os.path.join(src, f), case_dir)
        water = False
    backend = {"cuda": 1, "opencl": 2, "cpu": 0}[args.backend]
    dev = gpu_name(args.backend)

    t0 = time.time()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        out = RUN_SIM().RunCases(targets=[target], ID="case", basedir=work + os.sep, Frequencies=[f0],
                                 basePPW=[ppw], bWaterOnly=water, bUseCT=args.use_ct, COMPUTING_BACKEND=backend,
                                 deviceName=dev, ElementArrayFile=os.path.abspath(args.array),
                                 bForceRecalc=True, bDisplay=False, bMinimalSaving=False)
    wall = time.time() - t0
    open(os.path.join(args.out, "babelbrain.log"), "w").write(log.getvalue())

    D = ReadFromH5py(out[0])
    p = np.asarray(D["p_amp"], float)
    hs = float(D["SpatialStep"])
    # BabelBrain stores TargetLocation in the simulation's z orientation but saves the volumes flipped along z
    # (BabelIntegrationBASE.ReturnResults), so the target in the saved volumes is at z' = Nz - 1 - z.
    # Cross-check against the coordinate vectors (target at 0,0,0; z_vec is in the simulation orientation).
    tl = np.asarray(D["TargetLocation"], int).ravel()
    tgt = np.array([tl[0], tl[1], p.shape[2] - 1 - tl[2]])
    from_vec = np.array([np.argmin(np.abs(D["x_vec"])), np.argmin(np.abs(D["y_vec"])),
                         p.shape[2] - 1 - np.argmin(np.abs(D["z_vec"]))])
    if not np.array_equal(tgt, from_vec):
        raise SystemExit(f"target voxel mismatch: TargetLocation gives {tgt.tolist()}, coordinates give {from_vec.tolist()}")
    peak = np.array(np.unravel_index(np.argmax(p), p.shape))
    zone, cen, vol_mm3, ext = half_max_zone(p, peak, hs)
    m = dict(array=os.path.basename(args.array), array_sha256=ea["sha256"], f0_hz=f0, ppw=ppw,
             spatial_step_mm=hs * 1e3, water_only=water, backend=args.backend, device=dev, wall_s=round(wall, 1),
             domain=[ln.split("=")[-1] for ln in log.getvalue().splitlines() if "Final Domain size" in ln][-1:],
             target_voxel=tgt.tolist(), peak_voxel=peak.tolist(),
             peak_offset_mm=((peak - tgt) * hs * 1e3).tolist(), p_peak=float(p.max()), p_target=float(p[tuple(tgt)]),
             halfmax_zone_mm3=vol_mm3, halfmax_centroid_offset_mm=((cen - tgt) * hs * 1e3).tolist(),
             halfmax_extent_mm=ext.tolist(),
             peak_tx_mm=vox_to_tx(D, peak, p.shape).tolist(), halfmax_centroid_tx_mm=vox_to_tx(D, cen, p.shape).tolist(),
             element_source_voxels=[int(v) for v in np.asarray(D["ElementSourceVoxels"])],
             datafile=os.path.relpath(out[0], args.out))
    if args.envelope_axis:
        from scipy.ndimage import uniform_filter1d
        ax = "xyz".index(args.envelope_axis)
        # Odd window, so the moving average is centred (an even window shifts the envelope by half a voxel).
        win = 2 * int(round(1500.0 / f0 / 2 / hs / 2)) + 1
        env = np.sqrt(uniform_filter1d(p ** 2, size=win, axis=ax, mode="nearest"))
        epk = np.array(np.unravel_index(np.argmax(env), env.shape))
        _, ecen, evol, eext = half_max_zone(env, epk, hs)
        m.update(envelope_axis=args.envelope_axis, envelope_window_voxels=win,
                 envelope_peak_tx_mm=vox_to_tx(D, epk, p.shape).tolist(),
                 envelope_halfmax_centroid_tx_mm=vox_to_tx(D, ecen, p.shape).tolist(),
                 envelope_halfmax_zone_mm3=evol, envelope_halfmax_extent_mm=eext.tolist(),
                 envelope_target=float(env[tuple(tgt)]))
    if args.calibrate_pa:
        m["calibration_scale"] = args.calibrate_pa / m["p_target"]
    json.dump(m, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "element_source_voxels"}, indent=1))


if __name__ == "__main__":
    main()
