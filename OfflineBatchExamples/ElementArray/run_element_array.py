#!/usr/bin/env python3
"""Run BabelBrain's acoustic step for an element-array transducer without the GUI, and report field metrics.

Inputs: an `element_array_v1` definition file (see TranscranialModeling/BabelIntegrationELEMENT_ARRAY.py)
and either
  --water-mask-mm L   a synthetic water-only case: an L-mm cube mask with the target at its centre, or
  --mask FILE         a BabelBrain step-1 mask (<target>_<f>kHz_<PPW>PPW_BabelViscoInput.nii.gz) for a head
                      run, with the CT files step 1 wrote next to it (--use-ct). As BabelBrain's own step 2
                      does, the head run is followed by a water-only run on the same mask, which step 3
                      (thermal) needs; --no-water skips it.
Outputs in --out: BabelBrain's own result files, and metrics.json with:
  peak position and offset from the target (mm), |p| at the target, the connected half-maximum zone
  around the peak (volume, centroid, centroid offset from the target, extents), and the scale that makes
  |p(target)| equal --calibrate-pa (for a calibration run). Head runs add the water run's peak, |p| at
  the target and the head/water ratio there.

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
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "BabelBrain")]


def default_ppw(f0):
    """BabelBrain's GUI default: 9 PPW at or below 350 kHz, 6 above (BabelBrain.py)."""
    return 9 if f0 <= 350e3 else 6


class MemLog(threading.Thread):
    """Every `every` s, write this process's resident memory (VmRSS), its peak so far (VmHWM), the GPU memory in
    use (nvidia-smi, if present) and the last line of the newest BabelBrain log to a TSV, line-buffered, so the
    record survives an out-of-memory kill and the peak can be tied to a pipeline stage."""

    def __init__(self, path, every, logs):
        super().__init__(daemon=True)
        self.path, self.every, self.logs = path, every, logs
        self.halt = threading.Event()
        self.t0 = time.time()
        self.peak_gb = 0.0

    @staticmethod
    def _status():
        st = {}
        for ln in open("/proc/self/status"):
            if ln.startswith(("VmRSS:", "VmHWM:")):
                k, v = ln.split(":", 1)
                st[k] = int(v.split()[0]) * 1024 / 1e9          # kB -> GB
        return st.get("VmRSS", float("nan")), st.get("VmHWM", float("nan"))

    def _gpu_gb(self):
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                               capture_output=True, text=True, timeout=10)
            return float(r.stdout.split()[0]) * 1.048576e-3        # MiB -> GB
        except Exception:
            return float("nan")

    def _stage(self):
        for p in reversed(self.logs):
            try:
                with open(p, "rb") as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 4096))
                    lines = [ln.strip() for ln in f.read().decode(errors="replace").splitlines() if ln.strip()]
                if lines:
                    return f"{os.path.basename(p)}: {lines[-1][:160]}"
            except OSError:
                continue
        return ""

    def sample(self, f):
        rss, hwm = self._status()
        self.peak_gb = max(self.peak_gb, hwm)
        f.write(f"{time.time() - self.t0:.0f}\t{rss:.2f}\t{hwm:.2f}\t{self._gpu_gb():.2f}\t{self._stage()}\n")

    def run(self):
        with open(self.path, "w", buffering=1) as f:
            f.write("t_s\trss_gb\tpeak_rss_gb\tgpu_used_gb\tlast_log_line\n")
            self.sample(f)
            while not self.halt.wait(self.every):
                self.sample(f)
            self.sample(f)

    def stop(self):
        self.halt.set()
        self.join(timeout=30)


def gpu_name(backend):
    if backend == "cuda":
        import cupy as cp
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    if backend == "opencl":
        import pyopencl as cl
        return [d.name for p in cl.get_platforms() for d in p.get_devices()][0]
    return "CPU"


def install_medium_hook(P, case_dir, prefix, use_ct):
    """Apply medium settings (medium_params.apply_params) to every head (not water-only) run, after BabelBrain's Step2
    and before the sources and the solve."""
    import nibabel
    from TranscranialModeling import BabelIntegrationELEMENT_ARRAY as EA
    from medium_params import PARAM_KEYS, apply_params, material_rows
    if set(P) - PARAM_KEYS:
        raise SystemExit(f"unknown medium settings {sorted(set(P) - PARAM_KEYS)}")
    seg = bool((np.asarray(nibabel.load(os.path.join(case_dir, prefix + "BabelViscoInput.nii.gz")).dataobj) > 5).any())
    ct_hu = (np.load(os.path.join(case_dir, prefix + "CT-cal.npz"))["UniqueHU"].astype(np.float32) if use_ct else None)
    orig = EA.BabelFTD_Simulations.Step2_CalculateRayleighFieldsForward

    def step2(self, *a, **k):
        out = orig(self, *a, **k)
        if not self._bWaterOnly:
            brain_rows, bone_rows = material_rows(use_ct, seg, len(self._SIM_SETTINGS._Materials))
            apply_params(self, P, brain_rows, bone_rows, print, ct_hu=ct_hu)
            print(f"medium settings applied: {P}")
        return out
    EA.BabelFTD_Simulations.Step2_CalculateRayleighFieldsForward = step2


def vox_to_tx(D, ijk, shape):
    """Saved-volume voxel (possibly fractional) -> Tx-frame position in mm (target at the origin).
    The saved volumes are flipped along z relative to z_vec."""
    i, j, k = np.asarray(ijk, float)
    xs, ys, zs = (np.asarray(D[v], float) for v in ("x_vec", "y_vec", "z_vec"))
    kz = shape[2] - 1 - k
    return 1e3 * np.array([np.interp(i, np.arange(xs.size), xs), np.interp(j, np.arange(ys.size), ys),
                           np.interp(kz, np.arange(zs.size), zs)])


def offtarget_metrics(D, p, tgt, hs, f0, ea, args):
    """Off-target analysis in the Tx frame (see --offtarget)."""
    from scipy import ndimage
    xs, ys, zs = (1e3 * np.asarray(D[v], float) for v in ("x_vec", "y_vec", "z_vec"))
    zs_saved = zs[::-1]                                   # saved volumes are flipped along z
    X, Y, Z = np.meshgrid(xs, ys, zs_saved, indexing="ij", sparse=True)
    d = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    focal = float(p[d <= args.focal_radius_mm].max())
    fringe = 1500.0 / f0 / 2                              # standing-wave fringe period in water (m)
    size = 2 * int(np.ceil(fringe / hs / 2)) + 1          # odd, at least one fringe
    env = ndimage.maximum_filter(p, size=size)
    lab, _ = ndimage.label(env >= 0.5 * focal)
    main = lab == lab[tuple(tgt)]
    excl = ndimage.binary_dilation(main, iterations=int(round(args.main_dilate_mm / (hs * 1e3))))
    region = np.ones(p.shape, bool)
    if args.region_half_mm:
        coord = {"x": X, "y": Y, "z": Z}[args.region_axis]
        region = np.broadcast_to(np.abs(coord) <= args.region_half_mm, p.shape)
    off = np.where(excl | ~region, 0, p)
    oi = np.unravel_index(off.argmax(), p.shape)
    ext = []
    for ax, v in enumerate((xs, ys, zs_saved)):
        other = tuple(j for j in range(3) if j != ax)
        sel = v[np.any(main, axis=other)]
        ext.append([float(sel.min()), float(sel.max())])
    out = dict(ot_focal=focal, ot_envelope_maxfilter_voxels=size, ot_offtarget_max=float(off.max() / focal),
               ot_offtarget_loc_tx_mm=[float(xs[oi[0]]), float(ys[oi[1]]), float(zs_saved[oi[2]])],
               ot_main_lobe_extent_tx_mm=ext, ot_main_lobe_voxels=int(main.sum()),
               ot_saved_extent_tx_mm=[[float(v.min()), float(v.max())] for v in (xs, ys, zs_saved)])
    faces = {}
    for dd in args.face_offsets_mm:
        pts = (ea["centers_m"] + dd * 1e-3 * ea["normals"]) * 1e3
        ii = [(int(np.argmin(np.abs(xs - q[0]))), int(np.argmin(np.abs(ys - q[1]))),
               int(np.argmin(np.abs(zs_saved - q[2])))) for q in pts]
        inside = [all(lo - hs * 1e3 <= c <= hi + hs * 1e3 for c, (lo, hi) in zip(q, out["ot_saved_extent_tx_mm"]))
                  for q in pts]
        vals = np.array([env[i] for i, ok in zip(ii, inside) if ok]) / focal
        faces[f"{dd:g}mm"] = dict(max=float(vals.max()) if vals.size else None,
                                  mean=float(vals.mean()) if vals.size else None, n_inside=int(sum(inside)))
    out["ot_face_offset_envelope"] = faces
    return out


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
    g.add_argument("--water-mask-mm", type=float, nargs="+",
                   help="water-only case: mask size in mm, one value (cube) or three (x y z, Tx frame)")
    g.add_argument("--mask")
    ap.add_argument("--use-ct", action="store_true", help="head run on step-1 CT outputs (CT/pseudo-CT path)")
    ap.add_argument("--mem-log", type=float, default=0.0, metavar="SECONDS",
                    help="sample this process's resident memory (and GPU memory) every SECONDS into memory.tsv, "
                         "with the last line BabelBrain logged, so the peak can be tied to a stage")
    ap.add_argument("--no-water", action="store_true",
                    help="head runs: skip the water-only run on the same mask (step 3 then can't run)")
    ap.add_argument("--ppw", type=int, default=None, help="default: BabelBrain's GUI default for the frequency")
    ap.add_argument("--backend", choices=["cuda", "opencl", "cpu"], default="cuda")
    ap.add_argument("--params", default=None,
                    help="medium settings for the head run (JSON dict, or @file holding a dict or a one-item list); see "
                         "medium_params.py: bone_att, coupling_mm/_rho/_sos/_att_npm, bone_tex_sd_hu/_ell_mm/_seed, ...")
    ap.add_argument("--calibrate-pa", type=float, default=None)
    ap.add_argument("--offtarget", action="store_true",
                    help="off-target analysis: focal = max |p| within --focal-radius-mm of the target; envelope = "
                         "max filter one standing-wave fringe wide; main lobe = connected envelope >= 0.5 focal "
                         "containing the target, extents per axis; off-target = max |p| outside the main lobe "
                         "dilated by --main-dilate-mm and inside --region-half-mm along --region-axis; and the "
                         "envelope at points --face-offsets-mm in front of each element face")
    ap.add_argument("--focal-radius-mm", type=float, default=1.0)
    ap.add_argument("--main-dilate-mm", type=float, default=4.0)
    ap.add_argument("--region-axis", choices=["x", "y", "z"], default="z")
    ap.add_argument("--region-half-mm", type=float, default=None)
    ap.add_argument("--face-offsets-mm", type=float, nargs="*", default=[])
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
        L = args.water_mask_mm * 3 if len(args.water_mask_mm) == 1 else args.water_mask_mm
        if len(L) != 3:
            raise SystemExit("--water-mask-mm takes one or three values")
        # The mask's z axis is the simulation z flipped; a symmetric box makes that immaterial here.
        Mx, My, Mz = (int(np.ceil(v * 1e-3 / h)) | 1 for v in L)
        mask = np.zeros((Mx, My, Mz), np.uint8)
        mask[Mx // 2, My // 2, Mz // 2] = 5
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
    medium = None
    if args.params:
        medium = json.load(open(args.params[1:])) if args.params.startswith("@") else json.loads(args.params)
        medium = medium[0] if isinstance(medium, list) else medium
        install_medium_hook(medium, case_dir, target + tag, args.use_ct and not water)

    # BabelBrain's output goes straight to log files (line-buffered), so a killed run still leaves its log.
    log_head = os.path.join(args.out, "babelbrain.log")
    log_water = os.path.join(args.out, "babelbrain_water.log")
    mem = MemLog(os.path.join(args.out, "memory.tsv"), args.mem_log, [log_head, log_water]) if args.mem_log else None
    if mem:
        mem.start()
    t0 = time.time()
    with open(log_head, "w", buffering=1) as lf, contextlib.redirect_stdout(lf):
        out = RUN_SIM().RunCases(targets=[target], ID="case", basedir=work + os.sep, Frequencies=[f0],
                                 basePPW=[ppw], bWaterOnly=water, bUseCT=args.use_ct, COMPUTING_BACKEND=backend,
                                 deviceName=dev, ElementArrayFile=os.path.abspath(args.array),
                                 bForceRecalc=True, bDisplay=False, bMinimalSaving=False)
    wall = time.time() - t0
    out_w = None
    if not water and not args.no_water:
        # BabelBrain's own step 2 (CalculateFieldProcess) follows the head run with a water-only run on the same
        # mask. Step 3 (thermal) reads it next to the head result as <...>Water_DataForSim.h5.
        t1 = time.time()
        with open(log_water, "w", buffering=1) as lf, contextlib.redirect_stdout(lf):
            out_w = RUN_SIM().RunCases(targets=[target], ID="case", basedir=work + os.sep, Frequencies=[f0],
                                       basePPW=[ppw], bWaterOnly=True, bUseCT=args.use_ct,
                                       COMPUTING_BACKEND=backend, deviceName=dev,
                                       ElementArrayFile=os.path.abspath(args.array),
                                       bForceRecalc=True, bDisplay=False, bMinimalSaving=False)
        wall_w = time.time() - t1

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
             domain=[ln.split("=")[-1].strip() for ln in open(log_head) if "Final Domain size" in ln][-1:],
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
    if args.offtarget:
        m.update(offtarget_metrics(D, p, tgt, hs, f0, ea, args))
    if out_w:
        W = ReadFromH5py(out_w[0])
        pw = np.asarray(W["p_amp"], float)
        if pw.shape != p.shape:
            raise SystemExit(f"water run grid {pw.shape} differs from the head run's {p.shape}")
        m.update(water_datafile=os.path.relpath(out_w[0], args.out), water_wall_s=round(wall_w, 1),
                 water_p_peak=float(pw.max()), water_p_target=float(pw[tuple(tgt)]),
                 head_over_water_at_target=m["p_target"] / float(pw[tuple(tgt)]))
    if args.calibrate_pa:
        m["calibration_scale"] = args.calibrate_pa / m["p_target"]
    if mem:
        mem.stop()
        m.update(host_peak_rss_gb=mem.peak_gb, memory_log="memory.tsv")
    m["medium_settings"] = medium
    json.dump(m, open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in m.items() if k != "element_source_voxels"}, indent=1))


if __name__ == "__main__":
    main()
