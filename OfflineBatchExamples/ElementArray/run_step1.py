#!/usr/bin/env python3
"""Run BabelBrain's step 1 (domain generation) without the GUI, exactly as the GUI does.

It mirrors BabelBrain.py's mask worker: the grid step comes from the smallest speed of sound (shear
included) and the PPW; the T1 is resampled to 1 mm isotropic (save_T1W_iso); the advanced options take
BabelBrain's defaults (Options.py) unless overridden here; CalculateMaskProcess does the work.

Outputs go to --out/<ID>/ with the GUI's prefix <ID>_<TxSystem>_<f>kHz_<PPW>PPW_, e.g.
<prefix>BabelViscoInput.nii.gz (+ CT.nii.gz, CT-cal.npz, AirRegions.nii.gz on the CT path), which
run_element_array.py --mask <...BabelViscoInput.nii.gz> --use-ct consumes.

Note: step 1 writes meshes and a reuse marker INTO the SimNIBS m2m folder (bone.stl, csf.stl, skin.stl,
final_tissues_babelbrain_reuse.npy). Point --m2m at a copy you own.

Example (pseudo-CT on BabelBrain's CT path, box FOV around a two-sided array):
  python run_step1.py --m2m m2m_S1 --t1 m2m_S1/T1.nii.gz --ct S1_pct.nii --ct-type 1 \\
      --trajectory S1_target.txt --trajectory-type brainsight --id S1 --tx ElementArray \\
      --frequency 650e3 --box-fov 120 210 --out work --backend cuda
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "BabelBrain")]


def save_T1W_iso(T1W_fname, T1WIso_fname, new_spacing=[1.0, 1.0, 1.0]):
    """Verbatim from BabelBrain/BabelBrain.py (save_T1W_iso), copied to avoid importing the Qt GUI module."""
    import SimpleITK as sitk
    preT1 = sitk.ReadImage(T1W_fname)
    getMin = sitk.MinimumMaximumImageFilter()
    getMin.Execute(preT1)
    minval = getMin.GetMinimum()
    original_spacing = preT1.GetSpacing()
    original_size = preT1.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    preT1 = sitk.Resample(preT1, new_size, sitk.Transform(), sitk.sitkNearestNeighbor, preT1.GetOrigin(),
                          new_spacing, preT1.GetDirection(), minval, preT1.GetPixelID())
    sitk.WriteImage(preT1, T1WIso_fname)


def gpu_name(backend):
    if backend == "cuda":
        import cupy as cp
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    import pyopencl as cl
    return [d.name for p in cl.get_platforms() for d in p.get_devices()][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m2m", required=True, help="SimNIBS charm m2m folder (a copy you own: step 1 writes into it)")
    ap.add_argument("--t1", required=True, help="T1W the trajectory is defined in (charm's input T1)")
    ap.add_argument("--ct", default=None, help="CT / pseudo-CT / ZTE / PETRA (omit for T1W-only)")
    ap.add_argument("--ct-type", type=int, default=1, help="BabelBrain CTType: 1 CT, 2 ZTE, 3 PETRA, 4 density")
    ap.add_argument("--coreg-ct", type=int, default=0, help="BabelBrain CoregCT_MRI: 0 none (same world), 1 CT->MRI")
    ap.add_argument("--hu-threshold", type=float, default=300.0, help="GUI default 300")
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--trajectory-type", choices=["brainsight", "slicer"], default="brainsight")
    ap.add_argument("--id", required=True)
    ap.add_argument("--tx", default="ElementArray")
    ap.add_argument("--frequency", type=float, default=650e3)
    ap.add_argument("--ppw", type=int, default=None, help="default: GUI default (9 at <=350 kHz, else 6)")
    ap.add_argument("--box-fov", type=float, nargs=2, metavar=("DIAMETER_MM", "LENGTH_MM"), default=None,
                    help="BabelBrain's box FOV (bApplyBOXFOV) centred on the target, aligned with the trajectory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["cuda", "opencl"], default="cuda")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        from TranscranialModeling.BabelIntegrationBASE import GetSmallestSOS
    import numpy as np
    f0 = args.frequency
    ppw = args.ppw or (9 if f0 <= 350e3 else 6)
    spatial_step = np.round(GetSmallestSOS(f0, bShear=True) / f0 / ppw * 1e3, 3)        # as BabelBrain.py
    case = os.path.join(args.out, args.id)
    os.makedirs(case, exist_ok=True)
    t1 = os.path.join(case, os.path.basename(args.t1))
    if os.path.abspath(args.t1) != os.path.abspath(t1):
        shutil.copy2(args.t1, t1)
    t1_iso = os.path.join(case, "T1W_iso.nii.gz")
    save_T1W_iso(t1, t1_iso)
    prefix = f"{args.id}_{args.tx}_{int(f0 / 1e3)}kHz_{ppw}PPW_"

    kargs = dict(SimbNIBSDir=os.path.abspath(args.m2m), SimbNIBSType="charm", CoregCT_MRI=args.coreg_ct,
                 TrajectoryType=args.trajectory_type, Mat4Trajectory=os.path.abspath(args.trajectory),
                 T1Source_nii=t1, T1Conformal_nii=t1_iso, SpatialStep=float(spatial_step), Location=[0, 0, 0],
                 prefix=prefix, bPlot=False)
    if args.ct:
        kargs.update(CT_or_ZTE_input=os.path.abspath(args.ct), CTType=args.ct_type, HUThreshold=args.hu_threshold)
    # BabelBrain's advanced-option defaults that step 1 uses (Options.py), then our overrides.
    defaults = dict(bApplyBOXFOV=False, FOVDiameter=200.0, FOVLength=400.0, bForceUseBlender=False,
                    ElastixOptimizer="AdaptiveStochasticGradientDescent", TrabecularProportion=0.8,
                    PetraNPeaks=2, PetraMRIPeakDistance=50, bInvertZTE=False, bExtractAirRegions=True,
                    bDisableCTMedianFilter=False, bGeneratePETRAHistogram=False, PETRASlope=-2080.0,
                    PETRAOffset=2133.2, ZTESlope=-2085.0, ZTEOffset=2329.0, bSegmentBrainTissue=False,
                    SimbNINBSRoot="...")
    kargs.update(defaults)
    if args.box_fov:
        kargs.update(bApplyBOXFOV=True, FOVDiameter=args.box_fov[0], FOVLength=args.box_fov[1])
    kargs["bForceFullRecalculation"] = True

    from CalculateMaskProcess import CalculateMaskProcess
    backend = {"cuda": 1, "opencl": 2}[args.backend]
    dev = gpu_name(args.backend)
    q = mp.Queue()
    t0 = time.time()
    p = mp.Process(target=CalculateMaskProcess, args=(q, backend, dev), kwargs=kargs)
    p.start()
    log, outputs = [], None
    while p.is_alive() or not q.empty():
        while not q.empty():
            m = q.get()
            if isinstance(m, str):
                log.append(m)
            else:
                outputs = m
        time.sleep(0.2)
    p.join()
    wall = time.time() - t0
    open(os.path.join(case, prefix + "step1.log"), "w").write("".join(log))
    ok = p.exitcode == 0 and not any("--Babel-Brain-Low-Error" in s for s in log)
    mask = os.path.join(case, prefix + "BabelViscoInput.nii.gz")
    summary = dict(ok=bool(ok and os.path.isfile(mask)), wall_s=round(wall, 1), prefix=prefix, mask=mask,
                   spatial_step_mm=float(spatial_step), ppw=ppw, device=dev,
                   kargs={k: (v if isinstance(v, (int, float, str, bool, list, type(None))) else str(v))
                          for k, v in kargs.items()},
                   outputs=outputs if isinstance(outputs, (dict, list)) else str(outputs))
    if summary["ok"]:
        import nibabel
        im = nibabel.load(mask)
        d = np.asarray(im.dataobj)
        summary.update(mask_shape=list(d.shape), mask_labels={int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))},
                       focal_voxel=np.argwhere(d == 5).ravel().tolist())
    json.dump(summary, open(os.path.join(case, prefix + "step1.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "kargs"}, indent=1))
    if not summary["ok"]:
        print("".join(log[-60:]))
        sys.exit(1)


if __name__ == "__main__":
    main()
