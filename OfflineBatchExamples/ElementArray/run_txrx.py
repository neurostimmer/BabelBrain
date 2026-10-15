"""Headless transmit-receive (TxRx) runs of BabelBrain's ElementArray transducer.

Simulates an array's own full-matrix-capture style scan: one transmit set fires a tone burst, every element in the
region records its face-averaged pressure from burst onset. The domain (BabelBrain step 2's Step1/Step2: medium,
grid, element voxels) is built once; each transmit set, and each material parameter set, is then one FDTD run in
the same process.

  --mask M             step-1 head model (<ID>_<Tx>_<f>kHz_<PPW>PPW_BabelViscoInput.nii.gz); --use-ct for the CT path
  --water-mask-mm L    water-only case instead (cube of L mm, or x y z), target at its centre
  --region SPEC        elements put in the domain and recorded (default all), e.g. 0-125
  --crop-far-mm D      echo box: drop the head model more than D mm beyond the target, on the side away from the
                       region (the region must lie on one side of the target along the beam axis)
  --transmit SPEC ...  one transmit set per argument; each set fires together, e.g. 10  67  0-125
  --params JSON        list of material parameter sets (default [{}]); keys: bone_att, bone_sos, skin_att,
                       brain_att (multiply BabelBrain's values) and coupling_mm, coupling_att_npm (a lossy
                       water-like layer in the water within coupling_mm of the skin)
Element indices are 0-based. Outputs: <out>/txrx_p<k>_t<j>.npz (traces at the solver's sensor rate and on the
receive grid --fs/--n-samples, t = 0 at burst onset), <out>/txrx_index.json and the BabelBrain log.

Example (echo box for panel A, one transmitter, two parameter sets):
  python run_txrx.py --array a.npz --mask hm/ID_ElementArray_650kHz_6PPW_BabelViscoInput.nii.gz --use-ct \\
      --region 0-125 --crop-far-mm 10 --duration-us 100 --transmit 10 --params '[{}, {"bone_att": 1.5}]' --out o
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

PARAM_KEYS = {"bone_att", "bone_sos", "skin_att", "brain_att", "coupling_mm", "coupling_att_npm"}


def parse_set(spec):
    """'3', '0-125', '1,5,9-12' -> sorted int array."""
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return np.unique(np.array(out, np.int64))


def resample_to_grid(traces, t_src, fs, n):
    """Band-limited interpolation of uniformly sampled traces (t_src[0] = 0) onto t = k/fs, k < n; zero past
    the simulated time. NaN rows stay NaN."""
    from scipy.signal import resample
    dt = t_src[1] - t_src[0]
    m = int(round(n / fs / dt))
    x = np.zeros((traces.shape[0], max(m, traces.shape[1])), np.float64)
    x[:, :traces.shape[1]] = np.nan_to_num(traces)
    y = resample(x[:, :m], n, axis=1) if m != n else x[:, :n]
    y[:, int(np.floor(t_src[-1] * fs)) + 1:] = 0.0
    y[np.isnan(traces).all(1)] = np.nan
    return y.astype(np.float32)


def crop_far(case_dir, prefix, region_z_sign, d_mm):
    """Crop the case's mask, CT and air-region volumes beyond the target on the far side (NIfTI k decreases with
    the simulation's z, see BabelIntegrationELEMENT_ARRAY.GenerateSTLTx). Returns the (old, new) k range."""
    import nibabel
    fm = os.path.join(case_dir, prefix + "BabelViscoInput.nii.gz")
    im = nibabel.load(fm)
    L = np.asarray(im.dataobj)
    kt = int(np.argwhere(L == 5)[0][2])
    h = float(im.header.get_zooms()[2])
    dk = int(np.ceil(d_mm / h))
    nz = L.shape[2]
    lo, hi = (max(0, kt - dk), nz) if region_z_sign < 0 else (0, min(nz, kt + dk + 1))
    for suffix in ("BabelViscoInput.nii.gz", "CT.nii.gz", "AirRegions.nii.gz"):
        f = os.path.join(case_dir, prefix + suffix)
        if not os.path.isfile(f):
            continue
        v = nibabel.load(f)
        A = v.affine.copy()
        A[:3, 3] += A[:3, 2] * lo
        out = nibabel.Nifti1Image(np.asarray(v.dataobj)[:, :, lo:hi], A, v.header)
        out.set_qform(A)
        out.set_sform(A)
        out.to_filename(f)
    return (0, nz), (lo, hi)


def material_rows(use_ct, seg, n_mat):
    """Rows of BabelBrain's material table (Step1_InitializeConditions): 0 water, 1 skin, then
    CT path: 2 brain [3-5 white matter, grey matter, CSF if segmented], one row per bone HU level after that;
    label path: 2 cortical, 3 trabecular, 4 brain [5-7 if segmented]."""
    if use_ct:
        brain = [2] + ([3, 4, 5] if seg else [])
        bone = list(range(brain[-1] + 1, n_mat))
    else:
        bone = [2, 3]
        brain = [4] + ([5, 6, 7] if seg else [])
    return brain, bone


def apply_params(sim, P, brain_rows, bone_rows, log):
    """Scale materials in place and optionally add a coupling layer; returns a restore function."""
    S = sim._SIM_SETTINGS
    mats0 = [list(r) for r in S._Materials]
    map0 = S._MaterialMap.copy() if P.get("coupling_mm", 0) > 0 else None
    q0 = np.array(S._QCorrection, copy=True)
    M = S._Materials
    for r in bone_rows:
        M[r][3] *= P.get("bone_att", 1.0)
        M[r][1] *= P.get("bone_sos", 1.0)
    M[1][3] *= P.get("skin_att", 1.0)
    for r in brain_rows:
        M[r][3] *= P.get("brain_att", 1.0)
    if P.get("coupling_mm", 0) > 0:
        from scipy import ndimage
        water = S._MaterialMap == 0
        dist = ndimage.distance_transform_edt(water) * S._SpatialStep * 1e3
        layer = water & (dist <= P["coupling_mm"])
        if getattr(S, "_SourceMap", None) is not None:
            layer &= S._SourceMap == 0
        w = M[0]
        M.append([w[0], w[1], 0.0, float(P.get("coupling_att_npm", w[3])), 0.0])
        S._MaterialMap[layer] = len(M) - 1
        S._QCorrection = np.r_[np.ravel(q0), 1.0]
        log(f"coupling layer: {int(layer.sum())} voxels within {P['coupling_mm']} mm of tissue, "
            f"{M[-1][3]:.1f} Np/m")

    def restore():
        S._Materials = [list(r) for r in mats0]
        S._QCorrection = q0
        if map0 is not None:
            S._MaterialMap[...] = map0
    return restore


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mask")
    g.add_argument("--water-mask-mm", type=float, nargs="+")
    ap.add_argument("--use-ct", action="store_true")
    ap.add_argument("--water-only", action="store_true", help="head mask geometry, all water (the reference)")
    ap.add_argument("--region", default=None)
    ap.add_argument("--crop-far-mm", type=float, default=None)
    ap.add_argument("--transmit", nargs="+", required=True)
    ap.add_argument("--params", default="[{}]", help="JSON list of parameter sets, or @file.json")
    ap.add_argument("--duration-us", type=float, default=200.0)
    ap.add_argument("--burst-cycles", type=float, default=10.0)
    ap.add_argument("--edge-cycles", type=float, default=1.0)
    ap.add_argument("--sensor-ppp", type=int, default=10)
    ap.add_argument("--fs", type=float, default=27777777.77777778, help="receive grid rate, Hz")
    ap.add_argument("--n-samples", type=int, default=5504, help="receive grid length")
    ap.add_argument("--ppw", type=int, default=None)
    ap.add_argument("--backend", choices=["cuda", "opencl", "cpu"], default="cuda")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        import nibabel
        import TranscranialModeling.BabelIntegrationBASE as B
        from TranscranialModeling.BabelIntegrationELEMENT_ARRAY import BabelFTD_Simulations, LoadElementArray
    sys.path.insert(0, HERE)
    from run_element_array import default_ppw, gpu_name

    ea = LoadElementArray(args.array)
    f0 = ea["f0_hz"]
    ppw = args.ppw or default_ppw(f0)
    h = B.GetSmallestSOS(f0, bShear=True) / f0 / ppw
    params = json.load(open(args.params[1:])) if args.params.startswith("@") else json.loads(args.params)
    for P in params:
        if set(P) - PARAM_KEYS:
            raise SystemExit(f"unknown parameter keys {sorted(set(P) - PARAM_KEYS)}")
    region = parse_set(args.region) if args.region else np.arange(ea["NumberElems"])
    tsets = [parse_set(s) for s in args.transmit]
    for t in tsets:
        if not np.isin(t, region).all():
            raise SystemExit(f"transmit set {t.tolist()} is not inside the region")

    os.makedirs(args.out, exist_ok=True)
    work = os.path.join(args.out, "babelbrain")
    case = os.path.join(work, "case")
    os.makedirs(case, exist_ok=True)
    tag = f"_{int(f0 / 1e3)}kHz_{ppw}PPW_"
    if args.water_mask_mm:
        Ls = args.water_mask_mm * 3 if len(args.water_mask_mm) == 1 else args.water_mask_mm
        Mx, My, Mz = (int(np.ceil(v * 1e-3 / h)) | 1 for v in Ls)
        mask = np.zeros((Mx, My, Mz), np.uint8)
        mask[Mx // 2, My // 2, Mz // 2] = 5
        target = "T"
        nibabel.Nifti1Image(mask, np.diag([h * 1e3] * 3 + [1.0])).to_filename(
            os.path.join(case, target + tag + "BabelViscoInput.nii.gz"))
        water = True
    else:
        name = os.path.basename(args.mask)
        if tag + "BabelViscoInput.nii.gz" not in name:
            raise SystemExit(f"mask name must end with {tag}BabelViscoInput.nii.gz for PPW {ppw}")
        target = name.split(tag)[0]
        src = os.path.dirname(os.path.abspath(args.mask))
        for f in os.listdir(src):
            if f.startswith(target + tag):
                shutil.copy2(os.path.join(src, f), case)
        water = args.water_only
    prefix = target + tag
    crop = None
    if args.crop_far_mm is not None:
        zs = np.sign(ea["centers_m"][region, 2])
        if not (np.all(zs < 0) or np.all(zs > 0)):
            raise SystemExit("--crop-far-mm needs a region on one side of the target along z")
        crop = crop_far(case, prefix, zs[0], args.crop_far_mm)

    backend = {"cuda": 1, "opencl": 2, "cpu": 0}[args.backend]
    dev = gpu_name(args.backend)
    logf = open(os.path.join(args.out, "babelbrain_txrx.log"), "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    use_ct = args.use_ct and not water
    mfn = os.path.join(case, prefix + "BabelViscoInput.nii.gz")
    txrx = dict(region=region, duration_s=args.duration_us * 1e-6, burst_cycles=args.burst_cycles,
                edge_cycles=args.edge_cycles, sensor_ppp=args.sensor_ppp)
    t0 = time.time()
    with contextlib.redirect_stdout(logf):
        if backend == 1:
            B.InitCuda(dev)
        elif backend == 2:
            B.InitOpenCL(dev)
        sim = BabelFTD_Simulations(
            ElementArrayFile=os.path.abspath(args.array), TxRx=txrx, MASKFNAME=mfn, Frequency=f0, basePPW=ppw,
            SensorSubSampling=0, AlphaCFL=0.5, bWaterOnly=water, bDoRefocusing=False, bDisplay=False,
            bTightNarrowBeamDomain=False,
            CTFNAME=os.path.join(case, prefix + "CT.nii.gz") if use_ct else None,
            AIRMASK=(os.path.join(case, prefix + "AirRegions.nii.gz")
                     if use_ct and os.path.isfile(os.path.join(case, prefix + "AirRegions.nii.gz")) else None))
        sim.Step1_InitializeConditions()
        sim.Step2_CalculateRayleighFieldsForward(prefix=prefix, deviceName=dev, bSkipSavingSTL=True)
    S = sim._SIM_SETTINGS
    t_setup = time.time() - t0
    n_mat = len(S._Materials)
    seg = bool((np.asarray(nibabel.load(mfn).dataobj) > 5).any())
    brain_rows, bone_rows = material_rows(use_ct, seg, n_mat) if not water else ([], [])
    log(f"domain {S._N1}x{S._N2}x{S._N3} = {S._N1 * S._N2 * S._N3 / 1e6:.1f} M voxels, h {S._SpatialStep * 1e3:.4f} mm, "
        f"dt {S._TemporalStep * 1e9:.3f} ns, steps {int(round(S._TimeSimulation / S._TemporalStep))}, "
        f"materials {n_mat} (brain rows {brain_rows}, bone rows {len(bone_rows)}), crop {crop}, setup {t_setup:.0f} s")

    index = dict(array=os.path.basename(args.array), mask=None if args.water_mask_mm else os.path.basename(args.mask),
                 water=water, use_ct=use_ct, region=region.tolist(), crop_k=crop, ppw=ppw,
                 domain=[S._N1, S._N2, S._N3], spatial_step_m=S._SpatialStep, dt_s=S._TemporalStep,
                 duration_s=S._TimeSimulation, setup_s=t_setup, runs=[])
    for ip, P in enumerate(params):
        restore = apply_params(sim, P, brain_rows, bone_rows, log) if (P and not water) else (lambda: None)
        for it, tx in enumerate(tsets):
            t1 = time.time()
            with contextlib.redirect_stdout(logf):
                sim.SetTransmit(tx)
                sim.Step3_CreateSourceSignal_and_Sensor()
                sim.Step4_Run_Simulation(GPUName=dev, COMPUTING_BACKEND=backend)
            R = sim.ReturnTxRx()
            tr, t = R["traces"], R["time_s"]
            grid = resample_to_grid(tr[region], t, args.fs, args.n_samples)
            fn = f"txrx_p{ip}_t{it}.npz"
            np.savez_compressed(os.path.join(args.out, fn), traces=tr[region], time_s=t, traces_grid=grid,
                                fs_grid=args.fs, region=region, transmit=tx, params=json.dumps(P),
                                voxels=R["voxels"][region], f0_hz=f0, burst_cycles=args.burst_cycles,
                                edge_cycles=args.edge_cycles)
            wall = time.time() - t1
            index["runs"].append(dict(file=fn, params=P, transmit=tx.tolist(), wall_s=wall))
            log(f"params {ip} {P} transmit {tx.tolist()[:6]}{'...' if tx.size > 6 else ''}: {wall:.1f} s -> {fn}")
            json.dump(index, open(os.path.join(args.out, "txrx_index.json"), "w"), indent=1)
        restore()
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
