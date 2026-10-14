#!/usr/bin/env python3
"""Run BabelBrain's step 3 (thermal) without the GUI on a step-2 result, and report temperature metrics.

Inputs:
  --step2 DIR    a run_element_array.py output directory: its metrics.json names the head result
                 <...>_DataForSim.h5. BabelBrain also reads the water-only result <...>_Water_DataForSim.h5 next
                 to it, which run_element_array.py writes for head runs (as BabelBrain's own step 2 does).
  --profile YAML a thermal profile in BabelBrain's format (Profiles/*.yaml: BaseIsppa, AllDC_PRF_Duration,
                 optional bConcatenateSimulations), with the GUI's defaults for optional keys.

Intensity: BabelBrain scales the field so that the peak Isppa in the brain equals the requested Isppa
(CalculateTemperatureEffects.AnalyzeLosses).
  --isppa drive    (default) the Isppa the field already has, so the drive is simulated as sent. It mirrors
                   BabelBrain's normalisation: peak |p| over brain voxels, rho*c at TargetLocation (or at the
                   brain peak when the brain is segmented). BabelBrain then reports PressureRatio = 1, which
                   is checked and recorded.
  --isppa profile  the profile's BaseIsppa, as the GUI does.
  --isppa X        X W/cm2.

Runs CalculateThermalProcess in a worker process with the GUI's arguments (Babel_Thermal.RunThermalSim) and
the Options.py defaults (baseline 37 C, 100 BHTE iterations per process). Writes in --out:
  thermal.log           everything BabelBrain printed;
  thermal_metrics.json  per timing entry: peak temperature and rise in brain, skin and skull, CEM43 (min) in
                        each, the target's peak temperature, Isppa, MaxBrainPressure, MaxIsppa, MaxIspta,
                        PressureRatio, RatioLosses and the result file.

Example:
  python run_thermal.py --step2 runs/head/out --profile thermal.yaml --out runs/head/thermal --backend cuda
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "BabelBrain")]

# Babel_Thermal.ThermalProfileConfig defaults
PROFILE_DEFAULTS = dict(bConcatenateSimulations=False)
ENTRY_DEFAULTS = dict(Repetitions=1, NumberGroupedSonications=1, PauseBetweenGroupedSonications=0.0)


def device_name(backend):
    if backend == "cuda":
        import cupy as cp
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    import pyopencl as cl
    return [d.name for p in cl.get_platforms() for d in p.get_devices()][0]


def drive_isppa(fname):
    """Isppa (W/cm2) that BabelBrain's normalisation assigns to the field as it is (PressureRatio = 1).
    Mirrors CalculateTemperatureEffects: volumes flipped along z; brain is material 2 on the CT path (2-5 with a
    segmented brain), otherwise 4 (4-7); rho*c at TargetLocation, used as-is on the flipped maps."""
    from BabelViscoFDTD.H5pySimple import ReadFromH5py
    D = ReadFromH5py(fname)
    p = np.flip(np.asarray(D["p_amp"]), axis=2)
    orig = np.flip(np.asarray(D["MaterialMap"]), axis=2)
    ct = "MaterialMapCT" in D
    mm = np.flip(np.asarray(D["MaterialMapCT"]), axis=2) if ct else orig
    seg = bool(np.any(orig > 5))
    brain = ([2, 3, 4, 5] if seg else [2]) if ct else ([4, 5, 6, 7] if seg else [4])
    pb = np.where(np.isin(mm, brain), p, 0.0)
    if seg:
        loc = np.unravel_index(np.argmax(pb), pb.shape)
    else:
        loc = tuple(int(v) for v in np.asarray(D["TargetLocation"]).ravel())
    mat = np.asarray(D["Material"])
    rho, c = float(mat[mm[loc], 0]), float(mat[mm[loc], 1])
    pmax = float(pb.max())
    return pmax ** 2 / (2 * rho * c) / 1e4, dict(p_brain_max_pa=pmax, rho=rho, c=c, rho_c_voxel=[int(v) for v in loc],
                                                ct_path=ct, segmented_brain=seg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--step2", help="run_element_array.py output directory (head run)")
    g.add_argument("--data", help="head result <...>_DataForSim.h5 directly")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--isppa", default="drive", help="'drive' (default), 'profile', or W/cm2")
    ap.add_argument("--frequency", type=float, default=None, help="Hz; default: f0_hz from the step-2 metrics.json")
    ap.add_argument("--baseline", type=float, default=37.0, help="GUI default 37 C")
    ap.add_argument("--bhte-per-process", type=int, default=100, help="GUI default 100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["cuda", "opencl"], default="cuda")
    args = ap.parse_args()

    import yaml
    from BabelViscoFDTD.H5pySimple import ReadFromH5py
    from ThermalModeling.CalculateTemperatureEffects import GetThermalOutName
    from Babel_Thermal.CalculateThermalProcess import CalculateThermalProcess

    f0 = args.frequency
    if args.step2:
        m2 = json.load(open(os.path.join(args.step2, "metrics.json")))
        data = os.path.join(args.step2, m2["datafile"])
        f0 = f0 or m2["f0_hz"]
    else:
        data = args.data
    if not f0:
        raise SystemExit("--frequency is required with --data")
    data = os.path.abspath(data)
    water = data.replace("DataForSim.h5", "Water_DataForSim.h5")
    for f in (data, water):
        if not os.path.isfile(f):
            raise SystemExit(f"missing {f} (step 3 needs the head result and its water-only companion)")

    prof = yaml.safe_load(open(args.profile)) or {}
    for k, v in PROFILE_DEFAULTS.items():
        prof.setdefault(k, v)
    entries = prof["AllDC_PRF_Duration"]
    for e in entries:
        for k, v in ENTRY_DEFAULTS.items():
            e.setdefault(k, v)

    info = {}
    if args.isppa == "drive":
        isppa, info = drive_isppa(data)
    elif args.isppa == "profile":
        isppa = float(prof["BaseIsppa"])
    else:
        isppa = float(args.isppa)

    os.makedirs(args.out, exist_ok=True)
    dev = device_name(args.backend)
    kargs = dict(deviceName=dev, COMPUTING_BACKEND={"cuda": 1, "opencl": 2}[args.backend], Isppa=isppa,
                 Frequency=f0, BaselineTemperature=args.baseline,
                 LimitBHTEIterationsPerProcess=args.bhte_per_process, bForceHomogenousMedium=False,
                 bForceNoAbsorptionSkullScalp=False, bConcatenateSimulations=bool(prof["bConcatenateSimulations"]),
                 TxSystem="ElementArray", sel_p="p_amp")
    print(f"[thermal] {data}\n[thermal] Isppa {isppa:.4g} W/cm2 ({args.isppa}); {len(entries)} timing entries; "
          f"device {dev}", flush=True)

    # A fresh interpreter for the worker: the device query above has initialised the GPU runtime in this process,
    # and CUDA and PoCL don't survive a fork (the worker deadlocks or fails to initialise).
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    t0 = time.time()
    proc = ctx.Process(target=CalculateThermalProcess, args=(q, data, entries, {}), kwargs=kargs)
    proc.start()
    log, failed = [], False
    while proc.is_alive() or not q.empty():
        while not q.empty():
            msg = q.get()
            log.append(msg)
            print(msg, end="", flush=True)
            failed |= "--Babel-Brain-Low-Error" in msg
        time.sleep(0.2)
    proc.join()
    wall = time.time() - t0
    open(os.path.join(args.out, "thermal.log"), "w").write("".join(log))
    if failed or proc.exitcode != 0:
        raise SystemExit(f"[thermal] FAILED (exit code {proc.exitcode}); see {args.out}/thermal.log")

    results = []
    for e in entries:
        fname = GetThermalOutName(data, e["Duration"], e["DurationOff"], e["DC"], isppa, e["PRF"],
                                  e["Repetitions"]) + ".h5"
        R = ReadFromH5py(fname)
        g = lambda k: float(np.asarray(R[k]).ravel()[0])
        base = g("BaselineTemperature")
        # BabelBrain saves TI/TIS/TIC as rises above baseline, taken from the end-of-sonication map TempEndFUS.
        # The target is read from the same map at TargetLocation, not from TempProfileTarget: when the target is
        # also the brain's hottest voxel only three points are monitored, and IndTarget = 2 then selects the
        # skull's trace (seen on a synthetic case: the "target" trace equalled the skull peak).
        tl = tuple(int(v) for v in np.asarray(R["TargetLocation"]).ravel())
        rise = dict(brain=g("TI"), skin=g("TIS"), skull=g("TIC"), target=float(np.asarray(R["TempEndFUS"])[tl]) - base)
        r = dict(entry=e, result_file=os.path.relpath(fname, args.out), baseline_c=base, rise_c=rise,
                 peak_c={k: v + base for k, v in rise.items()},
                 cem43_min=dict(brain=g("CEMBrain"), skin=g("CEMSkin"), skull=g("CEMSkull")),
                 isppa_w_cm2=g("Isppa"), max_brain_pressure_pa=g("MaxBrainPressure"),
                 max_isppa_w_cm2=g("MaxIsppa"), max_ispta_w_cm2=g("MaxIspta"),
                 pressure_ratio=float(np.ravel(R["PressureRatio"])[0]), ratio_losses=g("RatioLosses"),
                 simulated_s=float(np.ravel(R["TimeProfileTarget"])[-1]))
        results.append(r)
    out = dict(data=os.path.relpath(data, args.out), profile=os.path.abspath(args.profile), isppa_mode=args.isppa,
               isppa_w_cm2=isppa, drive_isppa=info, frequency_hz=f0, backend=args.backend, device=dev,
               wall_s=round(wall, 1), results=results)
    if args.isppa == "drive":
        out["pressure_ratio_check"] = all(abs(r["pressure_ratio"] - 1) < 1e-3 for r in results)
    json.dump(out, open(os.path.join(args.out, "thermal_metrics.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    for r in results:
        print(json.dumps({k: r[k] for k in ("peak_c", "rise_c", "cem43_min", "pressure_ratio", "simulated_s")}))


if __name__ == "__main__":
    main()
