"""Medium settings shared by the ElementArray drivers (run_txrx.py, run_element_array.py), applied to a BabelBrain
simulation after its domain is built (Step1/Step2) and before the solve:
  bone_att, bone_sos, skin_att, brain_att   multiply BabelBrain's values for those materials
  gel_rho, gel_sos, gel_att_npm             replace the water's density, speed and attenuation everywhere outside the
                                            tissue (the coupling gel filling the gap); head runs only
  coupling_mm, coupling_rho, coupling_sos, coupling_att_npm
                                            a layer replacing the water (or gel) within coupling_mm of the tissue
  bone_tex_sd_hu, bone_tex_ell_mm, bone_tex_seed
                                            CT path: a seeded Gaussian random HU field on the bone (new material rows
                                            from BabelBrain's own CT mappings)
  outside_air, gel_margin_mm                air around the head (BabelViscoFDTD's pressure-release ReflectorMask,
                                            as BabelBrain's air regions) except a gel column between each panel and
                                            the scalp: the panel's face footprint widened by gel_margin_mm (default 10)
"""
import numpy as np

PARAM_KEYS = {"bone_att", "bone_sos", "skin_att", "brain_att", "gel_rho", "gel_sos", "gel_att_npm", "coupling_mm",
              "coupling_rho", "coupling_sos", "coupling_att_npm", "bone_tex_sd_hu", "bone_tex_ell_mm", "bone_tex_seed",
              "outside_air", "gel_margin_mm"}


def hash_normal(i, j, k, seed):
    """Standard normal per integer voxel offset (i, j, k), deterministic in the seed (splitmix64 + Box-Muller)."""
    def mix(x):
        x = (x + np.uint64(0x9E3779B97F4A7C15))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))
    off = np.int64(1 << 20)
    key = (((i + off).astype(np.uint64) << np.uint64(42)) | ((j + off).astype(np.uint64) << np.uint64(21))
           | (k + off).astype(np.uint64)) ^ np.uint64((int(seed) * 0x2545F4914F6CDD1D) % (1 << 64))
    with np.errstate(over="ignore"):
        a, b = mix(key), mix(key ^ np.uint64(0xD1B54A32D192ED03))
    u1 = ((a >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    u2 = ((b >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)
    return (np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)).astype(np.float32)


def bone_texture(S, P, ct_hu, first_bone, q_bone, log):
    """Replace every CT bone voxel's material by a textured HU level (new rows); returns nothing (in place)."""
    import TranscranialModeling.BabelIntegrationBASE as B
    from scipy import ndimage
    mm = S._MaterialMap
    bone = mm >= first_bone
    idx = np.nonzero(bone)
    lo, hi = np.min(idx, axis=1), np.max(idx, axis=1) + 1
    h = S._SpatialStep * 1e3
    ell, sd, seed = float(P["bone_tex_ell_mm"]), float(P["bone_tex_sd_hu"]), int(P.get("bone_tex_seed", 1))
    pad = int(np.ceil(3 * ell / h))
    lo, hi = np.maximum(lo - pad, 0), np.minimum(hi + pad, np.array(mm.shape))
    f = np.asarray(S._FocalSpotLocation, np.int64)
    box = np.empty(tuple(hi - lo), np.float32)
    ii, jj = np.meshgrid(np.arange(lo[0], hi[0]) - f[0], np.arange(lo[1], hi[1]) - f[1], indexing="ij")
    for kk in range(lo[2], hi[2]):
        box[:, :, kk - lo[2]] = hash_normal(ii, jj, np.full_like(ii, kk - f[2]), seed)
    ndimage.gaussian_filter(box, sigma=ell / (np.sqrt(2) * h), output=box, mode="reflect")
    inb = bone[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    box *= sd / float(np.std(box[inb]))
    hu = ct_hu[mm[bone] - first_bone].astype(np.float32) + box[inb]
    c_max = max(r[1] for r in S._Materials)
    grid = np.arange(-200.0, 4001.0, 5.0)
    c_grid = B.HUtoLongSpeedofSoundWebb(grid, params=S._CTMapCombo if hasattr(S, "_CTMapCombo") else ('GE', '120', 'B', '', '0.5, 0.6'))
    ok = (c_grid <= 1.3 * c_max) & (c_grid >= 1200.0)
    hu_lo, hu_hi = float(grid[ok].min()), float(grid[ok].max())
    clipped = float(np.mean((hu < hu_lo) | (hu > hu_hi)))
    hu = np.clip(hu, hu_lo, hu_hi)
    levels = np.arange(hu_lo, hu_hi + 2.5, 5.0)
    combo = getattr(S, "_CTMapCombo", ('GE', '120', 'B', '', '0.5, 0.6'))
    rho = B.HUtoDensityMarsac(levels)
    c = B.HUtoLongSpeedofSoundWebb(levels, params=combo)
    att = B.HUtoAttenuationWebb(levels, S._Frequency, params=combo) * P.get("bone_att", 1.0)
    base = len(S._Materials)
    for r_, c_, a_ in zip(rho, c, att):
        S._Materials.append([float(r_), float(c_), 0.0, float(a_), 0.0])
    S._QCorrection = np.r_[np.ravel(S._QCorrection), np.full(len(levels), q_bone)]
    mm[bone] = (base + np.rint((hu - hu_lo) / 5.0)).astype(mm.dtype)
    log(f"bone texture: sd {sd:.0f} HU, ell {ell} mm, seed {seed}: {bone.sum():,} bone voxels; achieved sd "
        f"{np.std(box[inb]):.0f} HU; HU clipped to {hu_lo:.0f}..{hu_hi:.0f} ({clipped * 100:.2f} % of voxels); "
        f"{len(levels)} new rows, speed {c.min():.0f}..{c.max():.0f} m/s (time step set for {c_max:.0f})")


def outside_air(S, P, log, out=None):
    """Mark the water outside the head as air (pressure release), except a gel column between each panel and the
    scalp: the transverse footprint of the panel's element faces, widened by gel_margin_mm, from the domain edge to
    the first tissue along the panel axis. Water not connected to the domain edge (pockets inside the head) is left
    alone. Sets S._SubAirRegions (the ReflectorMask BabelBrain passes to the solver); returns the previous mask."""
    from scipy import ndimage
    from TranscranialModeling.BabelIntegrationELEMENT_ARRAY import ElementFacePoints
    ea, h, mm = S._EA, S._SpatialStep, S._MaterialMap
    orig = np.array([S._XDim[0], S._YDim[0], S._ZDim[0]])
    shift = np.array([S._TxMechanicalAdjustmentX, S._TxMechanicalAdjustmentY, S._TxMechanicalAdjustmentZ])
    cen = (ea["centers_m"] + shift - orig) / h
    a = int(np.argmax(np.ptp(cen, axis=0)))             # panel axis: the grid axis the panels are separated along
    mid = (cen[:, a].min() + cen[:, a].max()) / 2
    sides = [np.flatnonzero(cen[:, a] < mid), np.flatnonzero(cen[:, a] >= mid)]
    d = cen[sides[1]].mean(0) - cen[sides[0]].mean(0)
    b, c = [k for k in range(3) if k != a]
    tissue = np.moveaxis(mm != 0, a, -1)                # (n_b, n_c, n_a)
    na = tissue.shape[-1]
    has = tissue.any(-1)
    first = np.where(has, tissue.argmax(-1), na)
    last = np.where(has, na - 1 - tissue[..., ::-1].argmax(-1), -1)
    del tissue
    margin = float(P.get("gel_margin_mm", 10.0))
    k = np.arange(na)
    gel = np.zeros(has.shape + (na,), bool)
    faces = np.zeros(mm.shape, bool)
    for s, els in enumerate(sides):
        ijk = np.concatenate([np.round((ElementFacePoints(ea, n, h / 3.0)[0] + shift - orig) / h).astype(np.int64)
                              for n in els])
        faces[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
        foot = np.zeros(has.shape, bool)
        foot[ijk[:, b], ijk[:, c]] = True
        foot = ndimage.distance_transform_edt(~foot) * h * 1e3 <= margin
        n_open = int((foot & ~has).sum())
        foot &= has
        gel |= foot[..., None] & ((k[None, None, :] < first[..., None]) if s == 0 else (k[None, None, :] > last[..., None]))
        log(f"outside air: panel {s} ({els.size} elements) gel footprint {foot.sum() * (h * 1e3) ** 2 / 100:.0f} cm2 "
            f"(+{margin:g} mm), {n_open} footprint columns with no tissue left as air")
    gel = np.moveaxis(gel, -1, a)
    lab, _ = ndimage.label(mm == 0)
    edge = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                     lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    air = np.isin(lab, edge[edge > 0])
    del lab
    n_out = int(air.sum())
    air &= ~gel
    air &= ~ndimage.binary_dilation(faces, iterations=3)
    prev = getattr(S, "_SubAirRegions", None)
    new = air if prev is None else (np.asarray(prev) != 0) | air
    S._SubAirRegions = new.astype(np.uint32)
    log(f"outside air: panel axis = grid axis {a} (panel-to-panel vector {np.degrees(np.arccos(abs(d[a]) / np.linalg.norm(d))):.2f} "
        f"deg off it); outside water {n_out:,} voxels, gel {int((gel & (mm == 0)).sum()):,}, air {int(air.sum()):,} "
        f"({100 * air.sum() / max(n_out, 1):.1f} % of the outside water){'; existing air regions kept' if prev is not None else ''}")
    if out:
        f = np.clip(np.asarray(S._FocalSpotLocation, np.int64), 0, np.array(mm.shape) - 1)
        lab3 = np.where(S._SubAirRegions != 0, 2, (mm != 0)).astype(np.uint8)
        lab3[faces] = 3
        import os
        np.savez_compressed(os.path.join(out, "outside_air_slices.npz"), axis=a, h_m=h, focus=f,
                            s0=lab3[f[0], :, :], s1=lab3[:, f[1], :], s2=lab3[:, :, f[2]],
                            legend="0 water/gel, 1 tissue, 2 air (pressure release), 3 element faces")
    return prev


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


def apply_params(sim, P, brain_rows, bone_rows, log, ct_hu=None, out=None):
    """Scale materials in place, optionally put air around the head, texture the bone and add a coupling layer;
    returns a restore function."""
    S = sim._SIM_SETTINGS
    mats0 = [list(r) for r in S._Materials]
    tex = P.get("bone_tex_sd_hu", 0) > 0
    if tex and (ct_hu is None or not bone_rows):
        raise SystemExit("bone texture needs the CT path (--use-ct)")
    map0 = S._MaterialMap.copy() if (P.get("coupling_mm", 0) > 0 or tex) else None
    q0 = np.array(S._QCorrection, copy=True)
    M = S._Materials
    for r in bone_rows:
        M[r][3] *= P.get("bone_att", 1.0)
        M[r][1] *= P.get("bone_sos", 1.0)
    M[1][3] *= P.get("skin_att", 1.0)
    for r in brain_rows:
        M[r][3] *= P.get("brain_att", 1.0)
    if any(k in P for k in ("gel_rho", "gel_sos", "gel_att_npm")):
        if P.get("gel_sos", M[0][1]) > max(r[1] for r in M):
            raise SystemExit(f"gel_sos {P['gel_sos']} exceeds the fastest material: the time step was set without it")
        M[0][0] = float(P.get("gel_rho", M[0][0]))
        M[0][1] = float(P.get("gel_sos", M[0][1]))
        M[0][3] = float(P.get("gel_att_npm", M[0][3]))
        log(f"gel (all water outside the tissue): rho {M[0][0]:.0f} kg/m3, c {M[0][1]:.0f} m/s, {M[0][3]:.2f} Np/m")
    air0, air_set = None, False
    if P.get("outside_air"):                           # on the original map: before any coupling layer
        air0, air_set = outside_air(S, P, log, out), True
    if tex:
        bone_texture(S, P, ct_hu, bone_rows[0], float(np.ravel(q0)[bone_rows[0]]), log)
    if P.get("coupling_mm", 0) > 0:
        from scipy import ndimage
        water = S._MaterialMap == 0
        dist = ndimage.distance_transform_edt(water) * S._SpatialStep * 1e3
        layer = water & (dist <= P["coupling_mm"])
        if getattr(S, "_SourceMap", None) is not None:
            layer &= S._SourceMap == 0
        w = M[0]
        c_max = max(r[1] for r in M)
        if P.get("coupling_sos", w[1]) > c_max:
            raise SystemExit(f"coupling_sos {P['coupling_sos']} exceeds the fastest material ({c_max} m/s): the "
                             "time step was set without it")
        M.append([float(P.get("coupling_rho", w[0])), float(P.get("coupling_sos", w[1])), 0.0,
                  float(P.get("coupling_att_npm", w[3])), 0.0])
        S._MaterialMap[layer] = len(M) - 1
        S._QCorrection = np.r_[np.ravel(S._QCorrection), 1.0]     # append: texture rows may precede this one
        log(f"coupling layer: {int(layer.sum())} voxels within {P['coupling_mm']} mm of tissue: rho {M[-1][0]:.0f} "
            f"kg/m3, c {M[-1][1]:.0f} m/s (Z {M[-1][0] * M[-1][1] / 1e6:.3f} MRayl), {M[-1][3]:.1f} Np/m")

    def restore():
        S._Materials = [list(r) for r in mats0]
        S._QCorrection = q0
        if map0 is not None:
            S._MaterialMap[...] = map0
        if air_set:
            S._SubAirRegions = air0
    return restore
