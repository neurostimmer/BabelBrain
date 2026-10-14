'''
Generic "element array" transducer for BabelBrain.

Any number of elements at arbitrary positions and orientations (square or circular faces), each driven with its
own complex amplitude, all read from a definition file (format `element_array_v1`, see below). Every element
is voxelised directly into the simulation grid as its own stress source, so elements may surround the target
(e.g. arrays on opposite sides of the head). The drive file carries any steering, correction or refocusing, so
this class does no Rayleigh pre-step, no built-in steering and no refocusing.

Definition file (`.npz`), all arrays in the Tx frame: metres, origin at the target (BabelBrain focal point),
z along BabelBrain's beam axis (the transducer side is z < 0, as for other transducers):
    format       'element_array_v1'
    centers_m    (N,3) element face centres
    normals      (N,3) unit normals pointing into the medium
    edge1        (N,3) unit vector along the first edge of each square face (unused for circles)
    size_m       (N,2) square: edge lengths along edge1 and normal x edge1; circle: (diameter, diameter)
    shape        'square' or 'circle' (one shape for all elements)
    drive        (N,) complex: a_n * exp(-1j * 2*pi*f0 * tau_n); tau > 0 means element n fires later
    f0_hz        drive frequency (must match the simulation frequency)
    pressure_scale_pa (optional) multiplies every source amplitude (calibration); default 1
Element n's source is  s * |drive_n| * sin(2*pi*f0*t + angle(drive_n)), ramped up over the first cycles.
'''
import hashlib
import os

import numpy as np
from stl import mesh

from .BabelIntegrationBASE import (RUN_SIM_BASE,
                                   BabelFTD_Simulations_BASE,
                                   SimulationConditionsBASE)

ELEMENT_ARRAY_FORMAT = 'element_array_v1'


def LoadElementArray(fname):
    '''Read and validate an element_array_v1 definition file. Returns a dict of arrays.'''
    d = np.load(fname, allow_pickle=False)
    fmt = str(d['format']) if 'format' in d else ''
    if fmt != ELEMENT_ARRAY_FORMAT:
        raise ValueError(f'{fname}: format {fmt!r} is not {ELEMENT_ARRAY_FORMAT!r}')
    out = dict(centers_m=np.asarray(d['centers_m'], np.float64),
               normals=np.asarray(d['normals'], np.float64),
               size_m=np.asarray(d['size_m'], np.float64),
               shape=str(d['shape']),
               drive=np.asarray(d['drive'], np.complex128),
               f0_hz=float(d['f0_hz']),
               pressure_scale_pa=float(d['pressure_scale_pa']) if 'pressure_scale_pa' in d else 1.0)
    n = out['centers_m'].shape[0]
    if out['shape'] not in ('square', 'circle'):
        raise ValueError(f'{fname}: shape must be square or circle')
    out['edge1'] = (np.asarray(d['edge1'], np.float64) if 'edge1' in d
                    else np.zeros((n, 3)))
    for k, shp in (('centers_m', (n, 3)), ('normals', (n, 3)), ('size_m', (n, 2)), ('drive', (n,)), ('edge1', (n, 3))):
        if out[k].shape != shp:
            raise ValueError(f'{fname}: {k} has shape {out[k].shape}, expected {shp}')
    out['normals'] /= np.linalg.norm(out['normals'], axis=1)[:, None]
    if out['shape'] == 'square':
        e1 = out['edge1'] - (out['edge1'] * out['normals']).sum(1)[:, None] * out['normals']
        out['edge1'] = e1 / np.linalg.norm(e1, axis=1)[:, None]
    out['NumberElems'] = n
    out['sha256'] = hashlib.sha256(open(fname, 'rb').read()).hexdigest()
    return out


def ElementFacePoints(ea, n, spacing):
    '''Cell-centred sample points (M,3) over element n's face, at most `spacing` apart, and its corners (4,3).'''
    c, nrm = ea['centers_m'][n], ea['normals'][n]
    if ea['shape'] == 'square':
        u1 = ea['edge1'][n]
    else:
        ref = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u1 = np.cross(nrm, ref)
        u1 /= np.linalg.norm(u1)
    u2 = np.cross(nrm, u1)
    w1, w2 = ea['size_m'][n]
    n1, n2 = int(np.ceil(w1 / spacing)), int(np.ceil(w2 / spacing))
    a = ((np.arange(n1) + 0.5) / n1 - 0.5) * w1
    b = ((np.arange(n2) + 0.5) / n2 - 0.5) * w2
    A, B = np.meshgrid(a, b, indexing='ij')
    if ea['shape'] == 'circle':
        keep = (A / (w1 / 2)) ** 2 + (B / (w2 / 2)) ** 2 <= 1.0
        A, B = A[keep], B[keep]
    pts = c[None, :] + A.reshape(-1, 1) * u1[None, :] + B.reshape(-1, 1) * u2[None, :]
    corners = np.array([c + sa * w1 / 2 * u1 + sb * w2 / 2 * u2 for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return pts, corners


class RUN_SIM(RUN_SIM_BASE):
    def CreateSimObject(self, **kargs):
        return BabelFTD_Simulations(ElementArrayFile=self._ElementArrayFile, **kargs)

    def RunCases(self, ElementArrayFile='', bDoRefocusing=False, bTightNarrowBeamDomain=False, **kargs):
        if not os.path.isfile(ElementArrayFile):
            raise FileNotFoundError(f'ElementArrayFile not found: {ElementArrayFile!r}')
        self._ElementArrayFile = ElementArrayFile
        # The drive file carries all steering and correction; the domain must hold elements on every side.
        for k in ('XSteering', 'YSteering', 'ZSteering', 'RotationZ', 'MultiPoint', 'DistanceConeToFocus'):
            kargs.pop(k, None)
        return super().RunCases(bDoRefocusing=False, bTightNarrowBeamDomain=False,
                                ExtraAdjustX=[0.0], ExtraAdjustY=[0.0], **kargs)


class BabelFTD_Simulations(BabelFTD_Simulations_BASE):
    def __init__(self, ElementArrayFile='', **kargs):
        self._ElementArrayFile = ElementArrayFile
        super().__init__(**kargs)
        self._bDoRefocusing = False
        self._bTightNarrowBeamDomain = False

    def CreateSimConditions(self, **kargs):
        kargs['bTightNarrowBeamDomain'] = False
        return SimulationConditions(ElementArrayFile=self._ElementArrayFile, **kargs)

    def GenerateSTLTx(self, prefix):
        '''Export the element faces as an STL in the mask's world coordinates (for Slicer or Brainsight).'''
        S = self._SIM_SETTINGS
        quads = np.array([ElementFacePoints(S._EA, n, 1.0)[1] for n in range(S._EA['NumberElems'])])  # (N,4,3) m
        V = quads.reshape(-1, 3).T / S.SpatialStep
        V = np.vstack([V, np.ones((1, V.shape[1]))])
        loc = np.array(np.where(self._SkullMask.get_fdata() == 5.0)).flatten()
        V[2, :] = -V[2, :]
        V[0, :] += loc[0]
        V[1, :] += loc[1]
        V[2, :] += loc[2] - S._TxMechanicalAdjustmentZ / S.SpatialStep
        V = (self._SkullMask.affine @ V).T[:, :3].reshape(-1, 4, 3)
        stl = mesh.Mesh(np.zeros(2 * len(V), dtype=mesh.Mesh.dtype))
        for i, q in enumerate(V):
            stl.vectors[2 * i] = q[[0, 1, 3]]
            stl.vectors[2 * i + 1] = q[[1, 2, 3]]
        stl.save(os.path.dirname(self._MASKFNAME) + os.sep + prefix + 'Tx.stl')

    def AddSaveDataSim(self, DataForSim):
        super().AddSaveDataSim(DataForSim)
        S = self._SIM_SETTINGS
        DataForSim['ElementArrayFile'] = os.path.basename(self._ElementArrayFile)
        DataForSim['ElementArraySHA256'] = S._EA['sha256']
        DataForSim['ElementDrive'] = S._EA['drive'].astype(np.complex64)
        DataForSim['ElementSourceVoxels'] = S._ElementVoxelCount
        DataForSim['bDoRefocusing'] = False


class SimulationConditions(SimulationConditionsBASE):
    '''Element array: per-element volumetric stress sources with complex drive; box domain around all elements.'''

    def __init__(self, ElementArrayFile='', DomainMargin=3, **kargs):
        self._EA = LoadElementArray(ElementArrayFile)
        c = self._EA['centers_m']
        ext = np.ptp(c, axis=0) + self._EA['size_m'].max()
        kargs['Aperture'] = float(ext.max())          # informational; the domain comes from DomeRegionMap
        kargs['FocalLength'] = 0.0
        kargs['bTightNarrowBeamDomain'] = False
        super().__init__(**kargs)
        if abs(self._EA['f0_hz'] - self._Frequency) > 1e-6 * self._Frequency:
            raise ValueError(f"drive file f0 {self._EA['f0_hz']} Hz != simulation frequency {self._Frequency} Hz")
        self._DomainMargin = DomainMargin                # voxels of water around the element box

    def UpdateConditions(self, SkullMaskNii, AlphaCFL=1.0, bWaterOnly=False, bForceHomogenousMedium=False,
                         BenchmarkTestFile='', DomeType=False):
        super().UpdateConditions(SkullMaskNii, AlphaCFL=AlphaCFL, bWaterOnly=bWaterOnly,
                                 bForceHomogenousMedium=bForceHomogenousMedium,
                                 BenchmarkTestFile=BenchmarkTestFile, DomeType=True)

    def DomeRegionMap(self, xfield, yfield, zfield):
        '''Region that must fit inside the domain: the box around every element face plus a margin, on both
        sides of the focus. Returns coordinates normalised so the box is |x|,|y|,|z| <= 1, and the map.'''
        half = self._EA['size_m'].max() / 2 + self._DomainMargin * self._SpatialStep
        c = self._EA['centers_m'] + np.array([self._TxMechanicalAdjustmentX, self._TxMechanicalAdjustmentY,
                                              self._TxMechanicalAdjustmentZ])
        lo, hi = c.min(0) - half, c.max(0) + half
        mid, ax = (lo + hi) / 2, (hi - lo) / 2
        xf2, yf2, zf2 = (xfield - mid[0]) / ax[0], (yfield - mid[1]) / ax[1], (zfield - mid[2]) / ax[2]
        inx, iny, inz = np.abs(xf2) <= 1.0, np.abs(yf2) <= 1.0, np.abs(zf2) <= 1.0
        RegionMap = inx[:, None, None] & iny[None, :, None] & inz[None, None, :]
        return xf2, yf2, zf2, RegionMap

    def GenTransducerGeom(self):
        ea = self._EA
        self._Tx = {'elemcenter': ea['centers_m'].copy(), 'normal': ea['normals'].copy(),
                    'NumberElems': ea['NumberElems'], 'Aperture': self._Aperture}
        self._TxOrig = self._Tx

    def CalculateRayleighFieldsForward(self, deviceName='6800'):
        '''No Rayleigh pre-step: the elements are sources inside the domain. Sets the drive.'''
        self.GenTransducerGeom()
        self.BasePhasedArrayProgramming = self._EA['drive'].astype(np.complex64)
        self.BasePhasedArrayProgrammingRefocusing = np.zeros_like(self.BasePhasedArrayProgramming)
        # The base class's result handling expects this array; there is no Rayleigh field here.
        self._u2RayleighField = np.zeros((self._N1, self._N2, self._N3), np.complex64)

    def _VoxeliseElements(self):
        '''Label map (N1,N2,N3) with element n as n+1. A voxel reached by several elements goes to the element
        with the most face samples in it (ties: lowest index). Returns the map and voxels per element.'''
        ea = self._EA
        h = self._SpatialStep
        orig = np.array([self._XDim[0], self._YDim[0], self._ZDim[0]])
        shift = np.array([self._TxMechanicalAdjustmentX, self._TxMechanicalAdjustmentY, self._TxMechanicalAdjustmentZ])
        lin_all, el_all = [], []
        for n in range(ea['NumberElems']):
            pts, _ = ElementFacePoints(ea, n, h / 3.0)
            ijk = np.round((pts + shift - orig) / h).astype(np.int64)
            if (ijk < self._PMLThickness).any() or (ijk[:, 0] >= self._N1 - self._PMLThickness).any() or \
               (ijk[:, 1] >= self._N2 - self._PMLThickness).any() or (ijk[:, 2] >= self._N3 - self._PMLThickness).any():
                raise ValueError(f'element {n} reaches the PML; increase the domain margin')
            lin = np.ravel_multi_index(ijk.T, (self._N1, self._N2, self._N3))
            lin_all.append(lin)
            el_all.append(np.full(lin.size, n, np.int64))
        lin_all, el_all = np.concatenate(lin_all), np.concatenate(el_all)
        key = lin_all * ea['NumberElems'] + el_all
        ukey, counts = np.unique(key, return_counts=True)
        vox, el = ukey // ea['NumberElems'], ukey % ea['NumberElems']
        order = np.lexsort((el, -counts, vox))             # per voxel: most samples first, then lowest index
        vox, el = vox[order], el[order]
        first = np.r_[True, vox[1:] != vox[:-1]]
        SourceMap = np.zeros(self._N1 * self._N2 * self._N3, np.uint32)
        SourceMap[vox[first]] = (el[first] + 1).astype(np.uint32)
        SourceMap = SourceMap.reshape((self._N1, self._N2, self._N3))
        nvox = np.bincount(el[first], minlength=ea['NumberElems'])
        if (nvox == 0).any():
            raise ValueError(f'{int((nvox == 0).sum())} element(s) got no source voxel')
        if (self._MaterialMap[SourceMap > 0] != 0).any():
            print('WARNING: some element source voxels lie in tissue, not water')
        return SourceMap, nvox

    def CreateSources(self, ramp_length=4):
        '''One stress-source time series per element: s * |d_n| * w_n * sin(w t + arg d_n), ramped up over
        `ramp_length` cycles (as BabelBrain's other transducer families do). w_n = (area_n / h^2) / voxels_n
        equalises the injected source strength across elements whatever their footprint on the grid.'''
        LengthSource = np.floor(self._TimeSimulation / (1.0 / self._Frequency)) * 1 / self._Frequency
        t = np.arange(0, LengthSource + self._TemporalStep, self._TemporalStep)
        nramp = int(np.round(ramp_length / self._Frequency / self._TemporalStep))
        ramp = (-np.cos(np.arange(0, np.pi, np.pi / nramp)) + 1) * 0.5
        self._SourceMap, self._ElementVoxelCount = self._VoxeliseElements()
        ea = self._EA
        area = (ea['size_m'][:, 0] * ea['size_m'][:, 1]) * (np.pi / 4 if ea['shape'] == 'circle' else 1.0)
        w = (area / self._SpatialStep ** 2) / self._ElementVoxelCount
        d = self.BasePhasedArrayProgramming.astype(np.complex128)
        amp = ea['pressure_scale_pa'] * np.abs(d) * w
        P = amp[:, None] * np.sin(2 * np.pi * self._Frequency * t[None, :] + np.angle(d)[:, None])
        P[:, :len(ramp)] *= ramp[None, :]
        self._PulseSource = P
        self._PulseAmplitude = ea['pressure_scale_pa']

    def CreateSensorMap(self):
        '''Record the whole interior (elements on several sides), plus no back-propagation sensors.'''
        p = self._PMLThickness
        self._SensorMap = np.zeros((self._N1, self._N2, self._N3), np.uint32)
        self._SensorMap[p:-p, p:-p, p:-p] = 1
        self._SensorMapBackPropagation = np.zeros((self._N1, self._N2, self._N3), np.uint32)

    def RUN_SIMULATION(self, bDoStressSource=False, SelRMSorPeak=1, bApplyCorrectionForDispersion=True, **kargs):
        kargs['bDoRefocusing'] = False
        super().RUN_SIMULATION(bDoStressSource=True, bApplyCorrectionForDispersion=False,
                               SelRMSorPeak=1, **kargs)

    def CalculatePhaseData(self, bRefocused=False, bDoRefocusing=True, bDoRefocusingVolume=False):
        super().CalculatePhaseData(bRefocused=bRefocused, bDoRefocusing=False)

    def ReturnResults(self, bDoRefocusing=True, bUseRayleighForWater=False, bDoRefocusingVolume=False):
        return super().ReturnResults(bDoRefocusing=False, bUseRayleighForWater=False)
