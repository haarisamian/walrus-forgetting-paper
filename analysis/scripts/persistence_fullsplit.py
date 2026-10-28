"""
Persistence baseline over the FULL Well validation split.

The original estimate used the evaluation's visualisation dumps, which contain
only three distinct rollout windows per dataset (six files, duplicated pairs).
This recomputes the same quantity over every validation trajectory.

Definition, unchanged: NRMSE of the trivial predictor y(t+1) = y(t), computed
per (trajectory, timestep, channel) over the spatial axes and then averaged.
NRMSE normalises by the RMS of the target, so it is comparable to the NRMSE the
model is scored with.

Conventions matched to the evaluation pipeline:
  * scored channel set taken from the stored evaluation loss dicts, so exactly
    the fields the model is scored on are used;
  * stride 1 for every dataset. step_downsample_factor does NOT subsample time
    -- it only scales n_steps_input (multidataset.py:259) -- and the config sets
    min_dt_stride = max_dt_stride = 1 throughout, so consecutive stored frames
    are what the model steps between;
  * log10 transform for turbulence_gravity_cooling density and temperature;
  * 3D datasets reduced to their central z slice, matching the dump convention.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import os
import re
import pickle
import io

import h5py
import numpy as np
import torch

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b), map_location="cpu", weights_only=False)

WELLD = f"{ROOT}/the_well/datasets/datasets"
WELL = f"{ROOT}/the_well/datasets"
FE = f"{ROOT}/walrus/experiments/forget_eval_results"
MAX_TRAJ = 32
MAX_T = 128    # frames per trajectory; ~4000 frame-pairs/channel is ample

# dir -> (eval name, stride, log10 fields, component suffixes)
SPEC = {
    "acoustic_scattering_discontinuous": ("acoustic_scattering_discontinuous", 1, (), "xyz"),
    "acoustic_scattering_inclusions":    ("acoustic_scattering_inclusions", 1, (), "xyz"),
    "acoustic_scattering_maze":          ("acoustic_scattering_maze", 1, (), "xyz"),
    "helmholtz_staircase":               ("helmholtz_staircase", 1, (), "xyz"),
    "planetswe":                         ("planetswe", 1, (), ("phi", "theta")),
    "shear_flow":                        ("shear_flow", 1, (), "xyz"),
    "active_matter":                     ("active_matter", 1, (), "xyz"),
    "MHD_64":                            ("MHD_64", 1, (), "xyz"),
    "turbulent_radiative_layer_2D":      ("trl2d", 1, (), "xyz"),
    "turbulence_gravity_cooling":        ("turbcool", 1, ("density", "temperature"), "xyz"),
}
TENSOR_IDX = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")


def scored_channels(evalname):
    fs = glob.glob(f"{FE}/pretrained/{evalname}/**/valid_loss_dict_*.pkl", recursive=True)
    D = pickle.load(open(fs[0], "rb"))
    return sorted({m.group(1) for k in D
                   if (m := re.match(r".*/(.+)_VRMSE_T=all_mean$", k))
                   and m.group(1) != "full"})


def channel_views(h, want, comps):
    """Yield (channel_name, dataset_handle, indexer) for each scored channel."""
    out = {}
    for grp in ("t0_fields", "t1_fields", "t2_fields"):
        if grp not in h:
            continue
        for name, d in h[grp].items():
            if grp == "t0_fields":
                if name in want and d.ndim >= 3:
                    out[name] = (d, ())
            elif grp == "t1_fields":
                for i in range(d.shape[-1]):
                    k = f"{name}_{comps[i]}"
                    if k in want:
                        out[k] = (d, (i,))
            else:                                   # t2_fields, e.g. D_xx
                nd = int(round(d.shape[-1]))
                for i in range(nd):
                    for j in range(d.shape[-2]):
                        k = f"{name}_{TENSOR_IDX[j * 3 + i][:2]}"
                        k2 = f"{name}_{'xyz'[j]}{'xyz'[i]}"
                        if k2 in want:
                            out[k2] = (d, (j, i))
    return out


def persistence_traj(a, stride, logf):
    """Returns (NRMSE, VRMSE) of copying the previous frame.

    NRMSE normalises by the RMS of the target, VRMSE by its standard deviation.
    VRMSE is the normalisation the model is scored with in the forgetting table,
    so it is the coherent choice when persistence is used to predict VRMSE
    degradation; NRMSE is reported alongside for comparability."""
    if logf:
        a = np.log10(np.clip(a, 1e-30, None))
    if a.ndim == 4:                                  # (T, X, Y, Z) -> central z
        a = a[:, :, :, a.shape[3] // 2]
    if a.shape[0] <= stride:
        return None
    y1, y0 = a[stride:], a[:-stride]
    sp = tuple(range(1, y1.ndim))
    num = np.sqrt(((y1 - y0) ** 2).mean(axis=sp))
    rms = np.sqrt((y1 ** 2).mean(axis=sp))
    sd = y1.std(axis=sp)
    return num / (rms + 1e-12), num / (sd + 1e-12)


def main():
    print(f"{'dataset':34}{'traj':>6}{'chan':>6}{'NRMSE':>11}{'VRMSE':>11}{'sem(V)':>9}")
    res = {}
    for ddir, (evalname, stride, logs, comps) in SPEC.items():
        want = scored_channels(evalname)
        files = sorted(glob.glob(f"{WELLD}/{ddir}/data/valid/*.hdf5")) or \
                sorted(glob.glob(f"{WELL}/{ddir}/data/valid/*.hdf5"))
        per_traj, ntraj = [], 0
        for p in files:
            if ntraj >= MAX_TRAJ:
                break
            with h5py.File(p, "r") as h:
                views = channel_views(h, want, comps)
                if not views:
                    continue
                n = next(iter(views.values()))[0].shape[0]
                for t in range(n):
                    if ntraj >= MAX_TRAJ:
                        break
                    vals = []
                    for cname, (d, idx) in views.items():
                        sl = ((t, slice(0, MAX_T))
                              + (slice(None),) * (d.ndim - 2 - len(idx)) + idx)
                        a = np.asarray(d[sl], dtype=np.float64)
                        r = persistence_traj(a, stride, cname in logs)
                        if r is not None:
                            vals.append((r[0].mean(), r[1].mean()))
                    if vals:
                        per_traj.append(np.mean(np.array(vals), axis=0))
                        ntraj += 1
        if not per_traj:
            print(f"{ddir:34}{'FAILED':>34}")
            continue
        v = np.array(per_traj)
        res[evalname] = (float(v[:, 0].mean()), float(v[:, 1].mean()))
        sem = (v[:, 1].std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        print(f"{ddir:34}{len(v):6d}{len(want):6d}"
              f"{v[:,0].mean():11.4f}{v[:,1].mean():11.4f}{sem:9.4f}")
    prev = {"acoustic_scattering_discontinuous": .1614, "acoustic_scattering_inclusions": .2142,
            "acoustic_scattering_maze": .9003, "active_matter": .3501, "helmholtz_staircase": .1283,
            "MHD_64": .2273, "planetswe": .0246, "shear_flow": .0282, "trl2d": .2189,
            "turbcool": .1387}
    print(f"\n{'dataset':34}{'NRMSE':>10}{'VRMSE':>10}{'3-window':>11}")
    for k, v in res.items():
        print(f"{k:34}{v[0]:10.4f}{v[1]:10.4f}{prev[k]:11.4f}")
    print("\nNEW_N =", {k: round(v[0], 4) for k, v in res.items()})
    print("NEW_V =", {k: round(v[1], 4) for k, v in res.items()})
    np.save(f"{ROOT}/realpdebench/analysis/persistence_fullsplit.npy", res)


if __name__ == "__main__":
    main()
