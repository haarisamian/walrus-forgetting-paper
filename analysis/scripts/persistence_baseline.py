"""
Persistence baseline: NRMSE of the trivial predictor y(t+1) = y(t).

This is the exact computation used for the `Pers.` column in the forgetting
table, preserved here so the numbers are reproducible.

Source data is the ground-truth reference dump (`yref_*.npy`) written by the
evaluation pipeline, so the frames are already at the evaluation's temporal
stride and in the evaluation's transformed space -- which is what makes the
number comparable to the model's reported error.

NRMSE = RMSE / RMS(target), computed per (timestep, channel) over spatial axes
and then averaged. 3D datasets are reduced to their central z slice.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import os

import numpy as np

DUMP = f"{ROOT}/walrus/experiments/dump_well_pretrained"
DATASETS = [
    "acoustic_scattering_discontinuous", "acoustic_scattering_inclusions",
    "acoustic_scattering_maze", "active_matter", "helmholtz_staircase",
    "MHD_64", "planetswe", "shear_flow", "turbulent_radiative_layer_2D",
    "turbulence_gravity_cooling",
]


def persistence(arr):
    """NRMSE of copying the previous frame. arr is [T, H, W, (Dz,) C]."""
    a = arr.astype(np.float64)
    if a.ndim == 5:
        a = a[:, :, :, a.shape[3] // 2, :]      # central z slice
    y1, y0 = a[1:], a[:-1]
    sp = tuple(range(1, y1.ndim - 1))           # spatial axes; keep time, channel
    num = np.sqrt(((y1 - y0) ** 2).mean(axis=sp))
    den = np.sqrt((y1 ** 2).mean(axis=sp))
    return float(np.mean(num / (den + 1e-12)))


def main(all_files=False):
    print("Persistence baseline: NRMSE of predicting y(t+1) = y(t)\n")
    print(f"  {'dataset':36}{'persistence':>12}{'trajectories':>14}")
    out = {}
    for d in DATASETS:
        fs = sorted(glob.glob(os.path.join(DUMP, d, "**", "yref_*.npy"),
                              recursive=True))
        if not fs:
            print(f"  {d:36}{'-- no dump found':>26}")
            continue
        use = fs if all_files else fs[:1]
        vals = [persistence(np.load(f)[0]) for f in use]
        out[d] = float(np.mean(vals))
        print(f"  {d:36}{out[d]:12.4f}{len(use):14d}")
    return out


if __name__ == "__main__":
    import sys
    main(all_files="--all" in sys.argv)
