"""
Figure 1 - paper schematic, Tufte pass.

Eraser test applied to the previous version. Removed as non-data ink:
  - coloured accent bars on each card (pure decoration, encoded nothing)
  - card border rectangles (whitespace already groups the three stages)
  - grey italic subtitles and bottom captions (provenance -> LaTeX caption,
    where words belong, rather than competing with the graphic)
  - stage-identity colour (position and title already encode stage)

Kept, because each carries information:
  - stage titles (structure), arrows and their labels (mechanism/causality),
    u and v row keys (identify the data), the image tiles themselves.

Colour now appears ONLY inside the data tiles. All chrome is greyscale, so
nothing competes with the fields for attention. The ink freed by deleting the
boxes goes into larger tiles: the cylinder frames grow ~16% on a side.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import os
import pathlib
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _paperstyle import use_latex_style

CYL_NUM = f"{ROOT}/realpdebench/well/cylinder_numerical.hdf5"
CYL_REAL = f"{ROOT}/realpdebench/well/cylinder_real.hdf5"
WELL_DUMP = f"{ROOT}/walrus/experiments/dump_well_pretrained"
OUT_DIR = f"{ROOT}/realpdebench/analysis/plots/well_appendix_final"
os.makedirs(OUT_DIR, exist_ok=True)

use_latex_style()

INK = "#111111"          # titles: the only near-black text
MID = "#444444"          # arrows and their labels
KEY = "#555555"          # row keys
CMAP_WELL, CMAP_MAG, CMAP_DIV = "cividis", "inferno", "RdBu_r"

FS_TITLE, FS_ARROW, FS_KEY = 9.2, 6.5, 7.0

WELL_DIRS = [
    "acoustic_scattering_discontinuous", "acoustic_scattering_inclusions",
    "acoustic_scattering_maze", "active_matter", "helmholtz_staircase",
    "MHD_64", "planetswe", "shear_flow", "turbulent_radiative_layer_2D",
    "turbulence_gravity_cooling",
]


def load_well_tile(ds_dir):
    files = sorted(glob.glob(os.path.join(WELL_DUMP, ds_dir, "**", "yref_*.npy"),
                             recursive=True))
    a = np.load(files[0])[0]
    a = a[a.shape[0] // 2]
    if a.ndim == 4:
        a = a[:, :, a.shape[2] // 2, :]
    ci = int(np.argmax([np.std(a[..., c]) for c in range(a.shape[-1])]))
    img = a[..., ci]
    h, w = img.shape
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


# The two velocity components, shown as a matched pair.
# u is one-sided (never reverses in PIV, 1.5% of sim cells) -> sequential map.
# v is symmetric about zero (~44% negative in both) -> diverging map.
# Different colormaps here is correct encoding, not inconsistency.
streamwise  = lambda vel: vel[..., 0].T
transverse  = lambda vel: vel[..., 1].T

T_PICKS = sorted([600, 2200])
RE_TARGET = 4000


def load_cylinder(path, label):
    with h5py.File(path, "r") as f:
        re = f["scalars/reynolds_number"][:]
        time = f["dimensions/time"][:]
        assert np.all(np.diff(time) > 0), f"{label}: time not increasing"
        i = int(np.argmin(np.abs(re - RE_TARGET)))
        vel = np.stack([f["t1_fields/velocity"][i, t] for t in T_PICKS])
    print(f"  {label}: traj {i}, Re={re[i]:.0f}, "
          f"t={[round(float(time[t]), 2) for t in T_PICKS]}")
    return vel, float(re[i]), [float(time[t]) for t in T_PICKS], re


print("loading ...")
well_tiles = [load_well_tile(d) for d in WELL_DIRS]
vel_n, RE_N, TIMES_N, re_n = load_cylinder(CYL_NUM, "simulation")
vel_r, RE_R, TIMES_R, re_r = load_cylinder(CYL_REAL, "PIV")
assert np.array_equal(re_n, re_r), "sim/real Reynolds arrays differ"
assert np.allclose(TIMES_N, TIMES_R), "sim/real sample times differ"
sim_u, sim_v = [streamwise(v) for v in vel_n], [transverse(v) for v in vel_n]
real_u, real_v = [streamwise(v) for v in vel_r], [transverse(v) for v in vel_r]

# ───────────────────────── geometry (inches) ─────────────────────────
# Width is fixed by \textwidth. HEIGHT IS DERIVED from the content below, so
# the canvas is exactly as tall as the figure needs and no taller -- tile size
# is width-bound, so any extra height is dead margin, not bigger data.
FIG_W = 7.20

M_L = M_R = 0.12
M_T, M_B = 0.10, 0.10
CARD_GAP = 0.52                    # must clear the arrow LABEL, not just the arrow
CARD_W = (FIG_W - M_L - M_R - 2 * CARD_GAP) / 3
CARD_X = [M_L + i * (CARD_W + CARD_GAP) for i in range(3)]

TITLE_H = 0.17                     # band reserved for the stage title
BLOCK_GAP = 0.06                   # title band to tiles

MOS_GAP = 0.024
MOS_S = (CARD_W - 4 * MOS_GAP) / 5
MOS_H = 2 * MOS_S + MOS_GAP

CYL_GAP = 0.030
ROWKEY_W = 0.125
CYL_W = (CARD_W - ROWKEY_W - CYL_GAP) / 2
CYL_HT = CYL_W / 2.0
CYL_H = 2 * CYL_HT + CYL_GAP

BLOCK_H = max(MOS_H, CYL_H)
CONTENT_H = TITLE_H + BLOCK_GAP + BLOCK_H
FIG_H = CONTENT_H + M_T + M_B          # snug by construction
CONTENT_TOP = FIG_H - M_T
BLOCK_TOP = CONTENT_TOP - TITLE_H - BLOCK_GAP
BLOCK_MID = BLOCK_TOP - BLOCK_H / 2

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
fx, fy = (lambda x: x / FIG_W), (lambda y: y / FIG_H)


def ax_in(x, y, w, h):
    a = fig.add_axes([fx(x), fy(y), fx(w), fy(h)])
    a.set_xticks([]); a.set_yticks([])
    return a


def tile(x, y, w, h, img, cmap, sym=False):
    a = ax_in(x, y, w, h)
    if sym:
        m = np.percentile(np.abs(img), 96)
        a.imshow(img, origin="lower", cmap=cmap, vmin=-m, vmax=m,
                 aspect="auto", interpolation="nearest")
    else:
        lo, hi = np.percentile(img, [2, 98])
        a.imshow(img, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                 aspect="auto", interpolation="nearest")
    for s in a.spines.values():          # hairline separator, not a frame
        s.set_edgecolor("white"); s.set_linewidth(0.5)
    return a


def text_in(x, y, s, size, color, weight="normal", ha="center", va="center"):
    fig.text(fx(x), fy(y), s, fontsize=size, color=color, fontweight=weight,
             ha=ha, va=va)


# ─────────────────────────── titles only ────────────────────────────
title_y = CONTENT_TOP - TITLE_H / 2
# "+ Replay" is already carried by the arrow label, so repeating it in the
# title is duplicate ink; a middle dot also costs less width than an em dash.
for i, t in enumerate(["Pretraining", "Phase I · Simulation",
                       "Phase II · Real PIV"]):
    text_in(CARD_X[i] + CARD_W / 2, title_y, t, FS_TITLE, INK, "bold")

# ───────────────────────── the Well mosaic ──────────────────────────
mos_y0 = BLOCK_MID - MOS_H / 2
for k, img in enumerate(well_tiles):
    r, c = divmod(k, 5)
    tile(CARD_X[0] + c * (MOS_S + MOS_GAP),
         mos_y0 + MOS_H - (r + 1) * MOS_S - r * MOS_GAP,
         MOS_S, MOS_S, img, CMAP_WELL)

# ───────────────────── cylinder simulation / PIV ────────────────────
for ci, (us, vs) in enumerate([(sim_u, sim_v), (real_u, real_v)]):
    x0 = CARD_X[ci + 1]
    xt = x0 + ROWKEY_W
    cyl_y0 = BLOCK_MID - CYL_H / 2
    for k in range(len(T_PICKS)):
        xk = xt + k * (CYL_W + CYL_GAP)
        tile(xk, cyl_y0 + CYL_HT + CYL_GAP, CYL_W, CYL_HT, us[k], CMAP_MAG)
        tile(xk, cyl_y0, CYL_W, CYL_HT, vs[k], CMAP_DIV, sym=True)
    key_x = xt - 0.028
    text_in(key_x, cyl_y0 + CYL_HT + CYL_GAP + CYL_HT / 2, "$u$",
            FS_KEY, KEY, ha="right")
    text_in(key_x, cyl_y0 + CYL_HT / 2, "$v$", FS_KEY, KEY, ha="right")

# ──────────────────────── arrows = mechanism ────────────────────────
for i, label in enumerate(["fine-tune", "adapt\n+ replay"]):
    xa = CARD_X[i] + CARD_W + 0.055
    xb = CARD_X[i + 1] - 0.055
    fig.add_artist(FancyArrowPatch((fx(xa), fy(BLOCK_MID)), (fx(xb), fy(BLOCK_MID)),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=6, linewidth=0.8, color=MID,
                                   shrinkA=0, shrinkB=0))
    fig.text(fx((xa + xb) / 2), fy(BLOCK_MID + 0.055), label, fontsize=FS_ARROW,
             color=MID, ha="center", va="bottom", linespacing=1.25)

png, pdf = f"{OUT_DIR}/fig1_schematic.png", f"{OUT_DIR}/fig1_schematic.pdf"
fig.savefig(png, dpi=400, facecolor="white")
fig.savefig(pdf, facecolor="white")
print(f"\nsaved  {FIG_W} x {FIG_H:.3f} in  (was 1.740 -> {100*(1-FIG_H/1.74):.0f}% shorter, same width)")
print(f"  cylinder tile {CYL_W:.3f} x {CYL_HT:.3f} in "
      f"(was 0.804 x 0.402 -> +{100*(CYL_W/0.804-1):.0f}% wider)")
print(f"  Well tile {MOS_S:.3f} in (was 0.336 -> +{100*(MOS_S/0.336-1):.0f}%)")
