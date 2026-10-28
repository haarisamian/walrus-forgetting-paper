"""
fig_sw_vs_gn_rollout.py
-----------------------
Extracts frames from two cylinder_real_cl rollout videos (SW vs GN, epoch 30,
validation batch 2) and assembles a 2×5 panel figure showing vx predictions
and spatial errors at t=20 and t=60.

Column order: Ground truth | GN pred | GN error | SW pred | SW error
Rows:         t=20         | t=60

SW model: phase2_sw_real / walrus_cl_v2_real100pct_sim0pct_fixed (SW normalization)
GN model: walrus_gn_to_gn_real100pct (GlobalRevNormalization)
Both: epoch 40, rollout_valid, batch 2, cylinder_real_cl dataset

Frames rotated 90° clockwise so the cylinder wake runs left→right.
Error panels: extracted from the video's bottom row (|pred − GT| rendered by walrus).
  Video layout (pixel rows, before rotation):
    GT   rows  90–264
    Pred rows 270–448
    Err  rows 448–636  ← actual computed error, same colormap walrus uses
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

OUT = Path(f"{ROOT}/realpdebench/analysis/plots")

# ---------------------------------------------------------------------------
# GT vx field change norm — loaded from HDF5 (actual float32 arrays)
# valid batch 2 = trajectory index 2 (Re=4406).
# GN/SW prediction arrays are not saved to disk; those norms are placeholders.
# ---------------------------------------------------------------------------
VALID_HDF5 = (f"{ROOT}/realpdebench/well/"
              "cylinder_real_cl/data/valid/cylinder_real_cl_valid.hdf5")
with h5py.File(VALID_HDF5, "r") as _f:
    _vx = _f["t1_fields"]["velocity"][2, :, :, :, 0]   # traj 2, all t, 128×64

def _l2_delta(vx_arr, t0, t1):
    return float(np.linalg.norm(vx_arr[t1] - vx_arr[t0]))

# Norms are ||vx(t=60) - vx(t=20)||_2 in physical units
# GT: from HDF5 valid split, traj 2.
# GN / SW: from ypred npy dumps (dump_vx_fields Slurm job, epoch-40 checkpoints).
_DUMP_GN = (f"{ROOT}/walrus/experiments/dump_vx_fields_gn"
            "/finetune/0/viz/realpdebench_cylinder_real_cl/full_trajectory_dumps/")
_DUMP_SW = (f"{ROOT}/walrus/experiments/dump_vx_fields_sw"
            "/finetune/0/viz/realpdebench_cylinder_real_cl/full_trajectory_dumps/")
import numpy as _np2
_gn_vx = _np2.load(_DUMP_GN + "ypred_realpdebench_cylinder_real_cl_rollout_valid_epoch101_rank0_2.npy")[0, :, :, :, 0, 0]
_sw_vx = _np2.load(_DUMP_SW + "ypred_realpdebench_cylinder_real_cl_rollout_valid_epoch51_rank0_2.npy")[0, :, :, :, 0, 0]

FIELD_NORMS = {
    "Ground truth": f"{_l2_delta(_vx, 20, 60):.3f}",
    "GN pred":      f"{float(_np2.linalg.norm(_gn_vx[60] - _gn_vx[20])):.3f}",
    "GN error":     None,
    "SW pred":      f"{float(_np2.linalg.norm(_sw_vx[60] - _sw_vx[20])):.3f}",
    "SW error":     None,
}

SW_VID = (f"{ROOT}/walrus/experiments/phase2_sw_real/"
          "walrus_cl_v2_real100pct_sim0pct_fixed-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/"
          "finetune/0/viz/realpdebench_cylinder_real_cl/rollout_video/"
          "epoch40_rank0_rollout_valid_batch2_realpdebench_cylinder_real_cl.mp4")

GN_VID = str(list(Path(f"{ROOT}/walrus/experiments/").glob(
    "walrus_gn_to_gn_real100pct*/finetune/0/viz/realpdebench_cylinder_real_cl"
    "/rollout_video/epoch40_rank0_rollout_valid_batch2*.mp4"))[0])

# ---------------------------------------------------------------------------
# Extract frames at timesteps t=20 and t=60
# ---------------------------------------------------------------------------
WANT = {20, 60}
sw_frames = {}; gn_frames = {}
for path, store in [(SW_VID, sw_frames), (GN_VID, gn_frames)]:
    r = iio.get_reader(path)
    for i, f in enumerate(r):
        if i in WANT: store[i] = f
        if i > 60: break
    r.close()

# ---------------------------------------------------------------------------
# Crop parameters (derived from video layout inspection)
# GT rows / Pred rows / vx left-half column slice
# Frames rotated 90° CW so cylinder wake runs horizontally
# ---------------------------------------------------------------------------
GT_ROWS   = slice(90,  264)    # 174 rows of field data
PRED_ROWS = slice(270, 444)    # 174 rows (trimmed from 178 to match GT aspect)
ERR_ROWS  = slice(454, 628)    # 174 rows — skip top border spike at 453, match GT
VX_COLS   = slice(33,  108)    # 75 cols — vx channel, excluding axis borders

def get(frames, t, rows, cols):
    return np.rot90(frames[t][rows, cols, :], k=3)

# Columns: GT | GN pred | GN err | SW pred | SW err
# Error panels are taken directly from each model's video bottom row —
# that is the actual |pred − GT| walrus computed (not pixel arithmetic).
panels = [
    [get(sw_frames, 20, GT_ROWS,   VX_COLS),   # GT        (from SW video, same GT)
     get(gn_frames, 20, PRED_ROWS, VX_COLS),   # GN pred
     get(gn_frames, 20, ERR_ROWS,  VX_COLS),   # GN error  (walrus-rendered)
     get(sw_frames, 20, PRED_ROWS, VX_COLS),   # SW pred
     get(sw_frames, 20, ERR_ROWS,  VX_COLS)],  # SW error  (walrus-rendered)
    [get(sw_frames, 60, GT_ROWS,   VX_COLS),
     get(gn_frames, 60, PRED_ROWS, VX_COLS),
     get(gn_frames, 60, ERR_ROWS,  VX_COLS),
     get(sw_frames, 60, PRED_ROWS, VX_COLS),
     get(sw_frames, 60, ERR_ROWS,  VX_COLS)],
]

# ---------------------------------------------------------------------------
# Figure layout — manual axes placement, pixel-exact aspect ratio preserved
# Target: ~7.5" wide (full-page paper column), 300 dpi
# ---------------------------------------------------------------------------
ph, pw = panels[0][0].shape[:2]

panel_scale = 0.008   # inches per pixel — shrinks panels to paper-fit width
gap         = 0.06    # gap between panels within a group
group_gap   = 0.14    # larger gap separating GT | GN-pair | SW-pair
label_w     = 0.50    # left margin for row ylabel
col_label_h = 0.82    # top margin for col titles (accommodates 2-line norm header)
FSIZE       = 11      # publication-ready font size

cell_w = pw * panel_scale
cell_h = ph * panel_scale

# Column layout: GT [group_gap] GN-pred gap GN-err [group_gap] SW-pred gap SW-err
fig_w  = label_w + 5 * cell_w + 2 * gap + 2 * group_gap
fig_h  = col_label_h + 2 * cell_h + gap

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

# Pre-compute left edges for each of the 5 columns
col_lefts = [
    label_w,                                          # 0: GT
    label_w + cell_w + group_gap,                     # 1: GN pred
    label_w + cell_w + group_gap + cell_w + gap,      # 2: GN error
    label_w + 3*cell_w + group_gap + gap + group_gap, # 3: SW pred
    label_w + 3*cell_w + group_gap + gap + group_gap + cell_w + gap,  # 4: SW error
]

col_labels = ["Ground truth",
              "GN pred",  "GN error",
              "SW pred",  "SW error"]
col_colors = ["#333333",
              "#1b6ca8",  "#1b6ca8",
              "#c0392b",  "#c0392b"]
row_labels  = ["$t=20$", "$t=60$"]

for ri, (row, rl) in enumerate(zip(panels, row_labels)):
    for ci, img in enumerate(row):
        left   = col_lefts[ci] / fig_w
        bottom = (fig_h - col_label_h - (ri + 1) * cell_h - ri * gap) / fig_h
        width  = cell_w / fig_w
        height = cell_h / fig_h

        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(img, aspect="equal", interpolation="bilinear")

        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)

        if ri == 0:
            norm_val = FIELD_NORMS.get(col_labels[ci])
            if norm_val is not None:
                title_str = f"{col_labels[ci]}\n$\|\\Delta v_x\|_2={norm_val}$"
                pad = 5
            else:
                title_str = col_labels[ci]
                pad = 8
            ax.set_title(title_str, fontsize=FSIZE, pad=pad,
                         color=col_colors[ci], fontfamily="DejaVu Serif",
                         linespacing=1.4)
        if ci == 0:
            ax.set_ylabel(rl, fontsize=FSIZE, labelpad=7,
                          fontfamily="DejaVu Serif")
        # Red border on SW columns
        if ci in (3, 4):
            for sp in ax.spines.values():
                sp.set_edgecolor("#c0392b"); sp.set_linewidth(2.2)

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_sw_vs_gn_rollout.{ext}",
                bbox_inches="tight", pad_inches=0.08,
                dpi=300 if ext == "png" else None)
    print(f"Saved fig_sw_vs_gn_rollout.{ext}")

plt.close(fig)
