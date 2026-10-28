"""
fig_appendix_p1_cyl_sim.py
--------------------------
Phase I rollout visualisations on cylinder_numerical (sim) for all 6 conditions:
3 learning rates × 2 normalisations (GN / SW), batch 0.

Layout: 3 LR row-groups (lr=1e-4 | lr=2e-5 | lr=5e-5),
each group = 2 time rows (t=20, t=60) × 5 cols (GT | GN pred | GN err | SW pred | SW err).
Total: 6 rows × 5 cols.

All videos: walrus_ft_lr{1e4,2e5,5e5}_{gn,sw_gninf}_eval
  /finetune/0/viz/realpdebench_cylinder_numerical/rollout_video/
  epoch51_rank0_rollout_test_batch0_realpdebench_cylinder_numerical.mp4
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT   = Path(f"{ROOT}/realpdebench/analysis/plots")
EXPBASE = Path(f"{ROOT}/walrus/experiments")
WANT  = {20, 60}

# ---------------------------------------------------------------------------
# Cylinder crop constants (same as fig_sw_vs_gn_rollout)
# ---------------------------------------------------------------------------
GT_ROWS   = slice(90,  264)
PRED_ROWS = slice(270, 444)
ERR_ROWS  = slice(454, 628)
VX_COLS   = slice(33,  108)

def load_frames(mp4_path):
    r = iio.get_reader(str(mp4_path))
    frames = {}
    for i, f in enumerate(r):
        if i in WANT: frames[i] = f
        if i > 60: break
    r.close()
    return frames

def get(frames, t, rows, cols):
    return np.rot90(frames[t][rows, cols, :], k=3)

# ---------------------------------------------------------------------------
# Video paths
# ---------------------------------------------------------------------------
LRS  = ["1e-4", "2e-5", "5e-5"]
DIRS = {"1e-4": "lr1e4", "2e-5": "lr2e5", "5e-5": "lr5e5"}

def vid_path(lr_tag, norm_suffix):
    # norm_suffix already includes the trailing portion, e.g. "gn_eval"
    folder = f"walrus_ft_{DIRS[lr_tag]}_{norm_suffix}"
    return (EXPBASE / folder / "finetune/0/viz"
            / "realpdebench_cylinder_numerical/rollout_video"
            / f"epoch51_rank0_rollout_test_batch0_realpdebench_cylinder_numerical.mp4")

# Pre-load all frames
gn_frames = {lr: load_frames(vid_path(lr, "gn_eval")) for lr in LRS}
sw_frames = {lr: load_frames(vid_path(lr, "sw_gninf_eval")) for lr in LRS}

# ---------------------------------------------------------------------------
# Build panel grid
# panels[ri][ci] — ri: row (0-5), ci: col (0-4)
# Row order: lr1e4-t20, lr1e4-t60, lr2e5-t20, lr2e5-t60, lr5e5-t20, lr5e5-t60
# Col order:  GT | GN pred | GN err | SW pred | SW err
# GT taken from GN video (same trajectory)
# ---------------------------------------------------------------------------
TIMES = [20, 60]
panels = []
row_meta = []   # (lr_str, t)
for lr in LRS:
    for t in TIMES:
        row = [
            get(gn_frames[lr], t, GT_ROWS,   VX_COLS),
            get(gn_frames[lr], t, PRED_ROWS, VX_COLS),
            get(gn_frames[lr], t, ERR_ROWS,  VX_COLS),
            get(sw_frames[lr], t, PRED_ROWS, VX_COLS),
            get(sw_frames[lr], t, ERR_ROWS,  VX_COLS),
        ]
        panels.append(row)
        row_meta.append((lr, t))

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
ph, pw = panels[0][0].shape[:2]   # (174, 75) after rotation

panel_scale = 0.008
gap         = 0.05   # between panels within a group
group_gap   = 0.12   # between GN-pair and SW-pair
t_gap       = 0.06   # between t=20 / t=60 within an LR group
lr_gap      = 0.18   # between LR groups
label_w     = 0.55
col_label_h = 0.60
FSIZE       = 10

cell_w = pw * panel_scale
cell_h = ph * panel_scale

# Column x-positions (same for all rows)
col_lefts = [
    label_w,
    label_w + cell_w + group_gap,
    label_w + cell_w + group_gap + cell_w + gap,
    label_w + 3*cell_w + group_gap + gap + group_gap,
    label_w + 3*cell_w + group_gap + gap + group_gap + cell_w + gap,
]

# Row y-positions (bottom of each panel, from top)
# Within each LR group: t=20 row, gap, t=60 row; between groups: lr_gap
row_bottoms_from_top = []
y = 0.0
for i_lr, lr in enumerate(LRS):
    if i_lr > 0:
        y += lr_gap
    for i_t, t in enumerate(TIMES):
        if i_t > 0:
            y += t_gap
        row_bottoms_from_top.append(y)
        y += cell_h

total_h_content = y
fig_w = col_lefts[-1] + cell_w + 0.10
fig_h = col_label_h + total_h_content + 0.10

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

col_labels = ["Ground truth", "GN pred", "GN error", "SW pred", "SW error"]
col_colors = ["#333333", "#1b6ca8", "#1b6ca8", "#c0392b", "#c0392b"]

for ri, (row, (lr_str, t)) in enumerate(zip(panels, row_meta)):
    y_from_top = row_bottoms_from_top[ri]
    bottom = (fig_h - col_label_h - y_from_top - cell_h) / fig_h

    for ci, img in enumerate(row):
        left   = col_lefts[ci] / fig_w
        width  = cell_w / fig_w
        height = cell_h / fig_h

        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(img, aspect="equal", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

        # Column headers on first data row only
        if ri == 0:
            ax.set_title(col_labels[ci], fontsize=FSIZE, pad=4,
                         color=col_colors[ci], fontfamily="DejaVu Serif")

        # Row label: lr and t on leftmost column
        if ci == 0:
            lbl = f"lr={lr_str}\n$t={t}$"
            ax.set_ylabel(lbl, fontsize=FSIZE-1, labelpad=5,
                          fontfamily="DejaVu Serif")

        # Red border on SW columns
        if ci in (3, 4):
            for sp in ax.spines.values():
                sp.set_edgecolor("#c0392b"); sp.set_linewidth(1.8)

# LR group separator lines
ax0 = fig.add_axes([0, 0, 1, 1], frameon=False)
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
ax0.axis("off")

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_appendix_p1_cyl_sim.{ext}",
                bbox_inches="tight", pad_inches=0.08,
                dpi=300 if ext == "png" else None)
    print(f"Saved fig_appendix_p1_cyl_sim.{ext}")

plt.close(fig)
