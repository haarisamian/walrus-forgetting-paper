"""
fig_appendix_p2_forgetting.py
------------------------------
Phase II forgetting trajectory on cylinder_numerical (sim).
6 training checkpoints × 2 time rows (t=20, t=60).

Columns: Phase-I ep100 | P2 ep20 | P2 ep40 | P2 ep60 | P2 ep80 | P2 ep100
Each condition: GT (shown once on far left) + Pred + Error.
Layout: 2 rows × 13 cols = (GT | 6×(pred+err))

Video source: forget_eval_results/{folder}/cyl_sim/finetune/0/viz/
    realpdebench_cylinder_numerical/rollout_video/
    epoch52_rank0_rollout_test_batch0_realpdebench_cylinder_numerical.mp4
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT    = Path(f"{ROOT}/realpdebench/analysis/plots")
FEBASE = Path(f"{ROOT}/walrus/experiments/forget_eval_results")
WANT   = {20, 60}

GT_ROWS   = slice(90,  264)
PRED_ROWS = slice(270, 444)
ERR_ROWS  = slice(454, 628)
VX_COLS   = slice(33,  108)

def load_frames(folder):
    path = (FEBASE / folder / "cyl_sim/finetune/0/viz"
            / "realpdebench_cylinder_numerical/rollout_video"
            / "epoch52_rank0_rollout_test_batch0_realpdebench_cylinder_numerical.mp4")
    r = iio.get_reader(str(path))
    frames = {}
    for i, f in enumerate(r):
        if i in WANT: frames[i] = f
        if i > 60: break
    r.close()
    return frames

def get(frames, t, rows, cols):
    return np.rot90(frames[t][rows, cols, :], k=3)

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("Phase I\nep100",  "phase1_gn_lr5e5"),
    ("P2 ep20",         "gn_to_gn_step20"),
    ("P2 ep40",         "gn_to_gn_step40"),
    ("P2 ep60",         "gn_to_gn_step60"),
    ("P2 ep80",         "gn_to_gn_step80"),
    ("P2 ep100",        "gn_to_gn"),
]

all_frames = {label: load_frames(folder) for label, folder in CONDITIONS}

# ---------------------------------------------------------------------------
# Panel grid
# Row 0 = t=20, Row 1 = t=60
# Col 0 = GT (from Phase I ep100, all GTs are identical)
# Cols 1-12 = cond0_pred, cond0_err, cond1_pred, cond1_err, ...
# ---------------------------------------------------------------------------
TIMES = [20, 60]
panels = []  # panels[ri] = list of images across all columns

gt_src = all_frames["Phase I\nep100"]
for t in TIMES:
    row = [get(gt_src, t, GT_ROWS, VX_COLS)]
    for label, _ in CONDITIONS:
        row.append(get(all_frames[label], t, PRED_ROWS, VX_COLS))
        row.append(get(all_frames[label], t, ERR_ROWS,  VX_COLS))
    panels.append(row)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
ph, pw    = panels[0][0].shape[:2]
panel_scale = 0.008
gap         = 0.04    # within a pred+err pair
cond_gap    = 0.10    # between conditions
gt_gap      = 0.14    # between GT and first condition
label_w     = 0.20
col_label_h = 0.75
FSIZE       = 9

cell_w = pw * panel_scale
cell_h = ph * panel_scale

# Build column x-positions
col_lefts = [label_w]   # GT at col 0
x = label_w + cell_w + gt_gap
for ci in range(len(CONDITIONS)):
    col_lefts.append(x)           # pred
    col_lefts.append(x + cell_w + gap)  # err
    x += 2*cell_w + gap + cond_gap

fig_w = x - cond_gap + 0.10
t_gap = 0.10
fig_h = col_label_h + 2*cell_h + t_gap + 0.10

fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

# Row bottoms from top
row_y = [0.0, cell_h + t_gap]

# Column colors: GT=dark, conditions cycle through blue gradient
cond_blue = ["#1b6ca8", "#2e86c1", "#3498db", "#5dade2", "#85c1e9", "#aed6f1"]

for ri, (row, t) in enumerate(zip(panels, TIMES)):
    bottom = (fig_h - col_label_h - row_y[ri] - cell_h) / fig_h

    for ci, img in enumerate(row):
        left   = col_lefts[ci] / fig_w
        width  = cell_w / fig_w
        height = cell_h / fig_h

        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(img, aspect="equal", interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

        # Column titles on top row
        if ri == 0:
            if ci == 0:
                ax.set_title("Ground truth", fontsize=FSIZE, pad=4,
                             color="#333333", fontfamily="DejaVu Serif")
            else:
                # ci = 1..12; pair index = (ci-1)//2
                pair_i = (ci - 1) // 2
                is_pred = (ci - 1) % 2 == 0
                label = CONDITIONS[pair_i][0]
                color = cond_blue[pair_i] if pair_i < 1 else "#c0392b" if pair_i > 0 else cond_blue[0]
                # Phase I ep100 = blue; P2 = progressively red
                if pair_i == 0:
                    color = "#1b6ca8"
                else:
                    # Reddening: step from blue → red as forgetting progresses
                    reds   = ["#e67e22", "#e74c3c", "#c0392b", "#922b21", "#7b241c"]
                    color  = reds[pair_i - 1]
                if is_pred:
                    ax.set_title(f"{label}\npred", fontsize=FSIZE, pad=4,
                                 color=color, fontfamily="DejaVu Serif",
                                 linespacing=1.3)
                else:
                    ax.set_title("error", fontsize=FSIZE, pad=4,
                                 color=color, fontfamily="DejaVu Serif")

        # Row label: time step
        if ci == 0:
            ax.set_ylabel(f"$t={t}$", fontsize=FSIZE, labelpad=5,
                          fontfamily="DejaVu Serif")

for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_appendix_p2_forgetting.{ext}",
                bbox_inches="tight", pad_inches=0.08,
                dpi=300 if ext == "png" else None)
    print(f"Saved fig_appendix_p2_forgetting.{ext}")

plt.close(fig)
