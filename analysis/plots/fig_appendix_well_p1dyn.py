"""
fig_appendix_well_p1dyn.py
--------------------------
Phase I step-20 dynamics: Well benchmark rollout field panels for all 10 datasets.

For each dataset we show GT / Pred / Error for every field, using:
 - Subplot pixel bounds derived from a dummy make_video-layout figure
   (same size_multiplier=0.4, dpi=75, wspace/hspace as training code)
 - Frame t=30 from the actual rollout video

Datasets (all in forget_eval_results/phase1_dyn_step20):
  active_matter, MHD_64, trl2d, turbcool,
  acoustic_scattering_{discontinuous,inclusions,maze},
  helmholtz_staircase, planetswe, shear_flow
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import io
import os
from pathlib import Path

import imageio.v2 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import numpy as np

OUT    = Path(f"{ROOT}/realpdebench/analysis/plots")
FEBASE = Path(f"{ROOT}/walrus/experiments/forget_eval_results/phase1_dyn_step20")

T_SHOW = 30    # representative time step

# ---------------------------------------------------------------------------
# Dataset registry
# folder_name, viz_dir_name, n_fields, spatial_H, spatial_W, field_labels
# ---------------------------------------------------------------------------
DATASETS = [
    ("acoustic_scattering_discontinuous", "acoustic_scattering_discontinuous", 3,
     128, 128, ["pressure", "velocity x", "velocity y"]),
    ("acoustic_scattering_inclusions", "acoustic_scattering_inclusions", 3,
     128, 128, ["pressure", "velocity x", "velocity y"]),
    ("acoustic_scattering_maze", "acoustic_scattering_maze", 3,
     128, 128, ["pressure", "velocity x", "velocity y"]),
    ("active_matter", "active_matter", 11,
     256, 256, ["concentration", "vx", "vy",
                "Dxx", "Dxy", "Dyx", "Dyy",
                "Exx", "Exy", "Eyx", "Eyy"]),
    ("helmholtz_staircase", "helmholtz_staircase", 2,
     256, 64, ["pressure re", "pressure im"]),
    ("MHD_64", "MHD_64", 7,
     64, 64, ["density", "Bx", "By", "Bz", "vx", "vy", "vz"]),
    ("planetswe", "planetswe", 4,
     256, 512, ["height", "vel θ", "vel φ", "vorticity"]),
    ("shear_flow", "shear_flow", 5,
     256, 1024, ["pressure", "tracer", "vx", "vy", "vz"]),
    ("trl2d", "turbulent_radiative_layer_2D", 8,
     128, 512, ["density", "pressure", "vx", "vy",
                "temp", "Bx", "By", "Bz"]),
    ("turbcool", "turbulence_gravity_cooling", 6,
     128, 128, ["density", "pressure", "temperature",
                "vx", "vy", "vz"]),
]

SIZE_MUL = 0.4
DPI      = 75   # < 0.5 size_multiplier

ROW_NAMES  = ["GT", "Pred", "Error"]
ROW_COLORS = ["#1b6ca8", "#2e86c1", "#c0392b"]


def make_video_figsize(n_fields, h, w):
    """Replicate make_video's figsize computation."""
    ar = w / h
    base_w, base_h = 3.5, 2.5
    if ar > 1.5:
        wpf = base_w * min(ar / 1.5, 4.0)
        hpr = base_h
    elif ar < 0.67:
        wpf = base_w
        hpr = base_h * min(1.5 / ar, 4.0)
    else:
        wpf = base_w
        hpr = base_h
    fw = SIZE_MUL * (1.5 + wpf * n_fields)
    fh = SIZE_MUL * (1.2 + hpr * 3)
    return fw, fh


def get_subplot_bounds(n_fields, h, w):
    """
    Create a dummy make_video figure, get each imshow's pixel bbox.
    Returns dict {(row, col): (x0, x1, y0, y1)} in frame pixel coords
    (y measured from top of image).
    """
    fw, fh = make_video_figsize(n_fields, h, w)
    fig, axes = plt.subplots(3, n_fields, figsize=(fw, fh), dpi=DPI)
    if n_fields == 1:
        axes = axes[:, np.newaxis]

    try:
        le = fig.get_layout_engine()
        le.set(w_pad=0.08, h_pad=0.08, hspace=0.02, wspace=0.25)
    except Exception:
        fig.subplots_adjust(wspace=0.25, hspace=0.02)

    dummy = np.zeros((h, w))
    for ri in range(3):
        for ci in range(n_fields):
            im = axes[ri, ci].imshow(dummy, aspect="auto", cmap="viridis")
            # replicate add_colorbar(im, aspect=20, pad_fraction=0.5)
            divider = axes_grid1.make_axes_locatable(axes[ri, ci])
            width = axes_grid1.axes_size.AxesY(axes[ri, ci], aspect=1.0 / 20)
            pad   = axes_grid1.axes_size.Fraction(0.5, width)
            cax   = divider.append_axes("right", size=width, pad=pad)
            fig.colorbar(im, cax=cax)
            axes[ri, ci].tick_params(labelsize=0, length=2)

    fig.canvas.draw()

    bounds = {}
    fig_height_px = int(round(fh * DPI))
    for ri in range(3):
        for ci in range(n_fields):
            bb = axes[ri, ci].get_window_extent()
            x0 = int(round(bb.x0))
            x1 = int(round(bb.x1))
            # matplotlib y=0 is bottom; video y=0 is top
            y0_top = fig_height_px - int(round(bb.y1))
            y1_top = fig_height_px - int(round(bb.y0))
            bounds[(ri, ci)] = (x0, x1, y0_top, y1_top)

    plt.close(fig)
    return bounds


def load_frame(folder, viz_dir, t=T_SHOW):
    path = FEBASE / folder / "finetune/0/viz" / viz_dir / "rollout_video"
    vids = sorted(glob.glob(str(path / f"epoch52_rank0_rollout_test_batch0*.mp4")))
    if not vids:
        return None
    r = iio.get_reader(vids[0])
    frame = None
    for i, f in enumerate(r):
        if i == t:
            frame = f.copy()
            break
    r.close()
    return frame


# ---------------------------------------------------------------------------
# Extract per-field per-row panels from video frame
# ---------------------------------------------------------------------------
def extract_panels(frame, bounds, n_fields):
    """Returns dict {(row, col): image_crop}"""
    panels = {}
    if frame is None:
        return panels
    H, W = frame.shape[:2]
    for ri in range(3):
        for ci in range(n_fields):
            x0, x1, y0, y1 = bounds[(ri, ci)]
            # Clamp to frame dimensions
            x0 = max(0, x0); x1 = min(W, x1)
            y0 = max(0, y0); y1 = min(H, y1)
            if x1 > x0 and y1 > y0:
                panels[(ri, ci)] = frame[y0:y1, x0:x1, :]
    return panels


# ---------------------------------------------------------------------------
# Figure: one page per dataset (grid of fields × rows)
# ---------------------------------------------------------------------------
FSIZE = 8

for ds_folder, ds_viz, n_fields, h, w, field_labels in DATASETS:
    print(f"\nProcessing {ds_folder} ...")

    frame = load_frame(ds_folder, ds_viz)
    if frame is None:
        print(f"  SKIP: no video found")
        continue

    bounds = get_subplot_bounds(n_fields, h, w)
    panels = extract_panels(frame, bounds, n_fields)

    if not panels:
        print(f"  SKIP: no panels extracted")
        continue

    # Figure: 3 row-groups (GT / Pred / Error) × n_fields columns
    # Use a compact layout; each panel maintains its own aspect
    ph_max = max(p.shape[0] for p in panels.values())
    pw_max = max(p.shape[1] for p in panels.values())

    cell_h = 1.0    # inches per row
    cell_w = cell_h * (pw_max / ph_max)
    cell_w = min(cell_w, 1.8)  # cap width
    row_gap  = 0.08
    ds_label = 0.25
    col_label = 0.35
    field_label_h = 0.30

    fig_w = ds_label + n_fields * cell_w + (n_fields - 1) * 0.04 + 0.15
    fig_h = col_label + field_label_h + 3 * cell_h + 2 * row_gap + 0.10

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

    for ri, (row_name, row_color) in enumerate(zip(ROW_NAMES, ROW_COLORS)):
        y_from_top = col_label + field_label_h + ri * (cell_h + row_gap)
        for ci in range(n_fields):
            if (ri, ci) not in panels:
                continue
            img = panels[(ri, ci)]

            x_from_left = ds_label + ci * (cell_w + 0.04)
            left   = x_from_left / fig_w
            bottom = (fig_h - y_from_top - cell_h) / fig_h
            width  = cell_w / fig_w
            height = cell_h / fig_h

            ax = fig.add_axes([left, bottom, width, height])
            ax.imshow(img, aspect="auto", interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_edgecolor(row_color)

            # Field label on top row only
            if ri == 0:
                label = field_labels[ci] if ci < len(field_labels) else f"field {ci}"
                ax.set_title(label, fontsize=FSIZE, pad=3,
                             color="#333333", fontfamily="DejaVu Serif")

        # Row label on left
        row_mid_y = (fig_h - col_label - field_label_h
                     - ri * (cell_h + row_gap) - cell_h / 2) / fig_h
        fig.text(ds_label / fig_w - 0.01, row_mid_y, row_name,
                 ha="right", va="center", fontsize=FSIZE, color=row_color,
                 fontweight="bold", fontfamily="DejaVu Serif",
                 transform=fig.transFigure)

    # Dataset title
    fig.text(0.5, (fig_h - col_label / 2) / fig_h,
             ds_folder.replace("_", " ").title() + f"  ($t={T_SHOW}$)",
             ha="center", va="center", fontsize=FSIZE + 2, fontweight="bold",
             fontfamily="DejaVu Serif", transform=fig.transFigure)

    out_name = f"fig_appendix_well_{ds_folder}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{out_name}.{ext}",
                    bbox_inches="tight", pad_inches=0.06,
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  Saved {out_name}.png/.pdf")

print("\nDone.")
