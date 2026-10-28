"""
fig_appendix_well_p1dyn_v2.py
------------------------------
Well benchmark rollout panels — all 10 datasets.

Usage:
  python fig_appendix_well_p1dyn_v2.py [dump_subdir] [title_suffix] [out_suffix]

  dump_subdir  — subdirectory under experiments/ containing dumps
                 default: dump_well_p1dyn_step20
  title_suffix — appended to dataset name in figure title
                 default: "Phase I epoch 20"
  out_suffix   — appended to output filenames, e.g. "_pretrained"
                 default: ""

Reads: experiments/{dump_subdir}/{ds_name}/finetune/0/viz/{ds_name}/
           full_trajectory_dumps/ypred_*_0.npy  [B, T, H, W, (D,) C]

Layout:
  - Most datasets: horizontal time layout
      3 rows (GT / Pred / Error) × time groups side-by-side
  - Many-field datasets (MHD_64, active_matter): vertical time layout
      time groups stacked vertically, each with 3 sub-rows, n_fields columns
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
DUMP_SUBDIR   = sys.argv[1] if len(sys.argv) > 1 else "dump_well_p1dyn_step20"
TITLE_SUFFIX  = sys.argv[2] if len(sys.argv) > 2 else "Phase I epoch 20"
OUT_SUFFIX    = sys.argv[3] if len(sys.argv) > 3 else ""

OUT      = Path(f"{ROOT}/realpdebench/analysis/plots")
DUMPBASE = Path(f"{ROOT}/walrus/experiments") / DUMP_SUBDIR

# ---------------------------------------------------------------------------
TIMES = [1, 20, 40, 60]   # rollout step indices (0 = first prediction = 1 step after initial cond.)

# Datasets that get vertical time stacking (too many fields for horizontal)
VERTICAL_DATASETS = {"MHD_64", "active_matter"}

DATASETS = [
    ("acoustic_scattering_discontinuous",
     ["pressure", "velocity x", "velocity y"]),
    ("acoustic_scattering_inclusions",
     ["pressure", "velocity x", "velocity y"]),
    ("acoustic_scattering_maze",
     ["pressure", "velocity x", "velocity y"]),
    ("active_matter",
     ["concentration", "v_x", "v_y",
      "D_xx", "D_xy", "D_yx", "D_yy",
      "E_xx", "E_xy", "E_yx", "E_yy"]),
    ("helmholtz_staircase",
     ["pressure (Re)", "pressure (Im)"]),
    ("MHD_64",
     ["density", "B_x", "B_y", "B_z", "v_x", "v_y", "v_z"]),
    ("planetswe",
     ["height", r"vel $\theta$", r"vel $\phi$", "vorticity"]),
    ("shear_flow",
     ["pressure", "tracer", "v_x", "v_y", "v_z"]),
    ("turbulent_radiative_layer_2D",
     ["density", "pressure", "v_x", "v_y",
      "temperature", "B_x", "B_y", "B_z"]),
    ("turbulence_gravity_cooling",
     ["density", "pressure", "temperature", "v_x", "v_y", "v_z"]),
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_dump(ds_name):
    dump_dir = DUMPBASE / ds_name / "finetune/0/viz" / ds_name / "full_trajectory_dumps"
    if not dump_dir.exists():
        print(f"  SKIP {ds_name}: dump dir missing")
        return None, None
    pred_files = sorted(glob.glob(str(dump_dir / "ypred_*_0.npy")))
    ref_files  = sorted(glob.glob(str(dump_dir / "yref_*_0.npy")))
    if not pred_files or not ref_files:
        print(f"  SKIP {ds_name}: no dump files")
        return None, None
    ypred = np.load(pred_files[0])
    yref  = np.load(ref_files[0])
    # Squeeze batch dim
    if ypred.ndim >= 5:
        ypred = ypred[0]; yref = yref[0]
    # Collapse extra spatial dims (size-1 → squeeze; otherwise take central slice)
    while ypred.ndim > 4:
        if ypred.shape[-2] == 1:
            ypred = ypred[..., 0, :]; yref = yref[..., 0, :]
        else:
            mid = ypred.shape[-2] // 2
            ypred = ypred[..., mid, :]; yref = yref[..., mid, :]
    return ypred, yref   # [T, H, W, C]

# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------
def field_clim(arr):
    return float(np.nanpercentile(arr, 1)), float(np.nanpercentile(arr, 99))

# ---------------------------------------------------------------------------
FSIZE      = 10
ROW_LABELS = ["GT", "Pred", "Error"]
ROW_COLORS = ["#2d6a2d", "#1b6ca8", "#c0392b"]

# Rotate field labels 90° when cells are too narrow for horizontal text.
# At FSIZE=10, a typical 9-char label is ~0.65" wide, so rotate below that.
NARROW_CELL_THRESHOLD = 0.70   # inches

def add_field_label(ax, label, cell_w):
    """Place a field-name header above ax, rotating if the cell is narrow."""
    if cell_w < NARROW_CELL_THRESHOLD:
        ax.text(0.5, 1.03, label,
                transform=ax.transAxes,
                rotation=90, ha="center", va="bottom",
                fontsize=FSIZE - 2, color="#222222")   # 8pt rotated — fits even long labels
    else:
        ax.set_title(label, fontsize=FSIZE, pad=3, color="#222222")

# ---------------------------------------------------------------------------
# HORIZONTAL layout: 3 rows × (n_t time groups × n_fields) cols
# Used for most datasets
# ---------------------------------------------------------------------------
def build_horizontal(ds_name, actual_labels, ts, ypred, yref, n_fields, H, W):
    aspect   = W / H
    cell_h   = 0.55
    cell_w   = min(cell_h * aspect, 1.5)

    col_gap    = 0.04
    row_gap    = 0.03
    t_gap      = 0.14
    row_lbl_w  = 0.40
    t_lbl_h    = 0.42   # band above field headers for t-labels
    # Rotated labels need more vertical space than horizontal ones
    field_hdr_h= 1.05 if cell_w < NARROW_CELL_THRESHOLD else 0.32
    fig_margin = 0.10
    n_t = len(ts)

    # Shared error colorscale per field across all displayed times
    err_vmax = []
    for ci in range(n_fields):
        all_err = np.concatenate([
            np.abs(ypred[t, :, :, ci] - yref[t, :, :, ci]).ravel() for t in ts
        ])
        err_vmax.append(float(np.nanpercentile(all_err, 99)))

    t_group_w = n_fields * cell_w + (n_fields - 1) * col_gap
    fig_w = fig_margin + row_lbl_w + n_t * t_group_w + (n_t - 1) * t_gap + fig_margin
    fig_h = (fig_margin + t_lbl_h + field_hdr_h
             + 3 * cell_h + 2 * row_gap + fig_margin)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    content_top = fig_margin + t_lbl_h + field_hdr_h

    for ti, t in enumerate(ts):
        x_group = fig_margin + row_lbl_w + ti * (t_group_w + t_gap)
        # t-label: at top of figure, above field names
        t_cx = (x_group + t_group_w / 2) / fig_w
        t_cy = (fig_h - fig_margin - t_lbl_h / 2) / fig_h
        fig.text(t_cx, t_cy, f"$t = {t}$",
                 ha="center", va="center", fontsize=FSIZE + 1,
                 fontweight="bold", color="#444444",
                 transform=fig.transFigure)

        for si, (rl, rc) in enumerate(zip(ROW_LABELS, ROW_COLORS)):
            y_top = content_top + si * (cell_h + row_gap)
            if ti == 0:
                row_mid_y = (fig_h - y_top - cell_h / 2) / fig_h
                fig.text((fig_margin + row_lbl_w * 0.92) / fig_w, row_mid_y,
                         rl, ha="right", va="center",
                         fontsize=FSIZE, color=rc, fontweight="bold",
                         transform=fig.transFigure)

            for ci in range(n_fields):
                gt = yref[t, :, :, ci];  pr = ypred[t, :, :, ci]
                if si == 0:
                    data, cmap = gt, "viridis"
                    vmin, vmax = field_clim(gt)
                elif si == 1:
                    data, cmap = pr, "viridis"
                    vmin, vmax = field_clim(gt)
                else:
                    data, cmap = pr - gt, "RdBu_r"
                    v = err_vmax[ci]   # shared scale across all times for this field
                    vmin, vmax = -v, v

                left   = (x_group + ci * (cell_w + col_gap)) / fig_w
                bottom = (fig_h - y_top - cell_h) / fig_h
                ax = fig.add_axes([left, bottom, cell_w / fig_w, cell_h / fig_h])
                ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                          aspect="auto", interpolation="bilinear", origin="upper")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5); sp.set_edgecolor(rc)
                if si == 0:
                    add_field_label(ax, actual_labels[ci], cell_w)

    return fig

# ---------------------------------------------------------------------------
# VERTICAL layout: time groups stacked vertically, each 3 sub-rows × n_fields cols
# Used for MHD_64, active_matter
# ---------------------------------------------------------------------------
def build_vertical(ds_name, actual_labels, ts, ypred, yref, n_fields, H, W):
    aspect   = W / H
    cell_h   = 0.50
    cell_w   = min(cell_h * aspect, 1.2)

    # Shared error colorscale per field across all displayed times
    err_vmax = []
    for ci in range(n_fields):
        all_err = np.concatenate([
            np.abs(ypred[t, :, :, ci] - yref[t, :, :, ci]).ravel() for t in ts
        ])
        err_vmax.append(float(np.nanpercentile(all_err, 99)))

    col_gap    = 0.04
    row_gap    = 0.02   # between GT/Pred/Error sub-rows
    t_gap      = 0.22   # between time groups (vertical)
    row_lbl_w  = 0.40
    t_lbl_h    = 0.36   # above each time group
    field_hdr_h= 1.05 if cell_w < NARROW_CELL_THRESHOLD else 0.30  # extra room for rotated labels
    fig_margin = 0.10
    n_t = len(ts)

    fig_w = fig_margin + row_lbl_w + n_fields * cell_w + (n_fields - 1) * col_gap + fig_margin

    # First group gets field headers; subsequent groups get t-label + 3 rows
    one_group_h = t_lbl_h + 3 * cell_h + 2 * row_gap
    fig_h = (fig_margin + field_hdr_h
             + n_t * one_group_h + (n_t - 1) * t_gap + fig_margin)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)

    for ti, t in enumerate(ts):
        # y-start (from top of figure) of this time group's t-label
        y_group_top = (fig_margin + field_hdr_h
                       + ti * (one_group_h + t_gap))

        # t-label
        t_cy = (fig_h - y_group_top - t_lbl_h / 2) / fig_h
        fig.text(0.5, t_cy, f"$t = {t}$",
                 ha="center", va="center", fontsize=FSIZE + 1,
                 fontweight="bold", color="#444444",
                 transform=fig.transFigure)

        # 3 sub-rows (GT / Pred / Error)
        for si, (rl, rc) in enumerate(zip(ROW_LABELS, ROW_COLORS)):
            y_top = y_group_top + t_lbl_h + si * (cell_h + row_gap)
            row_mid_y = (fig_h - y_top - cell_h / 2) / fig_h
            fig.text((fig_margin + row_lbl_w * 0.92) / fig_w, row_mid_y,
                     rl, ha="right", va="center",
                     fontsize=FSIZE, color=rc, fontweight="bold",
                     transform=fig.transFigure)

            for ci in range(n_fields):
                gt = yref[t, :, :, ci];  pr = ypred[t, :, :, ci]
                if si == 0:
                    data, cmap = gt, "viridis"
                    vmin, vmax = field_clim(gt)
                elif si == 1:
                    data, cmap = pr, "viridis"
                    vmin, vmax = field_clim(gt)
                else:
                    data, cmap = pr - gt, "RdBu_r"
                    v = err_vmax[ci]   # shared scale across all times for this field
                    vmin, vmax = -v, v

                x_left = fig_margin + row_lbl_w + ci * (cell_w + col_gap)
                left   = x_left / fig_w
                bottom = (fig_h - y_top - cell_h) / fig_h
                ax = fig.add_axes([left, bottom, cell_w / fig_w, cell_h / fig_h])
                ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                          aspect="auto", interpolation="bilinear", origin="upper")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5); sp.set_edgecolor(rc)
                # Field headers: placed via fig.text() in the field_hdr_h band
                # (NOT relative to the axes, to avoid bleeding into the t-label band)
                if ti == 0 and si == 0:
                    lbl_x = (fig_margin + row_lbl_w
                             + ci * (cell_w + col_gap) + cell_w / 2) / fig_w
                    lbl_y = (fig_h - fig_margin - field_hdr_h / 2) / fig_h
                    if cell_w < NARROW_CELL_THRESHOLD:
                        fig.text(lbl_x, lbl_y, actual_labels[ci],
                                 rotation=90, ha="center", va="center",
                                 fontsize=FSIZE - 2, color="#222222",
                                 transform=fig.transFigure)
                    else:
                        fig.text(lbl_x, lbl_y, actual_labels[ci],
                                 ha="center", va="center",
                                 fontsize=FSIZE, color="#222222",
                                 transform=fig.transFigure)

    return fig

# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    "MHD_64":                         "MHD-64",
    "turbulent_radiative_layer_2D":   "Turbulent Radiative Layer 2D",
    "turbulence_gravity_cooling":     "Turbulence Gravity Cooling",
    "acoustic_scattering_discontinuous": "Acoustic Scattering (Discontinuous)",
    "acoustic_scattering_inclusions":    "Acoustic Scattering (Inclusions)",
    "acoustic_scattering_maze":          "Acoustic Scattering (Maze)",
    "active_matter":    "Active Matter",
    "helmholtz_staircase": "Helmholtz Staircase",
    "planetswe":        "PlanetSWE",
    "shear_flow":       "Shear Flow",
}

def _add_title(fig, ds_name, fig_h, fig_margin, title_h):
    title_y = (fig_h - fig_margin - title_h / 2) / fig_h
    ds_display = DISPLAY_NAMES.get(ds_name, ds_name.replace("_", " ").title())
    fig.text(0.5, title_y,
             f"{ds_display} — {TITLE_SUFFIX}",
             ha="center", va="center",
             fontsize=FSIZE + 2, fontweight="bold",
             color="#111111", transform=fig.transFigure)

def build_figure(ds_name, field_labels, ypred, yref):
    n_fields = ypred.shape[-1]
    actual_labels = (field_labels[:n_fields]
                     + [f"field {i}" for i in range(len(field_labels), n_fields)])
    T = ypred.shape[0]
    ts = [t for t in TIMES if t < T]
    if not ts:
        print(f"  SKIP {ds_name}: trajectory too short (T={T})")
        return
    H, W = ypred.shape[1], ypred.shape[2]

    if ds_name in VERTICAL_DATASETS:
        fig = build_vertical(ds_name, actual_labels, ts, ypred, yref, n_fields, H, W)
    else:
        fig = build_horizontal(ds_name, actual_labels, ts, ypred, yref, n_fields, H, W)

    out_stem = f"fig_appendix_well_{ds_name}{OUT_SUFFIX}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{out_stem}.{ext}",
                    bbox_inches="tight", pad_inches=0.06,
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"  Saved {out_stem}.png / .pdf")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for ds_name, field_labels in DATASETS:
    print(f"\nProcessing {ds_name} ...")
    ypred, yref = load_dump(ds_name)
    if ypred is None:
        continue
    print(f"  Loaded: ypred {ypred.shape}, yref {yref.shape}")
    build_figure(ds_name, field_labels, ypred, yref)

print("\nDone.")
