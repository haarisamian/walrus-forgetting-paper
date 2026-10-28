"""
Figure: Cross-dataset channel-wise NRMSE degradation at Phase-1 epoch 20.

One horizontal lollipop per physical field, grouped by dataset.
Velocity channels rendered in blue; non-velocity in a neutral grey.
This makes the velocity-vs-rest story readable at a glance.

Saved to: fig_velocity_degradation.{pdf,png}
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import io
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Style — identical to forgetting_figures.py
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "axes.labelsize":     12,
    "font.size":          11,
    "xtick.labelsize":    11,
    "ytick.labelsize":    10,
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linewidth":     0.55,
})

EVAL = f"{ROOT}/walrus/experiments/forget_eval_results"
OUT  = f"{ROOT}/realpdebench/analysis/plots"

C_VEL    = "#0072B2"   # velocity (Wong blue — matches active matter file)
C_OTHER  = "#888888"   # non-velocity
C_REF    = "#444444"   # x=1 reference line

# ---------------------------------------------------------------------------
# PKL loader (CPU-safe)
# ---------------------------------------------------------------------------
class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)

def _load_dict(folder, ds_dir):
    pkls = glob.glob(
        os.path.join(EVAL, folder, ds_dir,
                     "finetune/0/viz/loss_dicts/valid_loss_dict_*.pkl")
    )
    if not pkls:
        return {}
    with open(pkls[0], "rb") as f:
        return _CPUUnpickler(f).load()

def get_nrmse(folder, ds_dir, key_suffix):
    """Return float NRMSE for key whose last segment matches key_suffix, or None.
    Keys look like: valid_<ds_name>/<field>_NRMSE_T=all_mean
    """
    d = _load_dict(folder, ds_dir)
    target = key_suffix + "_NRMSE_T=all_mean"
    for k, v in d.items():
        if k.endswith(target) and "full_NRMSE" not in k:
            return float(v)
    return None

# ---------------------------------------------------------------------------
# Dataset + channel definitions
# key      = field name in the pkl  (without _NRMSE_T=all_mean)
# label    = display label
# is_vel   = True if this is a velocity field
# ---------------------------------------------------------------------------
DATASETS = [
    # (display_name, ds_dir, [( key, label, is_vel ), ...])
    ("Helmholtz\nStaircase", "helmholtz_staircase", [
        ("pressure_im",  r"$p$ (Im)",      False),
        ("pressure_re",  r"$p$ (Re)",      False),
    ]),
    ("Planet\nSWE",  "planetswe", [
        ("height",           r"$h$",                 False),
        ("velocity_phi",     r"$v_\phi$",            True),
        ("velocity_theta",   r"$v_\theta$",          True),
    ]),
    ("Acoustic\nDisc.", "acoustic_scattering_discontinuous", [
        ("pressure",    r"$p$",   False),
        ("velocity_x",  r"$v_x$", True),
        ("velocity_y",  r"$v_y$", True),
    ]),
    ("Acoustic\nIncl.", "acoustic_scattering_inclusions", [
        ("pressure",    r"$p$",   False),
        ("velocity_x",  r"$v_x$", True),
        ("velocity_y",  r"$v_y$", True),
    ]),
    ("Acoustic\nMaze", "acoustic_scattering_maze", [
        ("pressure",    r"$p$",   False),
        ("velocity_x",  r"$v_x$", True),
        ("velocity_y",  r"$v_y$", True),
    ]),
    ("Active\nMatter", "active_matter", [
        ("D_xx",         r"$D_{xx}$", False),
        ("D_xy",         r"$D_{xy}$", False),
        ("D_yx",         r"$D_{yx}$", False),
        ("D_yy",         r"$D_{yy}$", False),
        ("E_xx",         r"$E_{xx}$", False),
        ("E_xy",         r"$E_{xy}$", False),
        ("E_yx",         r"$E_{yx}$", False),
        ("E_yy",         r"$E_{yy}$", False),
        ("velocity_x",   r"$v_x$",    True),
        ("velocity_y",   r"$v_y$",    True),
        # concentration excluded: pretrained NRMSE ≈ 0 → ratio unreliable
    ]),
    ("Shear\nFlow", "shear_flow", [
        ("tracer",      r"tracer",  False),
        ("pressure",    r"$p$",     False),
        ("velocity_x",  r"$v_x$",  True),
        ("velocity_y",  r"$v_y$",  True),
    ]),
    ("TRL-2D", "trl2d", [
        ("density",     r"$\rho$",  False),
        ("pressure",    r"$p$",     False),
        ("velocity_x",  r"$v_x$",  True),
        ("velocity_y",  r"$v_y$",  True),
    ]),
    ("Turbcool", "turbcool", [
        ("density",      r"$\rho$",  False),
        ("pressure",     r"$p$",     False),
        ("temperature",  r"$T$",     False),
        ("velocity_x",   r"$v_x$",  True),
        ("velocity_y",   r"$v_y$",  True),
        ("velocity_z",   r"$v_z$",  True),
    ]),
    ("MHD-64", "MHD_64", [
        ("density",           r"$\rho$",  False),
        ("magnetic_field_x",  r"$B_x$",   False),
        ("magnetic_field_y",  r"$B_y$",   False),
        ("magnetic_field_z",  r"$B_z$",   False),
        ("velocity_x",        r"$v_x$",   True),
        ("velocity_y",        r"$v_y$",   True),
        ("velocity_z",        r"$v_z$",   True),
    ]),
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading pretrained and phase1_dyn_step20 NRMSEs …")
rows = []   # list of (ds_display, ch_label, is_vel, pre, ep20, ratio)
for ds_label, ds_dir, channels in DATASETS:
    for key, ch_label, is_vel in channels:
        pre  = get_nrmse("pretrained",       ds_dir, key)
        ep20 = get_nrmse("phase1_dyn_step20", ds_dir, key)
        if pre is None or ep20 is None or pre < 1e-8:
            print(f"  SKIP {ds_dir}/{key}: pre={pre}, ep20={ep20}")
            continue
        ratio = ep20 / pre
        rows.append((ds_label, ch_label, is_vel, pre, ep20, ratio))
        tag = "VEL" if is_vel else "   "
        print(f"  [{tag}] {ds_dir:35s}  {key:22s}: pre={pre:.5f}  ep20={ep20:.5f}  ×{ratio:.2f}")

# ---------------------------------------------------------------------------
# Build y-axis layout (datasets top → bottom, channels within by ratio desc)
# ---------------------------------------------------------------------------
GAP = 0.8   # extra space between dataset groups

y_pos   = []
y_ticks = []
y_grp_labels = []   # (y_center, ds_label, has_vel)

y = 0.0
ds_prev = None
ds_start_y = 0.0
ds_rows = []
grp_order = []   # order datasets appear in DATASETS list

seen = []   # (ds_label, ch_label, is_vel, pre, ep20, ratio)
for ds_label, ds_dir, channels in DATASETS:
    grp = [(ch_label, is_vel, pre, ep20, ratio)
           for (dl, ch_label, is_vel, pre, ep20, ratio) in rows
           if dl == ds_label]
    if not grp:
        continue
    grp_order.append((ds_label, grp))

y = 0.0
for ds_label, grp in grp_order:
    # Sort within group: non-vel ascending, then vel ascending
    # (vel channels will end up together, slightly to the right of non-vel)
    def sort_key(item):
        ch_label, is_vel, pre, ep20, ratio = item
        return (int(is_vel), ratio)

    grp_sorted = sorted(grp, key=sort_key)
    y_start = y
    for ch_label, is_vel, pre, ep20, ratio in grp_sorted:
        y_ticks.append(y)
        seen.append((ds_label, ch_label, is_vel, pre, ep20, ratio))
        y += 1.0

    y_center = (y_start + y - 1) / 2
    has_vel = any(iv for _, iv, *_ in grp)
    y_grp_labels.append((y_center, ds_label, has_vel))
    y += GAP

total_h = y - GAP


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def fig_velocity_degradation():
    fig_h = max(6.0, 0.28 * len(seen) + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))

    # Reference line at ×1
    ax.axvline(1, color=C_REF, lw=0.85, ls="--", zorder=1, alpha=0.7)

    for yi, (ds_label, ch_label, is_vel, pre, ep20, ratio) in zip(y_ticks, seen):
        col = C_VEL if is_vel else C_OTHER
        lw  = 1.4   if is_vel else 0.9
        ms  = 6.5   if is_vel else 5.0
        alpha_stem = 0.65 if is_vel else 0.40

        # Stem
        ax.plot([1, ratio], [yi, yi], color=col, lw=lw, alpha=alpha_stem, zorder=2)
        # Dot
        marker = "o" if is_vel else "o"
        fc     = col if is_vel else "none"      # filled for vel, open for non-vel
        ec     = col
        ax.plot(ratio, yi, marker=marker, ms=ms, color=ec,
                mfc=fc, mew=1.3, zorder=3)

        # Value annotation right of dot (only when ratio > 3 to avoid clutter)
        if ratio >= 4.0:
            ax.text(ratio + 0.5, yi, f"×{ratio:.1f}",
                    va="center", ha="left", fontsize=7.5,
                    color=col, alpha=0.85)

    # Dataset group labels on the left (outside the plot)
    for y_center, ds_label, has_vel in y_grp_labels:
        ax.text(-0.5, y_center, ds_label,
                va="center", ha="right", fontsize=8.5,
                color="#222222", fontweight="normal",
                transform=ax.get_yaxis_transform())

    # Faint horizontal separator lines between dataset groups
    # (drawn between last channel of one group and first of next)
    for i in range(1, len(grp_order)):
        # find the gap position
        prev_ds = grp_order[i-1][0]
        curr_ds = grp_order[i][0]
        # last y of prev group
        prev_ys = [yi for yi, (dl, *_) in zip(y_ticks, seen) if dl == prev_ds]
        curr_ys = [yi for yi, (dl, *_) in zip(y_ticks, seen) if dl == curr_ds]
        if prev_ys and curr_ys:
            sep_y = (max(prev_ys) + min(curr_ys)) / 2
            ax.axhline(sep_y, color="#cccccc", lw=0.7, zorder=0)

    # Axes
    max_ratio = max(r for *_, r in seen)
    ax.set_xlim(0.5, max_ratio * 1.08)
    ax.set_ylim(-0.8, max(y_ticks) + 0.8)
    ax.invert_yaxis()

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([ch for _, ch, *_ in seen], fontsize=9.5)
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlabel("Degradation factor at Phase-1 ep 20  (NRMSE / pretrained NRMSE)", fontsize=11)
    ax.xaxis.set_label_position("bottom")

    # Inline legend — placed top-right inside plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=C_VEL,  mfc=C_VEL,  ms=6, lw=1.2, label="Velocity"),
        Line2D([0], [0], marker="o", color=C_OTHER, mfc="none", ms=5, lw=0.8,
               mew=1.2, label="Other fields"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              framealpha=0.9, fontsize=9, handletextpad=0.5,
              borderpad=0.6, labelspacing=0.4)

    # Extra left margin for group labels
    fig.subplots_adjust(left=0.22, right=0.97, top=0.98, bottom=0.07)

    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig_velocity_degradation.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_velocity_degradation()
    print("Done.")


# ============================================================================
# Figure 2 — Velocity vs non-velocity scatter: one point per dataset
# ============================================================================
def fig_vel_vs_nonvel():
    """
    Scatter: x = mean non-velocity degradation, y = mean velocity degradation.
    One point per dataset. Points above y=x line → velocity degrades more.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    # Per-dataset means (computed from full data above)
    DATASETS_SUMMARY = [
        # (label, vel_mean, nonvel_mean, ds_color)
        ("Helmholtz",     None,  35.55, "#9467bd"),   # no velocity
        ("PlanetSWE",     28.32, 25.18, "#17becf"),
        ("Ac. Disc.",     31.19, 29.45, "#ff7f0e"),
        ("Ac. Incl.",     12.97,  9.67, "#e377c2"),
        ("Ac. Maze",      11.37,  8.51, "#bcbd22"),
        ("Active Matter",  8.88,  3.38, "#e67e22"),
        ("Shear Flow",     4.07,  3.26, "#8c564b"),
        ("TRL-2D",         2.71,  1.80, "#c0392b"),
        ("Turbcool",       1.86,  1.85, "#27ae60"),
        ("MHD-64",         1.07,  1.07, "#1b6ca8"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    maxval = 35
    ax.plot([1, maxval], [1, maxval], color="#aaaaaa", lw=1.0, ls="--", zorder=1)
    ax.text(maxval*0.75, maxval*0.68, "vel = other", fontsize=8.5,
            color="#888888", rotation=45, va="center", ha="center")

    for label, vel, nonvel, col in DATASETS_SUMMARY:
        if vel is None:
            # Helmholtz — project onto x-axis at top with arrow
            ax.annotate("Helmholtz\n(no velocity)", xy=(nonvel, nonvel),
                        xytext=(nonvel - 2, nonvel + 2.5),
                        fontsize=7.5, color="#9467bd", ha="right",
                        arrowprops=dict(arrowstyle="-", color="#9467bd", lw=0.8))
            ax.scatter([nonvel], [nonvel], marker="x", s=55, color="#9467bd",
                       linewidths=1.5, zorder=4)
            continue

        ax.scatter([nonvel], [vel], color=col, s=52, zorder=4, clip_on=False)

        # label placement: offset to avoid crowding
        offsets = {
            "PlanetSWE":    (+1.0, +0.5),
            "Ac. Disc.":    (-1.5, +1.5),
            "Ac. Incl.":    (+0.5, +0.8),
            "Ac. Maze":     (+0.5, -1.2),
            "Active Matter":(+0.6, +0.6),
            "Shear Flow":   (+0.3, -1.1),
            "TRL-2D":       (+0.3, +0.4),
            "Turbcool":     (+0.3, -0.7),
            "MHD-64":       (+0.2, +0.3),
        }
        dx, dy = offsets.get(label, (0.3, 0.3))
        ax.annotate(label, xy=(nonvel, vel),
                    xytext=(nonvel + dx, vel + dy),
                    fontsize=8, color=col, va="center",
                    arrowprops=None)

    # Shade "velocity worse" region
    ax.fill_between([1, maxval], [1, maxval], [maxval, maxval],
                    color="#0072B2", alpha=0.04, zorder=0)
    ax.text(3.5, 28, "velocity\nworse", fontsize=8, color="#0072B2",
            alpha=0.6, va="center", ha="center")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.9, maxval * 1.15)
    ax.set_ylim(0.9, maxval * 1.15)
    ax.set_xlabel("Mean non-velocity degradation (×)")
    ax.set_ylabel("Mean velocity degradation (×)")

    from matplotlib.ticker import LogFormatter, LogLocator
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"×{x:.0f}" if x>=2 else f"×{x:.1f}"))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"×{x:.0f}" if x>=2 else f"×{x:.1f}"))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"fig_vel_vs_nonvel.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    fig_velocity_degradation()
    fig_vel_vs_nonvel()
    print("Done.")
