"""
Paper-ready figures and tables for active matter channel-wise catastrophic forgetting.

Phase 1 forgetting dynamics: how each physical field in active matter degrades during
cylinder fine-tuning (pretrained → ep5 → ep10 → ep15 → ep20), with no replay.

Outputs (all saved to OUT/):
  active_matter_forgetting_trajectories.{pdf,png}  — group-averaged VRMSE ratio line chart
  active_matter_channelwise_heatmap.{pdf,png}       — per-channel heatmap
  active_matter_forgetting.csv                      — full data table (raw + ratios)
  active_matter_forgetting_table.tex                — booktabs LaTeX table for paper
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import os
import csv
import pickle
import io

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Style — matches make_figures.py conventions
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

# Wong (2011) colorblind-safe palette — 4 groups
C_VEL  = "#0072B2"   # velocity (blue)
C_CONC = "#D55E00"   # concentration (vermilion)
C_D    = "#009E73"   # orientation tensor D (bluish green)
C_E    = "#CC79A7"   # stress tensor E (reddish purple)

EVAL = f"{ROOT}/walrus/experiments/forget_eval_results"
OUT  = f"{ROOT}/realpdebench/analysis/plots"

# ---------------------------------------------------------------------------
# Data — VRMSE values extracted from valid_loss_dict PKLs
# (pretrained = epoch 0; fine-tune checkpoints at epochs 5, 10, 15, 20)
# ---------------------------------------------------------------------------
EPOCHS = [0, 5, 10, 15, 20]

# Per-channel: (display_name, LaTeX_name, group, pre_VRMSE, [ratio_ep5, 10, 15, 20])
CHANNELS = [
    # concentration
    (r"$\phi$ (concentration)",  r"$\phi$",           "concentration", 0.03086, [4.22, 4.87, 5.28, 5.43]),
    # velocity
    (r"$v_x$",                   r"$v_x$",            "velocity",      0.00536, [5.98, 8.42, 9.91, 10.32]),
    (r"$v_y$",                   r"$v_y$",            "velocity",      0.00593, [3.96, 5.91, 7.36, 7.43]),
    # D tensor
    (r"$D_{xx}$",                r"$D_{xx}$",         "D",             0.01182, [2.29, 2.80, 3.17, 3.38]),
    (r"$D_{xy}$",                r"$D_{xy}$",         "D",             0.01005, [2.13, 2.84, 3.40, 3.63]),
    (r"$D_{yx}$",                r"$D_{yx}$",         "D",             0.01005, [2.13, 2.84, 3.41, 3.63]),
    (r"$D_{yy}$",                r"$D_{yy}$",         "D",             0.01260, [2.19, 2.72, 3.10, 3.31]),
    # E tensor
    (r"$E_{xx}$",                r"$E_{xx}$",         "E",             0.01861, [1.93, 2.60, 3.18, 3.39]),
    (r"$E_{xy}$",                r"$E_{xy}$",         "E",             0.01759, [2.02, 2.55, 2.97, 3.14]),
    (r"$E_{yx}$",                r"$E_{yx}$",         "E",             0.01758, [2.02, 2.55, 2.97, 3.14]),
    (r"$E_{yy}$",                r"$E_{yy}$",         "E",             0.01863, [1.95, 2.63, 3.23, 3.45]),
]

# Group-averaged VRMSE ratios (incl. pretrained = 1.0 baseline)
GROUPS = [
    (r"Velocity ($v_x$, $v_y$)",        C_VEL,  [1, 4.97, 7.17, 8.64, 8.88]),
    (r"Concentration ($\phi$)",          C_CONC, [1, 4.22, 4.87, 5.28, 5.43]),
    (r"Orientation tensor $\mathbf{D}$", C_D,    [1, 2.19, 2.80, 3.27, 3.47]),
    (r"Stress tensor $\mathbf{E}$",      C_E,    [1, 1.98, 2.58, 3.09, 3.28]),
]

# Group color lookup
GRP_COLOR = {"velocity": C_VEL, "concentration": C_CONC, "D": C_D, "E": C_E}


# ===========================================================================
# Figure A — Group-averaged forgetting trajectories
# ===========================================================================
def fig_trajectories():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    markers = ["o", "s", "^", "D"]
    # short name, point offset from the line endpoint (to stagger D/E which end close)
    label_info = {
        r"Velocity ($v_x$, $v_y$)":        ("Velocity",  0),
        r"Concentration ($\phi$)":          ("Conc.",     0),
        r"Orientation tensor $\mathbf{D}$": ("D tensor", +7),
        r"Stress tensor $\mathbf{E}$":      ("E tensor", -7),
    }

    for (label, color, ratios), marker in zip(GROUPS, markers):
        ax.plot(EPOCHS, ratios,
                color=color, lw=1.8, marker=marker, ms=5, zorder=3)
        short, dy_pt = label_info[label]
        ax.annotate(f"{short}  ×{ratios[-1]:.2f}",
                    xy=(EPOCHS[-1], ratios[-1]),
                    xytext=(6, dy_pt),
                    textcoords="offset points",
                    fontsize=9, color=color, va="center", ha="left",
                    annotation_clip=False)

    # Pretrained baseline — right-side label outside axes
    ax.axhline(1.0, color="grey", lw=0.9, ls="--", alpha=0.5, zorder=1)
    ax.annotate("Pretrained", xy=(EPOCHS[-1], 1.0), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, color="grey",
                va="center", ha="left", annotation_clip=False)

    ax.set_xlim(-1, 21)
    ax.set_ylim(0.5, 11.5)
    ax.set_xticks(EPOCHS)
    ax.set_xticklabels(["Pre.", "5", "10", "15", "20"])
    ax.set_ylabel("VRMSE ratio (vs. pretrained)")
    ax.set_xlabel("Phase-1 sim fine-tuning epoch")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f×"))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"active_matter_forgetting_trajectories.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


# ===========================================================================
# Figure B — Per-channel heatmap
# ===========================================================================
def fig_heatmap():
    # Build matrix: rows = channels, cols = ep5/10/15/20
    ratio_matrix = np.array([ch[4] for ch in CHANNELS], dtype=float)   # (11, 4)
    row_labels   = [ch[0] for ch in CHANNELS]

    # Group boundaries for horizontal separators
    groups_order = [ch[2] for ch in CHANNELS]
    sep_rows = []  # row indices AFTER which to draw a separator
    for i in range(1, len(groups_order)):
        if groups_order[i] != groups_order[i - 1]:
            sep_rows.append(i - 0.5)

    fig, ax = plt.subplots(figsize=(5.0, 4.8))

    # Colormap: sequential warm (Oranges/YlOrRd); vmin=1, vmax=11
    cmap = mpl.colormaps["YlOrRd"]
    vmin, vmax = 1.0, 11.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(ratio_matrix, aspect="auto", cmap=cmap,
                   norm=norm, interpolation="nearest")

    # Annotate cells
    for r in range(ratio_matrix.shape[0]):
        for c in range(ratio_matrix.shape[1]):
            val = ratio_matrix[r, c]
            # Use white text on dark cells, dark on light
            brightness = (val - vmin) / (vmax - vmin)
            tc = "white" if brightness > 0.55 else "#2a2a2a"
            ax.text(c, r, f"{val:.2f}×",
                    ha="center", va="center", fontsize=9,
                    color=tc, fontweight="bold" if brightness > 0.55 else "normal")

    # Group separator lines
    for yr in sep_rows:
        ax.axhline(yr, color="white", lw=2.0)

    # Group color bars on the left (colored strips showing field type)
    for i, ch in enumerate(CHANNELS):
        col = GRP_COLOR[ch[2]]
        ax.add_patch(mpl.patches.Rectangle(
            (-0.5 - 0.25, i - 0.5), 0.22, 1.0,
            color=col, clip_on=False, transform=ax.transData))

    ax.set_xticks(range(4))
    ax.set_xticklabels(["Ep 5", "Ep 10", "Ep 15", "Ep 20"])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.tick_params(left=False, bottom=False)
    ax.set_title("VRMSE ratio vs. pretrained Walrus", fontsize=11, pad=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("VRMSE ratio (×)", fontsize=10)
    cbar.set_ticks([1, 3, 5, 7, 9, 11])
    cbar.set_ticklabels(["1×", "3×", "5×", "7×", "9×", "11×"])

    # Legend patches for field groups
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=C_CONC, label=r"Concentration ($\phi$)"),
        Patch(facecolor=C_VEL,  label=r"Velocity ($v_x$, $v_y$)"),
        Patch(facecolor=C_D,    label=r"Orientation tensor $\mathbf{D}$"),
        Patch(facecolor=C_E,    label=r"Stress tensor $\mathbf{E}$"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              bbox_to_anchor=(0.0, -0.10), ncol=2, framealpha=0,
              fontsize=9, handlelength=1.0, handletextpad=0.4,
              bbox_transform=ax.transAxes)

    # Turn off grid (not meaningful for heatmap)
    ax.grid(False)
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"active_matter_channelwise_heatmap.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


# ===========================================================================
# CSV — full data table
# ===========================================================================
def save_csv():
    path = os.path.join(OUT, "active_matter_forgetting.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "group",
                    "pre_VRMSE",
                    "ep5_VRMSE", "ep5_ratio",
                    "ep10_VRMSE", "ep10_ratio",
                    "ep15_VRMSE", "ep15_ratio",
                    "ep20_VRMSE", "ep20_ratio"])
        for disp, latex, group, pre, ratios in CHANNELS:
            row = [latex.replace("$", "").replace("\\", ""), group, f"{pre:.6f}"]
            for r in ratios:
                row.append(f"{pre * r:.6f}")
                row.append(f"{r:.4f}")
            w.writerow(row)

    print(f"Saved {path}")

    # Also save group averages
    path2 = os.path.join(OUT, "active_matter_forgetting_groups.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "ep0_ratio", "ep5_ratio", "ep10_ratio", "ep15_ratio", "ep20_ratio"])
        group_names = {
            r"Velocity ($v_x$, $v_y$)":         "velocity",
            r"Concentration ($\phi$)":            "concentration",
            r"Orientation tensor $\mathbf{D}$":   "D_tensor",
            r"Stress tensor $\mathbf{E}$":        "E_tensor",
        }
        for (label, color, ratios) in GROUPS:
            w.writerow([group_names.get(label, label)] + [f"{r:.4f}" for r in ratios])
    print(f"Saved {path2}")


# ===========================================================================
# LaTeX table — booktabs style for paper
# ===========================================================================
def save_latex():
    path = os.path.join(OUT, "active_matter_forgetting_table.tex")

    group_map = {
        "concentration": (r"Concentration ($\phi$)",           r"\phi"),
        "velocity":      (r"Velocity ($v_x$, $v_y$)",         None),
        "D":             (r"Orientation tensor $\mathbf{D}$",  None),
        "E":             (r"Stress tensor $\mathbf{E}$",       None),
    }
    group_order = ["concentration", "velocity", "D", "E"]

    lines = []
    lines.append(r"% Active matter channel-wise VRMSE forgetting during cylinder fine-tuning.")
    lines.append(r"% Ratio = VRMSE at epoch / pretrained VRMSE. All evals: SamplewiseRevNorm,")
    lines.append(r"% debug_mode=True (first 20 validation batches, fixed order).")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Channel-wise VRMSE forgetting in \textit{active matter} "
                 r"during Phase~1 cylinder fine-tuning. Values are VRMSE ratios relative "
                 r"to the pretrained Walrus baseline; pretrained VRMSE is also shown for reference.}")
    lines.append(r"  \label{tab:active_matter_forgetting}")
    lines.append(r"  \begin{tabular}{l S[table-format=1.5] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2]}")
    lines.append(r"    \toprule")
    lines.append(r"    {Field} & {Pre VRMSE} & {Ep~5} & {Ep~10} & {Ep~15} & {Ep~20} \\")
    lines.append(r"    \midrule")

    prev_group = None
    for disp, latex, group, pre, ratios in CHANNELS:
        if group != prev_group:
            if prev_group is not None:
                lines.append(r"    \cmidrule(l){1-6}")
            grp_label = group_map[group][0]
            lines.append(f"    \\multicolumn{{6}}{{l}}{{\\textit{{{grp_label}}}}} \\\\")
            prev_group = group
        r5, r10, r15, r20 = ratios
        lines.append(f"    {latex} & {pre:.5f} & {r5:.2f} & {r10:.2f} & {r15:.2f} & {r20:.2f} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating active matter forgetting figures and tables...")
    fig_trajectories()
    fig_heatmap()
    save_csv()
    save_latex()
    print("Done.")
