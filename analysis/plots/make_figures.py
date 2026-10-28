"""
Publication-quality figures for the walrus cylinder sim-to-real transfer project.
Reads pkl files from experiment directories; hardcoded supplementary values from logs.
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    # text.usetex=True requires pdflatex; using mathtext CM instead (visually equivalent)
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern math via mathtext
    "axes.labelsize": 14, "font.size": 13,
    "legend.fontsize": 12, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

EXPBASE = f"{ROOT}/walrus/experiments"
OUT = f"{ROOT}/realpdebench/analysis/plots"

# Colour palette (accessible; checked for CVD)
C = {
    "gn_to_gn":      "#1b6ca8",   # deep blue
    "sw_to_gn_5e5":  "#c0392b",   # crimson
    "sw_to_gn_1e4":  "#e67e22",   # orange
    "replay10":      "#2ecc71",   # green (future)
    "rollout":       "#1b6ca8",
    "step1":         "#7fb3d3",   # lighter blue
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_training_curve(exp_name, key, prefix="rollout_valid"):
    """Return (epochs, values) from sequentially numbered pkl files."""
    exp_escaped = glob.escape(exp_name)
    pattern = os.path.join(EXPBASE, exp_escaped,
                           "finetune/0/viz/loss_dicts",
                           f"{prefix}_loss_dict_epoch*_rank0.pkl")
    pkls = sorted(glob.glob(pattern),
                  key=lambda p: int(p.split("epoch")[1].split("_rank")[0]))
    epochs, vals = [], []
    for p in pkls:
        ep = int(p.split("epoch")[1].split("_rank")[0])
        d = pickle.load(open(p, "rb"))
        # try both prefix patterns
        full_key = f"{prefix}_realpdebench_cylinder_real_cl/{key}"
        if full_key in d:
            epochs.append(ep)
            vals.append(d[full_key])
    return np.array(epochs), np.array(vals)


def load_test_metric(exp_name, key):
    p = os.path.join(EXPBASE, exp_name,
                     "finetune/0/viz/loss_dicts",
                     "rollout_test_loss_dict_epoch100_rank0.pkl")
    if not os.path.exists(p):
        return None
    d = pickle.load(open(p, "rb"))
    full_key = f"rollout_test_realpdebench_cylinder_real_cl/{key}"
    return d.get(full_key)


# ---------------------------------------------------------------------------
# Experiment map
# ---------------------------------------------------------------------------
EXP = {
    "gn_to_gn":     "walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
    "sw_to_gn_5e5": "walrus_sw_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
    "sw_to_gn_1e4": "walrus_sw_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001",
}

LABELS = {
    "gn_to_gn":     r"GN$\rightarrow$GN (Phase 2 best)",
    "sw_to_gn_5e5": r"SW$\rightarrow$GN, $\eta=5\times10^{-5}$",
    "sw_to_gn_1e4": r"SW$\rightarrow$GN, $\eta=10^{-4}$",
}


# ===========================================================================
# Figure 1 — gn_to_gn Phase 2 training curves (step-1 and rollout VRMSE)
# ===========================================================================
def fig1_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

    ep_r, vr = load_training_curve(EXP["gn_to_gn"], "full_VRMSE_T=all_mean", "rollout_valid")
    ep_s, vs = load_training_curve(EXP["gn_to_gn"], "full_VRMSE_T=all_mean", "valid")

    ax = axes[0]
    ax.plot(ep_s, vs, color=C["step1"], lw=1.8, marker="o", ms=3.5, label=r"Step-1 valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VRMSE")
    ax.set_title(r"Step-1 validation VRMSE")
    ax.legend(framealpha=0)

    ax = axes[1]
    ax.plot(ep_r, vr, color=C["rollout"], lw=1.8, marker="s", ms=3.5, label=r"60-step rollout valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VRMSE")
    ax.set_title(r"Rollout validation VRMSE")
    # mark best
    best_idx = np.argmin(vr)
    ax.axvline(ep_r[best_idx], color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.annotate(f"best ep {ep_r[best_idx]}\n$={vr[best_idx]:.3f}$",
                xy=(ep_r[best_idx], vr[best_idx]),
                xytext=(ep_r[best_idx] + 5, vr[best_idx] + 0.02),
                fontsize=10, color="grey",
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    ax.legend(framealpha=0)

    fig.suptitle(r"Phase 2 training: GN$\rightarrow$GN on real cylinder PIV", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_training_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_training_curves.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_training_curves")


# ===========================================================================
# Figure 2 — Test VRMSE bar chart: all Phase 2 models
# ===========================================================================
def fig2_test_comparison():
    models = ["sw_to_gn_5e5", "sw_to_gn_1e4", "gn_to_gn"]
    field_keys = {
        r"$u_x$": "velocity_x_VRMSE_T=all_mean",
        r"$u_y$": "velocity_y_VRMSE_T=all_mean",
        r"Full":  "full_VRMSE_T=all_mean",
    }
    field_colors = [C["sw_to_gn_5e5"], C["sw_to_gn_1e4"], C["gn_to_gn"]]

    n_groups = len(field_keys)
    n_bars = len(models)
    bar_w = 0.22
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4))
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w

    for i, (model, offset) in enumerate(zip(models, offsets)):
        vals = [load_test_metric(EXP[model], fk) for fk in field_keys.values()]
        bars = ax.bar(x + offset, vals, bar_w * 0.95, color=field_colors[i],
                      label=LABELS[model], alpha=0.88, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(list(field_keys.keys()))
    ax.set_ylabel("VRMSE (lower is better)")
    ax.set_title(r"Phase 2 test performance on real cylinder PIV (ep 100)")
    ax.legend(framealpha=0, loc="upper right", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_test_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig2_test_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_test_comparison")


# ===========================================================================
# Figure 3 — Phase 1 forgetting: turbcool step-1 VRMSE across SW checkpoints
#             (values from job 213483 logs — valid split, 300 batches)
# ===========================================================================
def fig3_phase1_forgetting():
    # Step-1 VRMSE (mean of 300 batches) from job 213483 logs
    # Phase 1 SW finetune on cylinder_numerical; evaluated on turbcool (valid split)
    epochs = [0,  20,    40,    51]
    vrmse  = [0.2151, 0.2026, 0.2240, 0.2242]
    labels = ["Pretrained", "ep 20", "ep 40", "ep 51 (best)"]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(epochs, vrmse, color=C["sw_to_gn_5e5"], lw=2, marker="o", ms=6, zorder=3)
    for ep, v in zip(epochs, vrmse):
        ax.annotate(f"{v:.4f}", (ep, v), textcoords="offset points",
                    xytext=(4, 4), fontsize=9)

    ax.set_xticks(epochs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_xlabel("Phase 1 checkpoint")
    ax.set_ylabel("Step-1 VRMSE (turbcool, valid)")
    ax.set_title("Phase 1 forgetting: turbulence-gravity-cooling\n"
                 "(step-1 VRMSE across Phase 1 SW checkpoints)")
    ax.set_ylim(0.18, 0.26)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_phase1_forgetting_turbcool.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig3_phase1_forgetting_turbcool.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_phase1_forgetting_turbcool")


# ===========================================================================
# Figure 4 — Phase 2 forgetting: gn_to_gn step-1 VRMSE on 3 Well datasets
#             (values from job 214878 logs — test split)
# ===========================================================================
def fig4_phase2_forgetting():
    datasets = ["Active matter", "MHD (64)", "Turbulent\nRadiative Layer (2D)"]
    # Step-1 VRMSE: mean of per-batch VRMSE from job 214878 logs (gn_to_gn Phase 2 ckpt)
    step1_vrmse = [0.1550, 0.2109, 0.4575]
    # Rollout test VRMSE: from rollout_test_loss_dict pkl (job 214878)
    rollout_vrmse = [3490.7, 0.330, 11.875]

    x = np.arange(len(datasets))
    bar_w = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    bars = ax.bar(x, step1_vrmse, bar_w * 2, color=C["gn_to_gn"], alpha=0.85,
                  edgecolor="white")
    for bar, v in zip(bars, step1_vrmse):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Step-1 VRMSE")
    ax.set_title("Step-1 VRMSE (test)")

    ax = axes[1]
    # rollout for active_matter is 3490 — put on log scale
    ax.bar(x, rollout_vrmse, bar_w * 2, color=C["gn_to_gn"], alpha=0.85,
           edgecolor="white")
    ax.set_yscale("log")
    for xi, v in zip(x, rollout_vrmse):
        ax.text(xi, v * 1.3, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Rollout VRMSE (log scale)")
    ax.set_title("60-step rollout VRMSE (test)")

    fig.suptitle(r"Phase 2 forgetting: GN$\rightarrow$GN (ep 100) on Well datasets"
                 "\n(no pretrained baseline yet; rollout destabilizes on active matter)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_phase2_forgetting.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig4_phase2_forgetting.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_phase2_forgetting")


# ===========================================================================
# Figure 5 — Training curve comparison: gn_to_gn rollout VRMSE + early stopping
# ===========================================================================
def fig5_rollout_with_pearsonr():
    ep_r, vr = load_training_curve(EXP["gn_to_gn"], "full_VRMSE_T=all_mean", "rollout_valid")
    ep_p, pr = load_training_curve(EXP["gn_to_gn"], "full_PearsonR_T=all_mean", "rollout_valid")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    l1, = ax1.plot(ep_r, vr, color=C["rollout"], lw=2, marker="o", ms=3.5,
                   label="Rollout VRMSE")
    l2, = ax2.plot(ep_p, pr, color="#27ae60", lw=2, marker="^", ms=3.5, ls="--",
                   label="Pearson $R$")

    best_vrmse = np.argmin(vr)
    ax1.axvline(ep_r[best_vrmse], color="grey", lw=0.8, ls=":", alpha=0.8)
    ax1.annotate(f"best ep {ep_r[best_vrmse]}",
                 xy=(ep_r[best_vrmse], vr[best_vrmse]),
                 xytext=(ep_r[best_vrmse] + 3, vr[best_vrmse] + 0.04),
                 fontsize=9, color="grey",
                 arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))

    # Add test result marker
    ax1.scatter([100], [0.307], color=C["rollout"], s=90, zorder=5, marker="*",
                label="Test VRMSE = 0.307")
    ax2.scatter([100], [0.9313], color="#27ae60", s=90, zorder=5, marker="*",
                label="Test $R$ = 0.931")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rollout VRMSE", color=C["rollout"])
    ax2.set_ylabel(r"Pearson $R$", color="#27ae60")
    ax1.tick_params(axis="y", labelcolor=C["rollout"])
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, framealpha=0, loc="upper right", fontsize=10)

    ax1.set_title(r"GN$\rightarrow$GN Phase 2: rollout VRMSE and Pearson $R$ vs epoch")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_rollout_vrmse_pearsonr.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig5_rollout_vrmse_pearsonr.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_rollout_vrmse_pearsonr")


# ===========================================================================
# Figure 6 — VRMSE vs time (T-intervals) for gn_to_gn test
# ===========================================================================
def fig6_temporal_breakdown():
    p = os.path.join(EXPBASE, EXP["gn_to_gn"],
                     "finetune/0/viz/loss_dicts",
                     "rollout_test_loss_dict_epoch100_rank0.pkl")
    d = pickle.load(open(p, "rb"))

    intervals = ["0:1", "1:2", "2:8", "8:22", "22:64"]
    interval_labels = ["T=1", "T=2", "T=2-8", "T=8-22", "T=22-64"]
    fields = {
        r"$u_x$": "velocity_x",
        r"$u_y$": "velocity_y",
        r"Full":  "full",
    }
    field_colors = [C["sw_to_gn_5e5"], C["sw_to_gn_1e4"], C["gn_to_gn"]]

    x = np.arange(len(intervals))
    bar_w = 0.25
    n = len(fields)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_w

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, field) in enumerate(fields.items()):
        vals = []
        for iv in intervals:
            key = f"rollout_test_realpdebench_cylinder_real_cl/{field}_VRMSE_T={iv}_mean"
            vals.append(d.get(key, np.nan))
        ax.bar(x + offsets[i], vals, bar_w * 0.92, label=label,
               color=field_colors[i], alpha=0.88, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(interval_labels)
    ax.set_xlabel("Autoregressive step interval")
    ax.set_ylabel("VRMSE")
    ax.set_title(r"GN$\rightarrow$GN test VRMSE by temporal interval (cylinder PIV, ep 100)")
    ax.legend(framealpha=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_temporal_breakdown.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig6_temporal_breakdown.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6_temporal_breakdown")


if __name__ == "__main__":
    fig1_training_curves()
    fig2_test_comparison()
    fig3_phase1_forgetting()
    fig4_phase2_forgetting()
    fig5_rollout_with_pearsonr()
    fig6_temporal_breakdown()
    print(f"\nAll figures saved to {OUT}")
