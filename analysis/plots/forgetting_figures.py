"""
Publication-quality forgetting figures for "Frozen in Time" (NeurIPS workshop).

Figures produced:
  fig_cyl_forgetting      — cyl-sim rollout NRMSE: Phase 1 ep100 baseline → Phase 2 trajectory
  fig_well_forgetting     — Well dataset NRMSE across Phase 1 + Phase 2 (2×2 small multiples)
  fig_replay_recovery     — cyl-sim NRMSE vs sim-replay fraction (fine + coarse sweep)
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
# Style  (identical to make_figures.py)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "text.usetex":        False,
    "font.family":        "serif",
    "mathtext.fontset":   "cm",
    "axes.labelsize":     14,
    "font.size":          13,
    "legend.fontsize":    12,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.6,
})

EXPBASE = f"{ROOT}/walrus/experiments"
EVAL    = os.path.join(EXPBASE, "forget_eval_results")
OUT     = f"{ROOT}/realpdebench/analysis/plots"

# Colours
C = {
    "mhd":          "#1b6ca8",   # deep blue
    "trl2d":        "#c0392b",   # crimson
    "turbcool":     "#27ae60",   # green
    "active":       "#e67e22",   # orange
    "cyl_danger":   "#8e44ad",   # purple  (catastrophic forgetting on cyl-sim)
    "replay":       "#1b6ca8",   # same blue for the recovery curve
    "replay_fine":  "#2ecc71",   # green for fine sweep
    "baseline_ref": "#555555",   # grey reference lines
    "phase_div":    "#888888",
}
DS_COLORS = [C["mhd"], C["trl2d"], C["turbcool"], C["active"]]

# Phase region fills
P1_COLOR = "#1b6ca8"
P2_COLOR = "#c0392b"

# ---------------------------------------------------------------------------
# PKL loader (handles CUDA tensors on a CPU-only head node)
# ---------------------------------------------------------------------------
class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                        weights_only=False)
        return super().find_class(module, name)

def load_pkl(path):
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()

def mean_nrmse_from_pkl(folder_path, ds_dir, ds_prefix, exclude_full=True):
    """Load valid_loss_dict pkl and return mean NRMSE across fields."""
    pkls = glob.glob(
        os.path.join(EVAL, folder_path, ds_dir,
                     "finetune/0/viz/loss_dicts/valid_loss_dict_*.pkl")
    )
    if not pkls:
        return None
    d = load_pkl(pkls[0])
    vals = [
        float(v) for k, v in d.items()
        if k.startswith(ds_prefix) and "NRMSE_T=all_mean" in k
        and not (exclude_full and k.endswith("full_NRMSE_T=all_mean"))
    ]
    return float(np.mean(vals)) if vals else None

def cyl_rollout_nrmse(folder_path):
    """Load rollout_valid_loss_dict pkl for cylinder_numerical, return mean NRMSE."""
    pkls = glob.glob(
        os.path.join(EVAL, folder_path,
                     "cyl_sim/finetune/0/viz/loss_dicts/rollout_valid_loss_dict_*.pkl")
    )
    if not pkls:
        return None
    d = load_pkl(pkls[0])
    prefix = "rollout_valid_realpdebench_cylinder_numerical/"
    vals = [
        float(v) for k, v in d.items()
        if k.startswith(prefix) and "NRMSE_T=all_mean" in k
        and not k.endswith("full_NRMSE_T=all_mean")
    ]
    return float(np.mean(vals)) if vals else None

def cyl_rollout_nrmse_prefer_reeval(base_name):
    """Try reeval folder first, fall back to original if reeval not ready."""
    reeval = cyl_rollout_nrmse(base_name + "_reeval")
    if reeval is not None:
        return reeval, "reeval"
    orig = cyl_rollout_nrmse(base_name)
    return orig, "original"

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DS = [
    dict(name="MHD-64",            color=C["mhd"],
         folder="MHD_64",          prefix="valid_MHD_64/"),
    dict(name="TRL-2D",            color=C["trl2d"],
         folder="trl2d",           prefix="valid_turbulent_radiative_layer_2D/"),
    dict(name="Turbcool",          color=C["turbcool"],
         folder="turbcool",        prefix="valid_turbulence_gravity_cooling/"),
    dict(name="Active Matter",     color=C["active"],
         folder="active_matter",   prefix="valid_active_matter/"),
]

# Phase 1 eval folders  (total epochs: 0=pretrained, 5,10,15,20, then 51=best Phase-1 ckpt)
# NOTE: walrus_finetune_cylinder_lr5e5 ran max_epoch=51; previously mislabeled as ep100.
P1_FOLDERS = {
    0:   "pretrained",
    5:   "phase1_dyn_step5",
    10:  "phase1_dyn_step10",
    15:  "phase1_dyn_step15",
    20:  "phase1_dyn_step20",
    51:  "phase1_gn_lr5e5",    # actual epoch 51 (Phase-1 complete)
}

# Phase 2 gn_to_gn eval folders  (Phase-2 step → total epoch = 51 + step)
# NOTE: "gn_to_gn_last_ep90" was eval'd Aug-10 against step_100 (Jul-27) — correct ep100 data.
P2_FOLDERS = {
    20:  "gn_to_gn_step20",    # total ep 71
    40:  "gn_to_gn_step40",    # total ep 91
    60:  "gn_to_gn_step60",    # total ep 111
    80:  "gn_to_gn_step80",    # total ep 131
    100: "gn_to_gn_last_ep90", # total ep 151  (step_100 checkpoint, mislabeled ep90)
}

# Coarse replay sweep (0 % = gn_to_gn last; 100 % = Phase-1 ep100)
# Arms marked "reeval" will try _reeval folder first (stale original checkpoint)
COARSE_REPLAY_ARMS = {
    0:   ("gn_to_gn",        False),    # no-replay baseline
    10:  ("replay10pct_sim", False),    # valid
    20:  ("replay20pct_sim", True),     # stale → try reeval
    30:  ("replay30pct_sim", False),    # valid
    40:  ("replay40pct_sim", False),    # valid
    50:  ("replay50pct_sim", True),     # stale → try reeval
    60:  ("replay60pct_sim", True),     # stale → try reeval
    70:  ("replay70pct_sim", True),     # stale → try reeval
    80:  ("replay80pct_sim", True),     # stale → try reeval
    90:  ("replay90pct_sim", False),    # valid
    100: ("phase1_gn_lr5e5", False),    # pure-sim upper bound (Phase-1 ep51)
}

# Fine replay sweep (1-9%)
FINE_REPLAY_ARMS = list(range(1, 10))   # 1..9 %

# ---------------------------------------------------------------------------
# Pull all data once
# ---------------------------------------------------------------------------
print("Loading Well dataset data …")
well_data = {}  # ds_name → {total_epoch: nrmse}
for ds in DS:
    well_data[ds["name"]] = {}
    for ep, folder in P1_FOLDERS.items():
        v = mean_nrmse_from_pkl(folder, ds["folder"], ds["prefix"])
        if v is not None:
            well_data[ds["name"]][ep] = v
    for p2_step, folder in P2_FOLDERS.items():
        total_ep = 51 + p2_step
        v = mean_nrmse_from_pkl(folder, ds["folder"], ds["prefix"])
        if v is not None:
            well_data[ds["name"]][total_ep] = v
    sorted_ep = sorted(well_data[ds["name"]])
    print(f"  {ds['name']:15s}: epochs {sorted_ep}")

print("\nLoading cyl-sim data …")
cyl_p1_ep100 = cyl_rollout_nrmse("phase1_gn_lr5e5")
cyl_p2 = {}
for step in [20, 40, 60, 80]:
    cyl_p2[step] = cyl_rollout_nrmse(f"gn_to_gn_step{step}")
# gn_to_gn folder uses step_100 checkpoint (eval ran Aug-06, after step_100 Jul-27)
cyl_p2[100] = cyl_rollout_nrmse("gn_to_gn")
print(f"  Phase-1 ep100 baseline: {cyl_p1_ep100:.5f}")
for s, v in sorted(cyl_p2.items()):
    if v is not None:
        print(f"  Phase-2 step {s:3d}: {v:.5f}")
    else:
        print(f"  Phase-2 step {s:3d}: MISSING")

print("\nLoading pretrained_to_gn data …")
# walrus.pt → GN → real PIV (no Phase-1 sim); comparison ablation
p2gn_cyl = cyl_rollout_nrmse("pretrained_to_gn")
p2gn_well = {}
for ds in DS:
    v = mean_nrmse_from_pkl("pretrained_to_gn", ds["folder"], ds["prefix"])
    if v is not None:
        p2gn_well[ds["name"]] = v
if p2gn_cyl is not None:
    print(f"  cyl_sim: {p2gn_cyl:.5f}")
else:
    print("  cyl_sim: MISSING (eval pending, job 250137)")
for ds in DS:
    if ds["name"] in p2gn_well:
        print(f"  {ds['name']:15s}: {p2gn_well[ds['name']]:.5f}")
    else:
        print(f"  {ds['name']:15s}: MISSING (eval pending, job 250138)")

print("\nLoading coarse replay sweep data …")
coarse_replay_data = {}   # pct → nrmse
coarse_replay_source = {} # pct → "reeval"|"original"|"missing"
for pct, (base, needs_reeval) in COARSE_REPLAY_ARMS.items():
    if needs_reeval:
        v, src = cyl_rollout_nrmse_prefer_reeval(base)
    else:
        v = cyl_rollout_nrmse(base)
        src = "original" if v is not None else "missing"
    if v is not None:
        coarse_replay_data[pct] = v
        coarse_replay_source[pct] = src
        flag = " [REEVAL]" if src == "reeval" else (" [STALE?]" if needs_reeval else "")
        print(f"  {pct:3d}%: {v:.5f}{flag}")
    else:
        coarse_replay_source[pct] = "missing"
        flag = " [REEVAL PENDING]" if needs_reeval else ""
        print(f"  {pct:3d}%: MISSING{flag}")

print("\nLoading fine replay sweep data (1-9%) …")
fine_replay_data = {}   # pct → nrmse
for pct in FINE_REPLAY_ARMS:
    v = cyl_rollout_nrmse(f"replay{pct}pct_sim")
    if v is not None:
        fine_replay_data[pct] = v
        print(f"  {pct:3d}%: {v:.5f}")
    else:
        print(f"  {pct:3d}%: MISSING (eval pending)")

# Detect stale arms still showing original (not yet re-evaluated)
stale_arms = [p for p, s in coarse_replay_source.items() if s == "original"
              and COARSE_REPLAY_ARMS.get(p, (None, False))[1]]
if stale_arms:
    print(f"\nWARNING: coarse arms {stale_arms} still using STALE PKLs "
          f"(reeval not complete). Recovery curve may be non-monotonic.")

# ============================================================================
# Figure A — cyl-sim rollout NRMSE: Phase-2 forgetting trajectory
# ============================================================================
def fig_cyl_forgetting():
    p2_steps = sorted(cyl_p2)
    p2_vals  = [cyl_p2[s] for s in p2_steps]

    xs = [0] + p2_steps           # 0 = Phase-1 ep100 baseline
    ys = [cyl_p1_ep100] + p2_vals

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    # Sim FT baseline reference — dashed line with right-side label
    ax.axhline(cyl_p1_ep100, color=C["baseline_ref"], lw=1.1, ls="--", zorder=1)
    ax.text(102, cyl_p1_ep100 + 0.007, "Sim FT baseline",
            va="bottom", ha="left", fontsize=10, color=C["baseline_ref"])

    # Trajectory
    ax.plot(xs, ys, color=C["cyl_danger"], lw=2, marker="o", ms=5, zorder=3)

    ax.set_xlabel("Real data training epoch")
    ax.set_ylabel("Rollout NRMSE (cylinder sim)")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(-4, 118)   # room for right-side label
    ax.set_ylim(0.02, ax.get_ylim()[1] if ax.get_ylim()[1] > 0.42 else 0.43)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_cyl_forgetting.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_cyl_forgetting")


# ============================================================================
# Figure B — Well dataset NRMSE ratio across Phase 1 + Phase 2 (single panel)
#            Y-axis: NRMSE / pretrained NRMSE  (ratio; pretrained = 1.0)
#            All four well datasets on one axes with legend.
# ============================================================================
def fig_well_forgetting():
    PHASE_DIV = 51   # true Phase-1 completion epoch

    # Determine global x range across all datasets that have data
    all_eps = set()
    for ds in DS:
        all_eps.update(well_data[ds["name"]].keys())
    if not all_eps:
        print("fig_well_forgetting: no data loaded, skipping")
        return
    xmax = max(all_eps)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Phase region shading
    ax.axvspan(0,         PHASE_DIV, alpha=0.06, color=P1_COLOR, lw=0)
    ax.axvspan(PHASE_DIV, xmax,      alpha=0.06, color=P2_COLOR, lw=0)

    # Phase divider
    ax.axvline(PHASE_DIV, color=C["phase_div"], lw=1.0, ls="--", alpha=0.7)

    # Pretrained baseline at ratio = 1.0
    ax.axhline(1.0, color="grey", lw=0.9, ls=":", alpha=0.6, zorder=1)

    legend_handles = []
    for ds in DS:
        ep_nrmse = well_data[ds["name"]]
        if not ep_nrmse:
            continue
        pretrained = ep_nrmse.get(0)
        if not pretrained or pretrained == 0:
            continue

        total_eps  = sorted(ep_nrmse)
        ratio_vals = [ep_nrmse[ep] / pretrained for ep in total_eps]

        line, = ax.plot(total_eps, ratio_vals,
                        color=ds["color"], lw=2, marker="o", ms=5, zorder=3,
                        label=ds["name"])
        legend_handles.append(line)

        # Right-side label with final ratio value, annotation_clip=False
        final_ratio = ratio_vals[-1]
        ax.annotate(
            f"{ds['name']}  ×{final_ratio:.2f}",
            xy=(total_eps[-1], final_ratio),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9, color=ds["color"],
            va="center", ha="left",
            annotation_clip=False,
        )

    # Phase labels just below x-axis spine, above tick label text.
    # get_xaxis_transform(): x=data coords, y=[0,1] axes fraction.
    # va="bottom" at y=-0.04 → text floats upward from 4% below spine,
    # landing in the gap between spine and rotated tick numbers.
    xform = ax.get_xaxis_transform()
    ax.text(PHASE_DIV / 2, 0.02, "Phase 1",
            ha="center", va="bottom", fontsize=9, color=P1_COLOR, style="italic",
            transform=xform)
    ax.text((PHASE_DIV + xmax) / 2, 0.02, "Phase 2",
            ha="center", va="bottom", fontsize=9, color=P2_COLOR, style="italic",
            transform=xform)

    ax.set_xlabel("Total fine-tuning epochs from pretrain")
    ax.set_ylabel("NRMSE ratio (vs. pretrained)")
    ax.set_xlim(left=-3, right=xmax + 2)
    ax.set_ylim(bottom=0)
    # Tick marks at the actual eval epochs — avoid crowding
    tick_eps = sorted(all_eps)
    ax.set_xticks(tick_eps)
    ax.set_xticklabels([str(e) for e in tick_eps], rotation=45, ha="right", fontsize=9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_well_forgetting.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_well_forgetting")


# ============================================================================
# Figure C — Replay fraction vs cyl-sim NRMSE (recovery curve)
# ============================================================================
def fig_replay_recovery():
    has_fine   = len(fine_replay_data) > 0
    has_coarse = len(coarse_replay_data) > 0

    p1_baseline = cyl_p1_ep100
    no_replay   = coarse_replay_data.get(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Reference lines — labelled with right-side text, no legend entry
    ax.axhline(p1_baseline, color=C["baseline_ref"], lw=1.1, ls="--", zorder=1)
    ax.text(102, p1_baseline + 0.006, "Sim FT baseline",
            va="bottom", ha="left", fontsize=10, color=C["baseline_ref"])
    if no_replay is not None:
        ax.axhline(no_replay, color=C["trl2d"], lw=1.1, ls="--", alpha=0.7, zorder=1)
        ax.text(102, no_replay + 0.006, "No replay",
                va="bottom", ha="left", fontsize=10, color=C["trl2d"])

    # ---- Unified curve: 0 → 1 → 2 → … → 9 → 10 → 20 → … → 100 % ----
    # Fine sweep fills in 1–9 %; coarse provides the rest.
    # Plot ONE continuous line so the fine region connects to 0 % and 10 %.
    all_data = dict(coarse_replay_data)
    all_data.update(fine_replay_data)   # fine takes priority for 1–9 %

    if all_data:
        all_pcts  = sorted(all_data)
        all_nrmse = [all_data[p] for p in all_pcts]

        # Single continuous line through all x-values
        ax.plot(all_pcts, all_nrmse, color=C["replay"], lw=2, zorder=3)

        # Coarse markers (circles) — 0, 10, 20, …, 100 %
        if has_coarse:
            cx = [p for p in all_pcts if p in coarse_replay_data]
            cy = [all_data[p] for p in cx]
            ax.scatter(cx, cy, color=C["replay"], marker="o", s=28, zorder=4)

        # Fine markers — same style as coarse
        if has_fine:
            fx = [p for p in all_pcts if p in fine_replay_data]
            fy = [all_data[p] for p in fx]
            ax.scatter(fx, fy, color=C["replay"], marker="o", s=28, zorder=5)

    ax.set_xlabel("Sim replay fraction (%)")
    ax.set_ylabel("Rollout NRMSE (cylinder sim)")

    major_ticks = list(range(0, 101, 10))
    ax.set_xticks(major_ticks)
    ax.set_xlim(-3, 118)   # extra right margin for inline text labels
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_replay_recovery.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_replay_recovery")


# ============================================================================
# Channel-wise degradation figure
# ============================================================================
def fig_channel_degradation():
    """
    Horizontal lollipop chart: per-channel degradation ratio (Phase-2 ep100 /
    Pretrained) for all four Well datasets, grouped and color-coded by dataset.
    """
    # ---- Data: (label, ratio, dataset_color) listed bottom→top per group ----
    groups = [
        ("MHD-64", C["mhd"], [
            ("density",  1.2), ("B$_x$", 1.4), ("B$_y$", 1.4), ("B$_z$", 1.4),
            ("v$_x$",    1.3), ("v$_y$", 1.4), ("v$_z$", 1.4),
        ]),
        ("Turbcool", C["turbcool"], [
            ("density",     2.2), ("pressure",    2.0), ("temperature", 2.1),
            ("v$_x$",       2.5), ("v$_y$",       2.4), ("v$_z$",       2.5),
        ]),
        ("TRL-2D", C["trl2d"], [
            ("pressure", 3.5), ("v$_x$", 3.7), ("density", 4.6), ("v$_y$", 5.1),
        ]),
        ("Active Matter", C["active"], [
            ("D$_{xy}$", 9.8), ("D$_{yx}$", 9.8), ("D$_{yy}$", 10.9), ("D$_{xx}$", 11.4),
            ("E$_{xy}$", 11.6), ("E$_{yx}$", 11.6), ("E$_{xx}$", 12.1), ("E$_{yy}$", 12.4),
            ("conc.",     6.6), ("v$_y$",    15.0), ("v$_x$",    16.8),
        ]),
    ]

    # Build flat y-tick lists with group gaps
    labels, ratios, colors = [], [], []
    group_mid = {}   # ds_name → y position of group midpoint (for label)
    GAP = 0.7        # extra vertical gap between groups

    y = 0.0
    for ds_name, ds_color, channels in groups:
        y_start = y
        for ch_label, ratio in channels:
            labels.append(ch_label)
            ratios.append(ratio)
            colors.append(ds_color)
            y += 1.0
        group_mid[ds_name] = (y_start + y - 1) / 2
        y += GAP

    n = len(labels)
    ys = list(range(n))
    # Remap y positions to include gaps
    y_pos = []
    y = 0.0
    for ds_name, ds_color, channels in groups:
        for _ in channels:
            y_pos.append(y)
            y += 1.0
        y += GAP

    fig, ax = plt.subplots(figsize=(6.5, 7.5))

    # Reference line at ×1
    ax.axvline(1, color=C["baseline_ref"], lw=0.9, ls="--", zorder=1)

    # Lollipop stems + dots
    for yp, ratio, col in zip(y_pos, ratios, colors):
        ax.plot([1, ratio], [yp, yp], color=col, lw=1.2, alpha=0.55, zorder=2)
        ax.plot(ratio, yp, "o", color=col, ms=6, zorder=3)

    # Dataset group labels on the right margin
    xmax = max(ratios) * 1.05
    for ds_name, ds_color, channels in groups:
        ymid = sum(y_pos[sum(len(g[2]) for g in groups[:groups.index((ds_name, ds_color, channels))])
                        :sum(len(g[2]) for g in groups[:groups.index((ds_name, ds_color, channels))+1])]) / len(channels)
        ax.text(xmax + 0.4, ymid, ds_name,
                va="center", ha="left", fontsize=9,
                color=ds_color, fontweight="semibold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Degradation factor (Phase-2 / Pretrained NRMSE)")
    ax.set_xlim(0, xmax + 4.5)
    ax.set_ylim(-0.6, max(y_pos) + 0.6)
    ax.invert_yaxis()   # most severe at bottom
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_channel_degradation.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print("Saved fig_channel_degradation")


# ============================================================================
# Run
# ============================================================================
if __name__ == "__main__":
    fig_cyl_forgetting()
    fig_well_forgetting()
    fig_replay_recovery()
    fig_channel_degradation()
    print(f"\nAll figures saved to {OUT}")
