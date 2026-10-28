#!/usr/bin/env python3
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
"""
Extract per-epoch cylinder task performance (sim + real) from training loss dicts.

Training loss dicts are saved during each experiment's training run at every N epochs
and contain cylinder_numerical (sim) and cylinder_real_cl (real PIV) VRMSE.
These are the task-performance curves — NOT the Well forgetting metrics.

To get Well forgetting vs. training step, see forget_eval_checkpoints.sh which runs
forget_eval on intermediate checkpoints and populates forget_eval_results/{exp}/step_XX/.

Usage:
    python scripts/collect_training_curves.py
    python scripts/collect_training_curves.py --out_dir results/training_curves
"""

import argparse
import io
import os
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch


EXPBASE = Path(f"{ROOT}/walrus/experiments")

# Experiments to include in the training curves analysis.
# label → (experiment_dir, display_name, group, color)
EXPERIMENTS = {
    # Baselines
    "phase1_gn_lr5e5": (
        "walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Phase-1 GN (lr=5e-5)", "baseline", "#ff7f0e"),
    # GN-to-GN
    "gn_to_gn_lr5e5": (
        "walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "GN→GN (lr=5e-5)", "gn_to_gn", "#e377c2"),
    "gn_to_gn_lr1e4": (
        "walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001",
        "GN→GN (lr=1e-4)", "gn_to_gn", "#f7b6d2"),
    # CL v2 real% sweep
    "cl_v2_real0pct": (
        "walrus_cl_v2_real0pct_sim100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "CL v2 0% real", "cl_v2", "#c5b0d5"),
    "cl_v2_real25pct": (
        "walrus_cl_v2_real25pct_sim75pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "CL v2 25% real", "cl_v2", "#aec7e8"),
    "cl_v2_real50pct": (
        "walrus_cl_v2_real50pct_sim50pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "CL v2 50% real", "cl_v2", "#98df8a"),
    "cl_v2_real75pct": (
        "walrus_cl_v2_real75pct_sim25pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "CL v2 75% real", "cl_v2", "#ffbb78"),
    "cl_v2_real100pct": (
        "walrus_cl_v2_real100pct_sim0pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "CL v2 100% real", "cl_v2", "#ff9896"),
    "cl_v2_stride4": (
        "walrus_cl_v2_real50pct_stride4-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "CL v2 stride-4 50%real", "cl_v2", "#8c564b"),
    # Replay sweep
    "replay10pct": (
        "walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 10% sim", "replay", "#bcbd22"),
    "replay20pct": (
        "walrus_replay20pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 20% sim", "replay", "#b5cf6b"),
    "replay30pct": (
        "walrus_replay30pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 30% sim", "replay", "#8ca252"),
    "replay40pct": (
        "walrus_replay40pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 40% sim", "replay", "#637939"),
    "replay50pct": (
        "walrus_replay50pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 50% sim", "replay", "#17becf"),
    "replay60pct": (
        "walrus_replay60pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 60% sim", "replay", "#9edae5"),
    "replay70pct": (
        "walrus_replay70pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 70% sim", "replay", "#6b6ecf"),
    "replay80pct": (
        "walrus_replay80pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 80% sim", "replay", "#9c9ede"),
    "replay90pct": (
        "walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05",
        "Replay 90% sim", "replay", "#5254a3"),
}

TASK_METRICS = {
    "cyl_sim":  ("valid_realpdebench_cylinder_numerical/full_VRMSE_T=all_mean",  "Cylinder Sim VRMSE"),
    "cyl_real": ("valid_realpdebench_cylinder_real_cl/full_VRMSE_T=all_mean",    "Cylinder Real VRMSE"),
}


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        d = CPU_Unpickler(f).load()
    return {k: v.float().item() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def extract_epoch(filename: str) -> int:
    m = re.search(r"epoch(\d+)", filename)
    return int(m.group(1)) if m else -1


def collect_curves() -> dict:
    """
    Returns: {label: {"epochs": [...], "cyl_sim": [...], "cyl_real": [...], "display": str, "group": str, "color": str}}
    """
    results = {}
    for label, (exp_dir, display, group, color) in EXPERIMENTS.items():
        loss_dict_dir = EXPBASE / exp_dir / "finetune/0/viz/loss_dicts"
        pkls = sorted(loss_dict_dir.glob("valid_loss_dict_epoch*.pkl"),
                      key=lambda p: extract_epoch(p.name))
        if not pkls:
            print(f"  MISSING: {label} — no valid_loss_dict pkl files in {loss_dict_dir}")
            continue

        epochs, sim_vals, real_vals = [], [], []
        for pkl_path in pkls:
            epoch = extract_epoch(pkl_path.name)
            try:
                d = load_pkl(pkl_path)
            except Exception as e:
                print(f"  ERROR loading {pkl_path}: {e}")
                continue

            sim_key, _ = TASK_METRICS["cyl_sim"]
            real_key, _ = TASK_METRICS["cyl_real"]
            sim_v = d.get(sim_key)
            real_v = d.get(real_key)
            # Accept partial: some exps log only sim (phase1_gn) or only real (gn_to_gn)
            if sim_v is not None or real_v is not None:
                epochs.append(epoch)
                sim_vals.append(sim_v)
                real_vals.append(real_v)

        if epochs:
            sim_str = f"{sim_vals[-1]:.4f}" if sim_vals[-1] is not None else "N/A"
            real_str = f"{real_vals[-1]:.4f}" if real_vals[-1] is not None else "N/A"
            results[label] = {
                "epochs": epochs, "cyl_sim": sim_vals, "cyl_real": real_vals,
                "display": display, "group": group, "color": color,
                "n_epochs": max(epochs),
            }
            print(f"  OK: {label:30s}  {len(epochs)} epochs, final sim={sim_str}, real={real_str}")
        else:
            print(f"  MISSING metrics: {label}")

    return results


def print_final_table(curves: dict):
    print("\n=== Final-epoch cylinder task VRMSE ===")
    header = f"{'Experiment':<32}  {'Epochs':>6}  {'Cyl-Sim':>10}  {'Cyl-Real':>10}"
    print(header)
    print("-" * len(header))
    for label, c in curves.items():
        sim_s = f"{c['cyl_sim'][-1]:>10.4f}" if c['cyl_sim'][-1] is not None else f"{'N/A':>10}"
        real_s = f"{c['cyl_real'][-1]:>10.4f}" if c['cyl_real'][-1] is not None else f"{'N/A':>10}"
        print(f"{c['display']:<32}  {c['n_epochs']:>6}  {sim_s}  {real_s}")


def plot_group(curves: dict, group: str, out_dir: Path, group_label: str):
    group_curves = {k: v for k, v in curves.items() if v["group"] == group}
    if not group_curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{group_label} — Cylinder Task VRMSE vs. Training Epoch",
                 fontsize=12, fontweight="bold")

    for metric_key, ax, ylabel in [("cyl_sim", axes[0], "Cylinder Sim VRMSE"),
                                    ("cyl_real", axes[1], "Cylinder Real VRMSE")]:
        for label, c in group_curves.items():
            vals = c[metric_key]
            epochs = c["epochs"]
            # Filter out None entries
            paired = [(e, v) for e, v in zip(epochs, vals) if v is not None]
            if not paired:
                continue
            ep, vv = zip(*paired)
            ax.plot(ep, vv, marker="o", markersize=3,
                    linewidth=1.5, color=c["color"], label=c["display"])
        ax.set_xlabel("Training epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = out_dir / f"training_curves_{group}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_replay_summary(curves: dict, out_dir: Path):
    """Final-epoch performance across replay fractions as a single bar/line plot."""
    replay_curves = {k: v for k, v in curves.items() if v["group"] == "replay"}
    if not replay_curves:
        return

    fracs = []
    sim_finals, real_finals = [], []
    colors = []
    for label in sorted(replay_curves.keys(), key=lambda x: int(x.replace("replay","").replace("pct",""))):
        c = replay_curves[label]
        frac = int(label.replace("replay","").replace("pct",""))
        fracs.append(frac)
        sim_finals.append(c["cyl_sim"][-1])
        real_finals.append(c["cyl_real"][-1])
        colors.append(c["color"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Replay fraction sweep — final-epoch cylinder task VRMSE", fontsize=11, fontweight="bold")

    for ax, vals, title in [(axes[0], sim_finals, "Cylinder Sim VRMSE"),
                             (axes[1], real_finals, "Cylinder Real VRMSE")]:
        bars = ax.bar(fracs, vals, width=6, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xlabel("Sim replay fraction (% of training batch)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(fracs)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = out_dir / "replay_sim_fraction_vs_task_performance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_all_final(curves: dict, out_dir: Path):
    """Bar chart of all experiments' final cylinder sim and real VRMSE."""
    if not curves:
        return

    labels = list(curves.keys())

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels)*0.7), 9))
    fig.suptitle("All experiments — Final cylinder task VRMSE", fontsize=12, fontweight="bold")

    for ax, metric_key, title in [(axes[0], "cyl_sim", "Cylinder Sim VRMSE"),
                                   (axes[1], "cyl_real", "Cylinder Real VRMSE")]:
        present = [(l, curves[l]) for l in labels if curves[l][metric_key][-1] is not None]
        if not present:
            ax.set_title(f"{title} (no data)")
            continue
        p_labels, p_curves = zip(*present)
        vals = [c[metric_key][-1] for c in p_curves]
        disp = [c["display"] for c in p_curves]
        colors = [c["color"] for c in p_curves]
        x = np.arange(len(vals))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(disp, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(bottom=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=6.5)

    plt.tight_layout()
    path = out_dir / "all_experiments_final_task_vrmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=f"{ROOT}/walrus/results/training_curves",
                        help="Output directory for plots")
    args = parser.parse_args()

    print("Collecting cylinder task training curves...")
    curves = collect_curves()
    print(f"\nFound curves for {len(curves)} experiments.\n")
    print_final_table(curves)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_group(curves, "replay",   out_dir, "Replay experiments")
    plot_group(curves, "cl_v2",    out_dir, "CL v2 real% sweep")
    plot_group(curves, "gn_to_gn", out_dir, "GN→GN variants")
    plot_group(curves, "baseline", out_dir, "Phase-1 GN baseline")
    plot_replay_summary(curves, out_dir)
    plot_all_final(curves, out_dir)

    print(f"\nDone. Plots saved to {out_dir}/")
    print("\nNote: these are TASK performance curves (cylinder sim/real VRMSE).")
    print("For FORGETTING curves (Well dataset VRMSE vs. training step),")
    print("see forget_eval_checkpoints.sh which eval Well datasets at each intermediate checkpoint.")


if __name__ == "__main__":
    main()
