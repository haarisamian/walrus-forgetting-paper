#!/usr/bin/env python3
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
"""
Collect and visualize cylinder sim/real performance vs. Well forgetting for replay experiments.

Shows how replay fraction affects the sim-real trade-off:
  - cyl_sim VRMSE: how well sim performance is retained (from forget_eval_cyl_sim_replay)
  - cyl_real VRMSE: real-data adaptation (from training logs)
  - Well forgetting: pretrained dataset degradation (from forget_eval_replay)

Produces:
  - replay_tradeoff_table.txt   — text summary table
  - replay_cyl_sim_vrmse.png   — cyl_sim VRMSE vs replay fraction
  - replay_tradeoff.png        — combined sim/real/forgetting trade-off

Usage:
    python scripts/collect_replay_cyl_sim_results.py
"""

import io
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


EXPBASE = Path(f"{ROOT}/walrus/experiments")
EVAL_BASE = EXPBASE / "forget_eval_results"

REPLAY_PCTS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
REPLAY_LABELS = [f"replay{p}pct_sim" for p in REPLAY_PCTS]

# Pretrained baseline (walrus.pt Well VRMSE) for forgetting ratios
PRETRAINED_WELL = {
    "MHD_64":   0.1493,
    "TRL_2D":   0.1148,
    "turbcool": 0.0842,
    "active":   0.0145,
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


def find_latest_valid_pkl(loss_dict_dir: Path) -> Path | None:
    if not loss_dict_dir.exists():
        return None
    pkls = sorted(loss_dict_dir.glob("valid_loss_dict_epoch*.pkl"))
    return pkls[-1] if pkls else None


def get_cyl_sim_vrmse(label: str) -> float | None:
    """Read cyl_sim forget_eval result for a checkpoint label."""
    d = EVAL_BASE / label / "cyl_sim/finetune/0/viz/loss_dicts"
    pkl = find_latest_valid_pkl(d)
    if pkl is None:
        return None
    data = load_pkl(pkl)
    key = "valid_realpdebench_cylinder_numerical/full_VRMSE_T=all_mean"
    return data.get(key)


def get_cyl_real_vrmse_from_training(label: str) -> float | None:
    """Read cyl_real VRMSE from training loss dicts (experiment training logs)."""
    # Map replay label to experiment dir
    pct = int(label.replace("replay", "").replace("pct_sim", ""))
    exp_dir = EXPBASE / f"walrus_replay{pct}pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/viz/loss_dicts"
    pkls = sorted(exp_dir.glob("valid_loss_dict_epoch*.pkl"))
    if not pkls:
        return None
    data = load_pkl(pkls[-1])
    key = "valid_realpdebench_cylinder_real_cl/full_VRMSE_T=all_mean"
    return data.get(key)


def get_well_forgetting(label: str) -> dict:
    """Read Well dataset step-1 VRMSE for a checkpoint label."""
    ds_info = [
        ("MHD_64",   "MHD_64",   "valid_MHD_64"),
        ("TRL_2D",   "trl2d",    "valid_turbulent_radiative_layer_2D"),
        ("turbcool", "turbcool", "valid_turbulence_gravity_cooling"),
        ("active",   "active_matter", "valid_active_matter"),
    ]
    result = {}
    for ds_key, ds_folder, val_prefix in ds_info:
        d = EVAL_BASE / label / ds_folder / "finetune/0/viz/loss_dicts"
        pkl = find_latest_valid_pkl(d)
        if pkl is None:
            continue
        data = load_pkl(pkl)
        key = f"{val_prefix}/full_VRMSE_T=all_mean"
        val = data.get(key)
        if val is not None:
            result[ds_key] = val
    return result


def collect_all() -> list[dict]:
    rows = []

    # Phase-1 GN baseline (0% real, 100% sim only)
    phase1_cyl_sim = get_cyl_sim_vrmse("phase1_gn_lr5e5")
    phase1_well = {k: 0.1676 if k == "MHD_64" else
                       0.2695 if k == "TRL_2D" else
                       0.1796 if k == "turbcool" else
                       0.0730
                   for k in PRETRAINED_WELL}
    rows.append({
        "label":    "phase1_gn_lr5e5",
        "display":  "Phase-1 GN (100% sim)",
        "sim_pct":  100,
        "cyl_sim":  phase1_cyl_sim,
        "cyl_real": None,
        "well":     phase1_well,
        "color":    "#ff7f0e",
    })

    # Replay experiments
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(REPLAY_PCTS)))
    for i, (pct, label) in enumerate(zip(REPLAY_PCTS, REPLAY_LABELS)):
        cyl_sim  = get_cyl_sim_vrmse(label)
        cyl_real = get_cyl_real_vrmse_from_training(label)
        well     = get_well_forgetting(label)
        rows.append({
            "label":    label,
            "display":  f"Replay {pct}% sim",
            "sim_pct":  pct,
            "cyl_sim":  cyl_sim,
            "cyl_real": cyl_real,
            "well":     well,
            "color":    colors[i],
        })

    # GN→GN (0% sim replay baseline)
    gn_cyl_sim = get_cyl_sim_vrmse("gn_to_gn")
    gn_well_raw = get_well_forgetting("gn_to_gn")
    # fallback to known values if PKL not found
    if not gn_well_raw:
        gn_well_raw = {"MHD_64": 0.2042, "TRL_2D": 0.4608, "turbcool": 0.1920, "active": 0.1554}
    rows.append({
        "label":    "gn_to_gn",
        "display":  "GN→GN (0% sim replay)",
        "sim_pct":  0,
        "cyl_sim":  gn_cyl_sim if gn_cyl_sim else 0.0374,
        "cyl_real": 0.0455,   # from training log
        "well":     gn_well_raw,
        "color":    "#e377c2",
    })

    return rows


def print_table(rows: list[dict]):
    print("\n=== Cylinder Sim/Real performance + Well forgetting (replay sweep) ===")
    header = f"{'Experiment':<28} {'Sim%':>5} {'CylSim':>9} {'CylReal':>9}  " \
             f"{'MHD':>8} {'TRL':>8} {'turb':>8} {'act':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sim_s  = f"{r['cyl_sim']:.4f}"  if r["cyl_sim"]  is not None else "  N/A  "
        real_s = f"{r['cyl_real']:.4f}" if r["cyl_real"] is not None else "  N/A  "

        def ratio(ds):
            v = r["well"].get(ds)
            if v is None:
                return "  N/A"
            return f"{v / PRETRAINED_WELL[ds]:.2f}x"

        print(f"{r['display']:<28} {r['sim_pct']:>5}  {sim_s:>9} {real_s:>9}  "
              f"{ratio('MHD_64'):>8} {ratio('TRL_2D'):>8} {ratio('turbcool'):>8} {ratio('active'):>8}")


def plot_cyl_sim_vs_pct(rows: list[dict], out_dir: Path):
    """Line plot: cyl_sim VRMSE vs. sim replay fraction."""
    replay_rows = [r for r in rows if r["sim_pct"] in REPLAY_PCTS and r["cyl_sim"] is not None]
    if not replay_rows:
        print("No cyl_sim data for replay experiments yet.")
        return

    pcts = [r["sim_pct"] for r in replay_rows]
    vals = [r["cyl_sim"] for r in replay_rows]

    phase1_val = next((r["cyl_sim"] for r in rows if r["label"] == "phase1_gn_lr5e5"), None)
    gn_val     = next((r["cyl_sim"] for r in rows if r["label"] == "gn_to_gn"), None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcts, vals, "o-", color="#2ca02c", linewidth=2, markersize=6, label="Replay sweep")

    if phase1_val is not None:
        ax.axhline(phase1_val, color="#ff7f0e", linestyle="--", linewidth=1.5,
                   label=f"Phase-1 GN (100% sim) = {phase1_val:.4f}")
    if gn_val is not None:
        ax.axhline(gn_val, color="#e377c2", linestyle="--", linewidth=1.5,
                   label=f"GN→GN (0% sim replay) = {gn_val:.4f}")

    ax.set_xlabel("Sim replay fraction (% of training batch)")
    ax.set_ylabel("Cyl-Sim VRMSE (step-1, lower=better)")
    ax.set_title("Cylinder Sim Performance Retained vs. Replay Fraction", fontweight="bold")
    ax.set_xticks(pcts)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = out_dir / "replay_cyl_sim_vrmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_tradeoff(rows: list[dict], out_dir: Path):
    """2×2 subplots: sim/real VRMSE and Well forgetting vs. replay fraction."""
    replay_rows = sorted([r for r in rows if r["sim_pct"] in REPLAY_PCTS], key=lambda r: r["sim_pct"])
    pcts = [r["sim_pct"] for r in replay_rows]

    phase1 = next((r for r in rows if r["label"] == "phase1_gn_lr5e5"), {})
    gn     = next((r for r in rows if r["label"] == "gn_to_gn"), {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Replay fraction sweep: sim-real task trade-off + Well forgetting",
                 fontsize=12, fontweight="bold")

    # Cyl-Sim VRMSE
    ax = axes[0, 0]
    vals = [r["cyl_sim"] for r in replay_rows]
    mask = [v is not None for v in vals]
    ax.plot([p for p, m in zip(pcts, mask) if m],
            [v for v, m in zip(vals, mask) if m],
            "o-", color="#2ca02c", linewidth=2, markersize=6)
    if phase1.get("cyl_sim"):
        ax.axhline(phase1["cyl_sim"], color="#ff7f0e", linestyle="--", linewidth=1.5,
                   label=f"Phase-1 GN={phase1['cyl_sim']:.4f}")
    if gn.get("cyl_sim"):
        ax.axhline(gn["cyl_sim"], color="#e377c2", linestyle="--", linewidth=1.5,
                   label=f"GN→GN={gn['cyl_sim']:.4f}")
    ax.set_title("Cyl-Sim VRMSE (↓ better = more sim retained)")
    ax.set_xlabel("Sim replay %"); ax.set_ylabel("VRMSE"); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0); ax.legend(fontsize=7)

    # Cyl-Real VRMSE
    ax = axes[0, 1]
    vals = [r["cyl_real"] for r in replay_rows]
    mask = [v is not None for v in vals]
    ax.plot([p for p, m in zip(pcts, mask) if m],
            [v for v, m in zip(vals, mask) if m],
            "o-", color="#1f77b4", linewidth=2, markersize=6)
    if gn.get("cyl_real"):
        ax.axhline(gn["cyl_real"], color="#e377c2", linestyle="--", linewidth=1.5,
                   label=f"GN→GN={gn['cyl_real']:.4f}")
    ax.set_title("Cyl-Real VRMSE (↓ better = adapts to real)")
    ax.set_xlabel("Sim replay %"); ax.set_ylabel("VRMSE"); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0); ax.legend(fontsize=7)

    # Well forgetting - TRL_2D (most sensitive)
    ax = axes[1, 0]
    for ds_key, color, label in [
        ("TRL_2D",   "#d62728", "TRL_2D"),
        ("active",   "#9467bd", "active_matter"),
    ]:
        ratios = [r["well"].get(ds_key, None) for r in replay_rows]
        ratios = [v / PRETRAINED_WELL[ds_key] if v is not None else None for v in ratios]
        mask = [v is not None for v in ratios]
        ax.plot([p for p, m in zip(pcts, mask) if m],
                [v for v, m in zip(ratios, mask) if m],
                "o-", color=color, linewidth=2, markersize=6, label=label)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="pretrained baseline")
    ax.set_title("Well forgetting ratio (TRL_2D + active_matter)")
    ax.set_xlabel("Sim replay %"); ax.set_ylabel("VRMSE / pretrained VRMSE"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Well forgetting - MHD + turbcool
    ax = axes[1, 1]
    for ds_key, color, label in [
        ("MHD_64",   "#ff7f0e", "MHD_64"),
        ("turbcool", "#2ca02c", "turbcool"),
    ]:
        ratios = [r["well"].get(ds_key, None) for r in replay_rows]
        ratios = [v / PRETRAINED_WELL[ds_key] if v is not None else None for v in ratios]
        mask = [v is not None for v in ratios]
        ax.plot([p for p, m in zip(pcts, mask) if m],
                [v for v, m in zip(ratios, mask) if m],
                "o-", color=color, linewidth=2, markersize=6, label=label)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, label="pretrained baseline")
    ax.set_title("Well forgetting ratio (MHD_64 + turbcool)")
    ax.set_xlabel("Sim replay %"); ax.set_ylabel("VRMSE / pretrained VRMSE"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = out_dir / "replay_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=f"{ROOT}/walrus/results/replay_cyl_sim",
                        help="Output directory for plots")
    args = parser.parse_args()

    print("Collecting replay cylinder sim/real + Well forgetting results...")
    rows = collect_all()

    print_table(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cyl_sim_vs_pct(rows, out_dir)
    plot_tradeoff(rows, out_dir)

    print(f"\nDone. Plots saved to {out_dir}/")
    print("\nNote: cyl_sim values come from forget_eval_cyl_sim_replay.sh (SamplewiseRevNorm eval).")
    print("Run once jobs 232444+232445 complete.")


if __name__ == "__main__":
    main()
