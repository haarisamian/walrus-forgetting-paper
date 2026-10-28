#!/usr/bin/env python3
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
"""
Collect step-1 VRMSE from forget-eval PKL files and produce forgetting curves.

Step-1 VRMSE is the primary metric for the forgetting evaluation:
- Immune to MHD_64 rollout instability (Alfvénic mode amplification via samplewise norm)
- Measured as a single-step prediction, so it reflects model knowledge not rollout stability

Key = valid_{dataset_key}/full_VRMSE_T=all_mean  (full = mean over all physical fields)

Usage:
    python scripts/collect_forgetting_results.py
    python scripts/collect_forgetting_results.py --out_dir results/forgetting_plots
    python scripts/collect_forgetting_results.py --print_table
"""

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ─── Config ──────────────────────────────────────────────────────────────────

EXPBASE = Path(f"{ROOT}/walrus/experiments")
VIDBASE = Path(f"{ROOT}/walrus/video_evals")
EVAL_BASE = EXPBASE / "forget_eval_results"

# Dataset canonical names → pkl key prefix
DATASETS = {
    "MHD_64":        "valid_MHD_64",
    "TRL_2D":        "valid_turbulent_radiative_layer_2D",
    "turbcool":      "valid_turbulence_gravity_cooling",
    "active_matter": "valid_active_matter",
}

# Folder name used by eval scripts (may differ from canonical key)
DS_FOLDER = {
    "MHD_64":        "MHD_64",
    "TRL_2D":        "trl2d",
    "turbcool":      "turbcool",
    "active_matter": "active_matter",
}

# gn_to_gn folder varies depending on whether best or last checkpoint was used
GN_TO_GN_FOLDER_CANDIDATES = ["gn_to_gn", "gn_to_gn_best", "gn_to_gn_last_ep90"]

# Checkpoint display order along the finetuning progression
# label → (display_name, color)
CHECKPOINT_STYLES = {
    "pretrained":          ("Pretrained walrus.pt",    "#1f77b4"),
    "phase1_gn_lr5e5":    ("Phase-1 GN (lr=5e-5)",    "#ff7f0e"),
    # SW-based phase-2
    "sw_to_gn_5e5_ep100": ("SW→GN (lr=5e-5)",         "#2ca02c"),
    "sw_to_gn_1e4_ep100": ("SW→GN (lr=1e-4)",         "#d62728"),
    "sw_lr1e4_ep100":     ("SW lr=1e-4",               "#9467bd"),
    # Continual learning v2 (replay fraction sweep)
    "cl_v2_real0pct":     ("CL v2 0% real",            "#c5b0d5"),
    "cl_v2_real25pct":    ("CL v2 25% real",           "#aec7e8"),
    "cl_v2_real50pct":    ("CL v2 50% real",           "#98df8a"),
    "cl_v2_real75pct":    ("CL v2 75% real",           "#ffbb78"),
    "cl_v2_real100pct":   ("CL v2 100% real",          "#ff9896"),
    "cl_v2_stride4_ep50": ("CL v2 stride-4 50%real",   "#8c564b"),
    # GN-to-GN variants
    "gn_to_gn":           ("GN→GN (lr=5e-5)",          "#e377c2"),
    "gn_to_gn_lr1e4":     ("GN→GN (lr=1e-4)",          "#f7b6d2"),
    # Replay (sim data fraction during phase-1 training)
    "replay10pct_sim":    ("Replay 10% sim",            "#bcbd22"),
    "replay20pct_sim":    ("Replay 20% sim",            "#dbdb8d"),
    "replay30pct_sim":    ("Replay 30% sim",            "#17becf"),
    "replay40pct_sim":    ("Replay 40% sim",            "#9edae5"),
    "replay50pct_sim":    ("Replay 50% sim",            "#7f7f7f"),
    "replay60pct_sim":    ("Replay 60% sim",            "#c7c7c7"),
    "replay70pct_sim":    ("Replay 70% sim",            "#e7ba52"),
    "replay80pct_sim":    ("Replay 80% sim",            "#cedb9c"),
    "replay90pct_sim":    ("Replay 90% sim",            "#8ca252"),
}
CKPT_ORDER = list(CHECKPOINT_STYLES.keys())

# Per-checkpoint PKL search paths (multiple candidates, first existing wins)
# Format: list of (ckpt_label, dataset_key, [candidate_paths])
def build_search_map():
    entries = []
    for ckpt_label in CKPT_ORDER:
        for ds_key in DATASETS:
            ds_folder = DS_FOLDER[ds_key]
            candidates = []

            if ckpt_label == "gn_to_gn":
                # gn_to_gn_lr5e5: folder name depends on best/last checkpoint availability
                for gn_folder in GN_TO_GN_FOLDER_CANDIDATES:
                    candidates.append(EVAL_BASE / gn_folder / ds_folder / "finetune/0/viz/loss_dicts")
            else:
                candidates.append(EVAL_BASE / ckpt_label / ds_folder / "finetune/0/viz/loss_dicts")

            # video_evals use dataset key directly (e.g. pretrained_MHD_64); active_matter uses ds_key as-is
            candidates.append(VIDBASE / f"{ckpt_label}_{ds_key}" / "finetune/0/viz/loss_dicts")

            # phase1_gn in video_evals uses "phase1_gn" (without "_lr5e5")
            if ckpt_label == "phase1_gn_lr5e5":
                candidates.append(VIDBASE / f"phase1_gn_{ds_key}" / "finetune/0/viz/loss_dicts")

            # sw_gn_1e4 in video_evals uses "sw_gn_1e4"
            if ckpt_label == "sw_to_gn_1e4_ep100":
                candidates.append(VIDBASE / f"sw_gn_1e4_{ds_key}" / "finetune/0/viz/loss_dicts")

            entries.append((ckpt_label, ds_key, candidates))
    return entries


# ─── Helpers ─────────────────────────────────────────────────────────────────

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def find_latest_pkl(loss_dict_dir: Path) -> Path | None:
    """Return the most recent valid_loss_dict_epoch*.pkl in a dir."""
    if not loss_dict_dir.exists():
        return None
    pkls = sorted(loss_dict_dir.glob("valid_loss_dict_epoch*.pkl"))
    return pkls[-1] if pkls else None


def load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        d = CPU_Unpickler(f).load()
    return {k: v.float().item() if isinstance(v, torch.Tensor) else v for k, v in d.items()}


def extract_vrmse(d: dict, ds_prefix: str, field: str = "full") -> float | None:
    key = f"{ds_prefix}/{field}_VRMSE_T=all_mean"
    return d.get(key)


def extract_per_field_vrmse(d: dict, ds_prefix: str) -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if k.startswith(ds_prefix) and "VRMSE_T=all_mean" in k:
            field = k.replace(ds_prefix + "/", "").replace("_VRMSE_T=all_mean", "")
            out[field] = v
    return out


# ─── Collection ──────────────────────────────────────────────────────────────

def collect_results(verbose: bool = False) -> dict:
    """
    Returns nested dict: results[ckpt_label][ds_key] = {
        'full': float,
        'per_field': {field: float},
        'source': str (path),
        'epoch': int,
    }
    """
    results = {}
    search_map = build_search_map()

    for ckpt_label, ds_key, candidates in search_map:
        found_path = None
        for cand_dir in candidates:
            pkl = find_latest_pkl(cand_dir)
            if pkl is not None:
                found_path = pkl
                break

        if found_path is None:
            if verbose:
                print(f"  MISSING: {ckpt_label} / {ds_key}")
            continue

        try:
            d = load_pkl(found_path)
        except Exception as e:
            print(f"  ERROR loading {found_path}: {e}")
            continue

        ds_prefix = DATASETS[ds_key]
        full_vrmse = extract_vrmse(d, ds_prefix, "full")
        if full_vrmse is None:
            if verbose:
                print(f"  KEY MISSING in {found_path} for {ds_prefix}")
            continue

        epoch_str = found_path.name.replace("valid_loss_dict_epoch", "").replace("_rank0.pkl", "")
        try:
            epoch = int(epoch_str)
        except ValueError:
            epoch = -1

        if ckpt_label not in results:
            results[ckpt_label] = {}
        results[ckpt_label][ds_key] = {
            "full": full_vrmse,
            "per_field": extract_per_field_vrmse(d, ds_prefix),
            "source": str(found_path),
            "epoch": epoch,
        }

        if verbose:
            print(f"  OK: {ckpt_label:30s} / {ds_key:12s}  full_VRMSE={full_vrmse:.4f}  (epoch {epoch}, {found_path.parent.parent.parent.parent.parent.name})")

    return results


# ─── Table printing ──────────────────────────────────────────────────────────

def print_table(results: dict):
    ds_keys = list(DATASETS.keys())
    header = f"{'Checkpoint':<28}" + "".join(f"  {ds:>12}" for ds in ds_keys)
    print("\n=== Step-1 VRMSE (full field average) ===")
    print(header)
    print("-" * len(header))

    for ckpt_label in CKPT_ORDER:
        name, _ = CHECKPOINT_STYLES[ckpt_label]
        row = f"{name:<28}"
        for ds_key in ds_keys:
            val = results.get(ckpt_label, {}).get(ds_key, {}).get("full")
            row += f"  {val:>12.4f}" if val is not None else f"  {'N/A':>12}"
        print(row)

    print()

    # Per-dataset forgetting ratio (vs pretrained)
    pretrained_vals = {ds: results.get("pretrained", {}).get(ds, {}).get("full") for ds in ds_keys}
    print("=== Forgetting ratio (checkpoint / pretrained — >1 means worse) ===")
    print(header)
    print("-" * len(header))
    for ckpt_label in CKPT_ORDER:
        if ckpt_label == "pretrained":
            continue
        name, _ = CHECKPOINT_STYLES[ckpt_label]
        row = f"{name:<28}"
        for ds_key in ds_keys:
            val = results.get(ckpt_label, {}).get(ds_key, {}).get("full")
            ref = pretrained_vals.get(ds_key)
            if val is not None and ref is not None:
                ratio = val / ref
                row += f"  {ratio:>12.3f}x"
            else:
                row += f"  {'N/A':>12}"
        print(row)
    print()


def print_per_field_table(results: dict):
    for ds_key, ds_prefix in DATASETS.items():
        print(f"\n=== {ds_key} — per-field VRMSE ===")
        # Collect all field names
        all_fields = set()
        for ckpt_label in CKPT_ORDER:
            pf = results.get(ckpt_label, {}).get(ds_key, {}).get("per_field", {})
            all_fields.update(pf.keys())
        all_fields = sorted(all_fields)
        if not all_fields:
            print("  (no data)")
            continue

        header = f"{'Checkpoint':<28}" + "".join(f"  {f[:14]:>14}" for f in all_fields)
        print(header)
        print("-" * len(header))
        for ckpt_label in CKPT_ORDER:
            name, _ = CHECKPOINT_STYLES[ckpt_label]
            row = f"{name:<28}"
            pf = results.get(ckpt_label, {}).get(ds_key, {}).get("per_field", {})
            for field in all_fields:
                v = pf.get(field)
                row += f"  {v:>14.4f}" if v is not None else f"  {'N/A':>14}"
            print(row)


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_forgetting_curves(results: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_keys = list(DATASETS.keys())

    # --- 1. Bar chart per dataset, checkpoints on x-axis ---
    fig, axes = plt.subplots(1, len(ds_keys), figsize=(5 * len(ds_keys), 5), sharey=False)
    if len(ds_keys) == 1:
        axes = [axes]

    for ax, ds_key in zip(axes, ds_keys):
        ckpt_labels = []
        values = []
        colors = []
        for ckpt_label in CKPT_ORDER:
            v = results.get(ckpt_label, {}).get(ds_key, {}).get("full")
            if v is not None:
                ckpt_labels.append(CHECKPOINT_STYLES[ckpt_label][0])
                values.append(v)
                colors.append(CHECKPOINT_STYLES[ckpt_label][1])

        if not values:
            ax.set_title(f"{ds_key}\n(no data)")
            continue

        x = np.arange(len(values))
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ckpt_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Step-1 VRMSE (lower = better)")
        ax.set_title(f"{ds_key}\nStep-1 VRMSE")

        # Add value labels on bars
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        # Horizontal reference line at pretrained level
        ref = results.get("pretrained", {}).get(ds_key, {}).get("full")
        if ref is not None:
            ax.axhline(ref, color="#1f77b4", linestyle="--", linewidth=1.5, alpha=0.7, label="Pretrained")

    fig.suptitle("Catastrophic Forgetting on Well Datasets\n(Step-1 VRMSE; lower = better, dashed = pretrained baseline)",
                 fontsize=11)
    plt.tight_layout()
    path = out_dir / "forgetting_bar_step1_vrmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- 2. Forgetting ratio plot (normalized by pretrained) ---
    fig, axes = plt.subplots(1, len(ds_keys), figsize=(5 * len(ds_keys), 5), sharey=True)
    if len(ds_keys) == 1:
        axes = [axes]

    for ax, ds_key in zip(axes, ds_keys):
        ref = results.get("pretrained", {}).get(ds_key, {}).get("full")
        if ref is None:
            ax.set_title(f"{ds_key}\n(no pretrained ref)")
            continue

        ckpt_labels = []
        ratios = []
        colors = []
        for ckpt_label in CKPT_ORDER:
            if ckpt_label == "pretrained":
                continue
            v = results.get(ckpt_label, {}).get(ds_key, {}).get("full")
            if v is not None:
                ckpt_labels.append(CHECKPOINT_STYLES[ckpt_label][0])
                ratios.append(v / ref)
                colors.append(CHECKPOINT_STYLES[ckpt_label][1])

        if not ratios:
            ax.set_title(f"{ds_key}\n(no finetuned data)")
            continue

        x = np.arange(len(ratios))
        bars = ax.bar(x, ratios, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Pretrained (ratio=1)")
        ax.set_xticks(x)
        ax.set_xticklabels(ckpt_labels, rotation=40, ha="right", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("VRMSE ratio (finetuned / pretrained)")
        ax.set_title(f"{ds_key}")

        for bar, r in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{r:.2f}x", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Forgetting Ratio vs Pretrained Baseline\n(>1 = worse than pretrained = forgetting)",
                 fontsize=11)
    plt.tight_layout()
    path = out_dir / "forgetting_ratio_step1_vrmse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- 3. Per-field heatmap per dataset ---
    for ds_key in ds_keys:
        all_fields = set()
        per_field_data = {}
        for ckpt_label in CKPT_ORDER:
            pf = results.get(ckpt_label, {}).get(ds_key, {}).get("per_field", {})
            all_fields.update(pf.keys())
            per_field_data[ckpt_label] = pf

        all_fields = sorted(f for f in all_fields if f != "full")
        present_ckpts = [c for c in CKPT_ORDER if any(per_field_data.get(c, {}).get(f) for f in all_fields)]
        if not all_fields or not present_ckpts:
            continue

        matrix = np.full((len(present_ckpts), len(all_fields)), np.nan)
        for i, ckpt in enumerate(present_ckpts):
            for j, field in enumerate(all_fields):
                v = per_field_data.get(ckpt, {}).get(field)
                if v is not None:
                    matrix[i, j] = v

        fig, ax = plt.subplots(figsize=(max(6, len(all_fields) * 1.2), max(4, len(present_ckpts) * 0.8)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(np.arange(len(all_fields)))
        ax.set_xticklabels(all_fields, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(present_ckpts)))
        ax.set_yticklabels([CHECKPOINT_STYLES[c][0] for c in present_ckpts], fontsize=8)
        plt.colorbar(im, ax=ax, label="Step-1 VRMSE")
        ax.set_title(f"{ds_key} — Per-field Step-1 VRMSE\n(NaN = missing data)")

        for i in range(len(present_ckpts)):
            for j in range(len(all_fields)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > matrix[~np.isnan(matrix)].max() * 0.7 else "black")

        plt.tight_layout()
        path = out_dir / f"per_field_heatmap_{ds_key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=f"{ROOT}/walrus/results/forgetting_plots",
                        help="Output directory for plots")
    parser.add_argument("--print_table", action="store_true", help="Print VRMSE table to stdout")
    parser.add_argument("--per_field", action="store_true", help="Print per-field VRMSE breakdown")
    parser.add_argument("--verbose", action="store_true", help="Print path resolution details")
    args = parser.parse_args()

    print("Collecting results...")
    results = collect_results(verbose=True)

    n_found = sum(len(v) for v in results.values())
    print(f"\nFound {n_found} checkpoint×dataset results across {len(results)} checkpoints.\n")

    # Always print the summary table
    print_table(results)

    if args.per_field:
        print_per_field_table(results)

    out_dir = Path(args.out_dir)
    plot_forgetting_curves(results, out_dir)

    print(f"\nDone. Plots saved to {out_dir}/")
    print("\nNote: MHD_64 rollout VRMSE is excluded from analysis (confounded by samplewise")
    print("      normalization feedback loop amplifying Alfvénic modes in pretrained model).")
    print("      Step-1 VRMSE is immune to this and correctly reflects forgetting signal.")


if __name__ == "__main__":
    main()
