"""
Aggregate per-chunk partial JSON files from eval_walrus_realpdebench.py into
a single Rel L2 per model.

Usage:
    python aggregate_walrus_metrics.py
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import json
import pathlib
import numpy as np

OUT_BASE = pathlib.Path(f"{ROOT}/realpdebench/analysis")
N_CHUNKS = 8
LABELS = ["lr1e4", "lr5e5", "lr1e4_samplewise", "lr5e5_samplewise"]

print("=== RealPDEBench cylinder_real — Walrus Rel L2 ===\n")
print(f"{'Model':<22}  {'Rel L2':>8}  {'n_windows':>10}  {'Chunks OK':>10}")
print("-" * 58)

results = {}
for label in LABELS:
    chunks = []
    for ci in range(N_CHUNKS):
        p = OUT_BASE / f"walrus_rpde_chunk_{label}_{ci}.json"
        if not p.exists():
            print(f"  MISSING: {p.name}")
            continue
        with open(p) as f:
            d = json.load(f)
        chunks.append(d)

    if len(chunks) == 0:
        print(f"{label:<22}  (no chunks found)")
        continue

    # Aggregate: Rel L2 is mean of per-window ratios; since each chunk stores
    # its own mean and n_windows we reconstruct the overall mean via weighted sum.
    total_n = sum(c["n_windows"] for c in chunks)
    agg_rl2 = sum(c["rel_l2"] * c["n_windows"] for c in chunks) / total_n

    results[label] = {"rel_l2": agg_rl2, "n_windows": total_n, "n_chunks": len(chunks)}
    print(f"{label:<22}  {agg_rl2:>8.4f}  {total_n:>10d}  {len(chunks):>10d}/{N_CHUNKS}")

print(f"\n  DPOT-L simulated baseline: 0.1446")

agg_path = OUT_BASE / "walrus_rpde_final.json"
with open(agg_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {agg_path}")
