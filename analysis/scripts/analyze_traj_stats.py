"""
analyze_traj_stats.py

For each trajectory in cylinder_real (test split), compute per-trajectory
velocity RMS stats and compare against the GlobalRevNorm value derived from
cylinder_numerical/stats.yaml.

Motivation: if per-trajectory RMS varies substantially across Re numbers,
samplewise normalization will outperform global normalization because the
global norm applies the wrong scale to each trajectory.
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import math
import pathlib
import yaml
import h5py
import numpy as np

WELL_ROOT = pathlib.Path(f"{ROOT}/realpdebench/well")
HDF5_PATHS = {
    "test":  WELL_ROOT / "cylinder_real/data/test/cylinder_real_test.hdf5",
    "valid": WELL_ROOT / "cylinder_real/data/valid/cylinder_real_valid.hdf5",
}
STATS_YAML = WELL_ROOT / "cylinder_numerical/stats.yaml"

# ── GlobalRevNorm reference value ────────────────────────────────────────────
with open(STATS_YAML) as f:
    stats = yaml.safe_load(f)
rms_vec = np.array(stats["rms"]["velocity"])          # [rms_vx, rms_vy] from numerical sims
# GlobalRevNorm flattens fields: ||rms_vec||₂ / sqrt(n_spatial_padded=3)
global_rms = np.sqrt((rms_vec**2).sum()) / math.sqrt(3)
print(f"cylinder_numerical stats.yaml:")
print(f"  rms_vx={rms_vec[0]:.5f}  rms_vy={rms_vec[1]:.5f}")
print(f"  GlobalRevNorm combined RMS = {global_rms:.6f}\n")

# ── Per-trajectory stats ─────────────────────────────────────────────────────
records = []

for split, hdf5_path in HDF5_PATHS.items():
    if not hdf5_path.exists():
        print(f"[skip] {hdf5_path} not found")
        continue
    with h5py.File(hdf5_path, "r") as f:
        reynolds = np.array(f.attrs["reynolds_number"])  # (n_traj,)
        vel = f["t1_fields/velocity"]                    # (n_traj, T, x, y, 2)
        n_traj = vel.shape[0]
        print(f"Processing {split}: {n_traj} trajectories × {vel.shape[1]} timesteps ...")
        for i in range(n_traj):
            v = vel[i, ::10]                 # subsample every 10th frame — fast, same stats
            vx, vy = v[..., 0], v[..., 1]
            rms_vx = float(np.sqrt(np.mean(vx**2)))
            rms_vy = float(np.sqrt(np.mean(vy**2)))
            # Combined RMS matching GlobalRevNorm formula
            traj_combined_rms = math.sqrt(rms_vx**2 + rms_vy**2) / math.sqrt(3)
            mean_vx = float(np.mean(vx))
            max_speed = float(np.max(np.sqrt(vx**2 + vy**2)))
            records.append({
                "split": split,
                "re": float(reynolds[i]),
                "rms_vx": rms_vx,
                "rms_vy": rms_vy,
                "combined_rms": traj_combined_rms,
                "norm_ratio": traj_combined_rms / global_rms,   # ratio to GlobalRevNorm
                "mean_vx": mean_vx,
                "max_speed": max_speed,
            })

records.sort(key=lambda r: r["re"])

# ── Summary statistics ───────────────────────────────────────────────────────
combined_rms_vals = np.array([r["combined_rms"] for r in records])
norm_ratios       = np.array([r["norm_ratio"]   for r in records])
re_vals           = np.array([r["re"]           for r in records])

print(f"\n{'='*65}")
print(f"Per-trajectory combined RMS vs GlobalRevNorm ({global_rms:.5f})")
print(f"{'='*65}")
print(f"  n_traj:   {len(records)}")
print(f"  min RMS:  {combined_rms_vals.min():.5f}  (Re={re_vals[combined_rms_vals.argmin()]:.0f})")
print(f"  max RMS:  {combined_rms_vals.max():.5f}  (Re={re_vals[combined_rms_vals.argmax()]:.0f})")
print(f"  mean RMS: {combined_rms_vals.mean():.5f}")
print(f"  std RMS:  {combined_rms_vals.std():.5f}")
print(f"  CoV:      {combined_rms_vals.std()/combined_rms_vals.mean()*100:.1f}%  (coefficient of variation)")
print(f"  min ratio to global: {norm_ratios.min():.3f}x")
print(f"  max ratio to global: {norm_ratios.max():.3f}x")
print(f"  max/min ratio:       {norm_ratios.max()/norm_ratios.min():.2f}x spread")

# ── Per-trajectory table ─────────────────────────────────────────────────────
print(f"\n{'Re':>8}  {'rms_vx':>8}  {'rms_vy':>8}  {'comb_rms':>10}  {'ratio':>7}  {'mean_vx':>8}  split")
print("-" * 70)
for r in records:
    print(f"{r['re']:8.0f}  {r['rms_vx']:8.5f}  {r['rms_vy']:8.5f}  "
          f"{r['combined_rms']:10.5f}  {r['norm_ratio']:7.3f}x  "
          f"{r['mean_vx']:8.5f}  {r['split']}")

# ── Re-binned summary ────────────────────────────────────────────────────────
bins = [(1000,3000), (3000,5000), (5000,7000), (7000,9000), (9000,12000)]
print(f"\n{'Re range':>15}  {'n':>4}  {'mean comb_rms':>14}  {'std':>8}  {'ratio range':>14}")
print("-" * 62)
for lo, hi in bins:
    subset = [r for r in records if lo <= r["re"] < hi]
    if not subset:
        continue
    vals = np.array([r["combined_rms"] for r in subset])
    ratios = np.array([r["norm_ratio"] for r in subset])
    print(f"{lo:>6}–{hi:<6}    {len(subset):>4}  {vals.mean():>14.5f}  "
          f"{vals.std():>8.5f}  "
          f"{ratios.min():.3f}–{ratios.max():.3f}x")

# ── Correlation with Re ──────────────────────────────────────────────────────
corr = np.corrcoef(re_vals, combined_rms_vals)[0, 1]
print(f"\nCorrelation(Re, combined_rms) = {corr:.4f}")
print(f"\nKey takeaway:")
print(f"  GlobalRevNorm uses a single fixed RMS = {global_rms:.5f}")
print(f"  Real PIV trajectories span {norm_ratios.min():.3f}–{norm_ratios.max():.3f}x that value.")
print(f"  Samplewise norm adapts per-trajectory; global norm is off by up to "
      f"{max(abs(1-norm_ratios.min()), abs(norm_ratios.max()-1))*100:.0f}% for some trajectories.")
