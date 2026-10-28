"""
create_cylinder_real_cl_splits.py

Apply the same get_splits() logic used for cylinder_numerical to cylinder_real,
creating train/valid/test splits for the continual learning experiments.

Output: /cephfs/store/gr-mc2473/ham51/realpdebench/well/cylinder_real_cl/
  data/train/cylinder_real_cl_train.hdf5   (74 trajectories)
  data/valid/cylinder_real_cl_valid.hdf5   (9  trajectories)
  data/test/cylinder_real_cl_test.hdf5     (9  trajectories)
  stats.yaml                                (computed from train split)

Split logic mirrors convert_realpde.py::get_splits():
  Sort by Re, hold out every 10th for test (pos 9,19,...), every 10th
  at offset 4 for valid (pos 4,14,...), remaining 74 → train.
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import h5py
import numpy as np
import os
import yaml

SRC   = f"{ROOT}/realpdebench/well/cylinder_real.hdf5"
OUTD  = f"{ROOT}/realpdebench/well/cylinder_real_cl"
DNAME = "realpdebench_cylinder_real_cl"


def get_splits(re_values):
    sorted_idx = np.argsort(re_values)
    test_idx   = sorted_idx[9::10]
    valid_idx  = sorted_idx[4::10]
    held_out   = set(test_idx.tolist()) | set(valid_idx.tolist())
    train_idx  = np.array([i for i in sorted_idx if i not in held_out])
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


with h5py.File(SRC, "r") as src:
    re_values = np.array(src.attrs["reynolds_number"], dtype=np.float32)
    vel_all   = src["t1_fields/velocity"]      # (92, T, 128, 64, 2) — lazy
    T = vel_all.shape[1]

    # Copy metadata we need
    t_axis = src["dimensions/time"][:]
    x_1d   = src["dimensions/x"][:]
    y_1d   = src["dimensions/y"][:]

    splits = get_splits(re_values)
    print("Split sizes:")
    for name, idxs in splits.items():
        print(f"  {name:5s}: {len(idxs)} trajectories  Re=[{re_values[idxs].min():.0f},{re_values[idxs].max():.0f}]")

    # ── Write one HDF5 per split ─────────────────────────────────────────────
    for split_name, idxs in splits.items():
        n = len(idxs)
        out_dir  = os.path.join(OUTD, "data", split_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"cylinder_real_cl_{split_name}.hdf5")

        print(f"\nWriting {split_name} ({n} traj) → {out_path}")
        with h5py.File(out_path, "w") as dst:
            dst.attrs["dataset_name"]          = DNAME
            dst.attrs["grid_type"]             = "cartesian"
            dst.attrs["n_spatial_dims"]        = 2
            dst.attrs["n_trajectories"]        = n
            dst.attrs["simulation_parameters"] = ["reynolds_number"]
            dst.attrs["reynolds_number"]       = re_values[idxs]

            dims = dst.create_group("dimensions")
            dims.attrs["spatial_dims"] = ["x", "y"]
            td = dims.create_dataset("time", data=t_axis)
            td.attrs["time_varying"] = True;  td.attrs["sample_varying"] = False
            xd = dims.create_dataset("x", data=x_1d)
            xd.attrs["time_varying"] = False; xd.attrs["sample_varying"] = False
            yd = dims.create_dataset("y", data=y_1d)
            yd.attrs["time_varying"] = False; yd.attrs["sample_varying"] = False

            # Boundary conditions — same as existing cylinder_real
            bcs = dst.create_group("boundary_conditions")
            x_open = bcs.create_group("x_open")
            xm = np.zeros(len(x_1d), dtype=np.int8); xm[0] = xm[-1] = 1
            x_open.create_dataset("mask", data=xm)
            x_open.attrs.update({"bc_type": "OPEN", "associated_dims": ["x"],
                                  "associated_fields": [], "sample_varying": False,
                                  "time_varying": False})

            y_wall = bcs.create_group("y_wall")
            ym = np.zeros(len(y_1d), dtype=np.int8); ym[0] = ym[-1] = 1
            y_wall.create_dataset("mask", data=ym)
            y_wall.attrs.update({"bc_type": "WALL", "associated_dims": ["y"],
                                  "associated_fields": [], "sample_varying": False,
                                  "time_varying": False})

            scalars = dst.create_group("scalars")
            scalars.attrs["field_names"] = ["reynolds_number"]
            re_ds = scalars.create_dataset("reynolds_number", data=re_values[idxs])
            re_ds.attrs["sample_varying"] = True
            re_ds.attrs["time_varying"]   = False

            dst.create_group("t0_fields").attrs["field_names"] = []
            dst.create_group("t2_fields").attrs["field_names"] = []

            t1 = dst.create_group("t1_fields")
            t1.attrs["field_names"] = ["velocity"]
            vel_ds = t1.create_dataset(
                "velocity", shape=(n, T, len(x_1d), len(y_1d), 2), dtype=np.float32
            )
            vel_ds.attrs.update({"dim_varying": np.array([True, True]),
                                  "time_varying": True, "sample_varying": True})

            for p, orig_idx in enumerate(idxs):
                vel_ds[p] = vel_all[orig_idx][:]
                print(f"  [{p+1}/{n}] Re={re_values[orig_idx]:.0f}", flush=True)

print("\nAll splits written.")

# ── Compute stats.yaml from train split ──────────────────────────────────────
print("\nComputing stats.yaml from train split ...")
train_path = os.path.join(OUTD, "data", "train", "cylinder_real_cl_train.hdf5")

rms_vx_sq, rms_vy_sq   = 0.0, 0.0
mean_vx,   mean_vy     = 0.0, 0.0
rdms_vx_sq, rdms_vy_sq = 0.0, 0.0   # rms_delta
n_frames_total          = 0
n_deltas_total          = 0

with h5py.File(train_path, "r") as f:
    vel = f["t1_fields/velocity"]   # (74, T, 128, 64, 2)
    n_traj = vel.shape[0]
    for i in range(n_traj):
        v = vel[i][:]               # (T, 128, 64, 2)
        n_frames = v.shape[0]
        rms_vx_sq += np.mean(v[..., 0] ** 2) * n_frames
        rms_vy_sq += np.mean(v[..., 1] ** 2) * n_frames
        mean_vx   += np.mean(v[..., 0]) * n_frames
        mean_vy   += np.mean(v[..., 1]) * n_frames
        d = v[1:] - v[:-1]
        rdms_vx_sq += np.mean(d[..., 0] ** 2) * (n_frames - 1)
        rdms_vy_sq += np.mean(d[..., 1] ** 2) * (n_frames - 1)
        n_frames_total += n_frames
        n_deltas_total += (n_frames - 1)
        if (i + 1) % 10 == 0:
            print(f"  stats [{i+1}/{n_traj}]", flush=True)

rms_vx = float(np.sqrt(rms_vx_sq / n_frames_total))
rms_vy = float(np.sqrt(rms_vy_sq / n_frames_total))
mean_vx = float(mean_vx / n_frames_total)
mean_vy = float(mean_vy / n_frames_total)
rms_delta_vx = float(np.sqrt(rdms_vx_sq / n_deltas_total))
rms_delta_vy = float(np.sqrt(rdms_vy_sq / n_deltas_total))

stats = {
    "mean":       {"velocity": [mean_vx, mean_vy]},
    "rms":        {"velocity": [rms_vx,  rms_vy]},
    "rms_delta":  {"velocity": [rms_delta_vx, rms_delta_vy]},
}
os.makedirs(OUTD, exist_ok=True)
yaml_path = os.path.join(OUTD, "stats.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(stats, f, default_flow_style=False)

print(f"\nstats.yaml → {yaml_path}")
print(f"  rms_vx={rms_vx:.5f}  rms_vy={rms_vy:.5f}")
print(f"  rms_delta_vx={rms_delta_vx:.6f}  rms_delta_vy={rms_delta_vy:.6f}")
print(f"  mean_vx={mean_vx:.5f}  mean_vy={mean_vy:.5f}")
print("Done.")
