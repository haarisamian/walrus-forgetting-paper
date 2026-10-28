import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import numpy as np
import os
import h5py
from datasets import load_from_disk

DATA_DIR   = f"{ROOT}/realpdebench/data"
OUTPUT_DIR = f"{ROOT}/realpdebench/well"


def get_splits(re_values):
    """
    Partition 92 trajectories into train / valid / test by Re value.

    Sort by Re, then hold out every 10th trajectory for test (positions
    9, 19, ..., 89 → 9 traj) and every 10th at a different offset for
    valid (positions 4, 14, ..., 84 → 9 traj). Remaining 74 go to train.

    Striding ensures test/valid Re values are spread evenly across
    [1781, 11812], so we're measuring the sim-to-real gap at Re values
    the model has seen numerically — isolating domain gap from Re
    generalisation.
    """
    sorted_idx = np.argsort(re_values)
    test_idx   = sorted_idx[9::10]
    valid_idx  = sorted_idx[4::10]
    held_out   = set(test_idx.tolist()) | set(valid_idx.tolist())
    train_idx  = np.array([i for i in sorted_idx if i not in held_out])
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


def convert(data_type):
    """data_type: 'real' or 'numerical'"""

    # ── Load Arrow dataset ────────────────────────────────────────────────
    path    = f"{DATA_DIR}/cylinder/hf_dataset/{data_type}"
    dataset = load_from_disk(path)
    n_traj  = len(dataset)
    print(f"\nLoaded {data_type}: {n_traj} trajectories")

    row0 = dataset[0]
    T = row0["shape_t"]
    H = row0["shape_h"]
    W = row0["shape_w"]
    t = np.frombuffer(row0["t"], dtype=np.float64)
    print(f"  Source shape: T={T}, H={H}, W={W}")

    # ── Grid coordinates: always use real PIV grid (128×64) ──────────────
    # After box-averaging numerical down to 128×64 both datasets live on
    # the same physical domain, so we declare the output grid using the
    # real PIV coordinates directly.
    real_row0 = load_from_disk(f"{DATA_DIR}/cylinder/hf_dataset/real")[0]
    H_out = real_row0["shape_h"]   # 64
    W_out = real_row0["shape_w"]   # 128
    x_2d  = np.frombuffer(real_row0["x"], dtype=np.float64).reshape(H_out, W_out)
    y_2d  = np.frombuffer(real_row0["y"], dtype=np.float64).reshape(H_out, W_out)
    x_1d  = x_2d[0, :]   # (128,) streamwise
    y_1d  = y_2d[:, 0]   # (64,)  cross-stream
    print(f"  Output grid:  W={W_out}, H={H_out}")

    # ── Reynolds numbers and split assignment ─────────────────────────────
    re_values = np.array([
        float(row["sim_id"].replace(".h5", "")) for row in dataset
    ], dtype=np.float32)

    # Numerical: proper train/valid/test split.
    # Real: all 92 trajectories go to test — zero-shot eval only.
    if data_type == "numerical":
        splits = get_splits(re_values)
    else:
        splits = {"test": np.arange(n_traj)}

    for split_name, idxs in splits.items():
        print(f"  {split_name:5s}: {len(idxs):3d} trajectories  "
              f"Re [{re_values[idxs].min():.0f}, {re_values[idxs].max():.0f}]")

    # ── Pre-create one HDF5 file per split ────────────────────────────────
    # Pre-allocate the velocity dataset at full size, then fill one
    # trajectory at a time — keeps peak memory at ~250 MB per trajectory.
    hdf = {}   # split_name → open h5py.File
    pos = {}   # original dataset index → (split_name, row_position_in_split)

    for split_name, idxs in splits.items():
        n_split  = len(idxs)
        out_dir  = os.path.join(OUTPUT_DIR, f"cylinder_{data_type}", "data", split_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"cylinder_{data_type}_{split_name}.hdf5")

        f = h5py.File(out_path, "w")

        f.attrs["dataset_name"]          = f"realpdebench_cylinder_{data_type}"
        f.attrs["grid_type"]             = "cartesian"
        f.attrs["n_spatial_dims"]        = 2
        f.attrs["n_trajectories"]        = n_split
        f.attrs["simulation_parameters"] = ["reynolds_number"]
        f.attrs["reynolds_number"]       = re_values[idxs]

        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        td = dims.create_dataset("time", data=t)
        td.attrs["time_varying"] = True;  td.attrs["sample_varying"] = False
        xd = dims.create_dataset("x", data=x_1d)
        xd.attrs["time_varying"] = False; xd.attrs["sample_varying"] = False
        yd = dims.create_dataset("y", data=y_1d)
        yd.attrs["time_varying"] = False; yd.attrs["sample_varying"] = False

        bcs    = f.create_group("boundary_conditions")
        x_open = bcs.create_group("x_open")
        xm = np.zeros(W_out, dtype=np.int8); xm[0] = xm[-1] = 1
        x_open.create_dataset("mask", data=xm)
        x_open.attrs["bc_type"]           = "OPEN"
        x_open.attrs["associated_dims"]   = ["x"]
        x_open.attrs["associated_fields"] = []
        x_open.attrs["sample_varying"]    = False
        x_open.attrs["time_varying"]      = False

        y_wall = bcs.create_group("y_wall")
        ym = np.zeros(H_out, dtype=np.int8); ym[0] = ym[-1] = 1
        y_wall.create_dataset("mask", data=ym)
        y_wall.attrs["bc_type"]           = "WALL"
        y_wall.attrs["associated_dims"]   = ["y"]
        y_wall.attrs["associated_fields"] = []
        y_wall.attrs["sample_varying"]    = False
        y_wall.attrs["time_varying"]      = False

        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = ["reynolds_number"]
        re_ds = scalars.create_dataset("reynolds_number", data=re_values[idxs])
        re_ds.attrs["sample_varying"] = True
        re_ds.attrs["time_varying"]   = False

        f.create_group("t0_fields").attrs["field_names"] = []

        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = ["velocity"]
        vel_ds = t1.create_dataset(
            "velocity",
            shape=(n_split, T, W_out, H_out, 2),
            dtype=np.float32,
        )
        vel_ds.attrs["dim_varying"]    = np.array([True, True])
        vel_ds.attrs["time_varying"]   = True
        vel_ds.attrs["sample_varying"] = True

        f.create_group("t2_fields").attrs["field_names"] = []

        hdf[split_name] = f
        for p, orig_idx in enumerate(idxs):
            pos[int(orig_idx)] = (split_name, p)

        print(f"  Created {out_path}")

    # ── Stream trajectories one at a time ─────────────────────────────────
    print(f"\nProcessing trajectories ...")
    for i, row in enumerate(dataset):
        u = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
        v = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)

        # (T, H, W) → (T, W, H) for Well spatial ordering, then stack u/v
        vel = np.stack([u.transpose(0, 2, 1),
                        v.transpose(0, 2, 1)], axis=-1)   # (T, W, H, 2)

        if data_type == "numerical":
            # 2×2 box average: (T, 256, 128, 2) → (T, 128, 64, 2)
            vel = vel.reshape(T, 128, 2, 64, 2, 2).mean(axis=(2, 4))

        split_name, p = pos[i]
        hdf[split_name]["t1_fields"]["velocity"][p] = vel

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_traj}")

    for f in hdf.values():
        f.close()
    print(f"Done: {data_type}")


if __name__ == "__main__":
    convert("real")
    convert("numerical")
