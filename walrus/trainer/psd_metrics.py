"""Dynamics metrics for rollout evaluation.

Three complementary metrics that expose delta collapse / frozen-field failure:

  1. Temporal PSD L2 error  — collapsed field has all power at DC (f=0)
  2. Delta VRMSE            — VRMSE of frame-to-frame differences; frozen field
                              predicts ~0 delta → huge error
  3. Fluctuation VRMSE      — subtract time-mean before VRMSE; frozen field has
                              no fluctuations → VRMSE_fluc ≈ 1
"""
import numpy as np
import os
import torch


# ── 1. Temporal PSD ─────────────────────────────────────────────────────────

def compute_rollout_psd(y: torch.Tensor) -> torch.Tensor:
    """Spatially-averaged one-sided temporal PSD.

    Args:
        y: (B, T, *spatial, C) float tensor on any device
    Returns:
        psd: (F, C) where F = T//2 + 1
    """
    B, T, C = y.shape[0], y.shape[1], y.shape[-1]
    y_flat = y.reshape(B, T, -1, C)              # (B, T, H_flat, C)
    yf = torch.fft.rfft(y_flat.float(), dim=1)   # (B, F, H_flat, C)
    psd = (yf.abs() ** 2) / T                    # Parseval normalisation
    return psd.mean(dim=(0, 2))                  # (F, C)


def psd_l2_error(psd_pred: torch.Tensor, psd_ref: torch.Tensor) -> torch.Tensor:
    """Integrated L2 distance between two PSD curves.

    Args:
        psd_pred, psd_ref: (F, C) tensors
    Returns:
        error: (C,) — one scalar per channel
    """
    return torch.sqrt(((psd_pred - psd_ref) ** 2).sum(dim=0))


def save_psd_results(
    psd_pred_mean: torch.Tensor,
    psd_ref_mean: torch.Tensor,
    viz_folder: str,
    dataset_name: str,
    epoch: int,
    rank: int,
    valid_or_test: str,
    field_names: list,
):
    """Save PSD arrays to the same rollout_losses directory used by other metrics.

    Saves per-channel:
      velocity_x_psd_pred_rollout_mean.npy  (F,)
      velocity_x_psd_gt_rollout_mean.npy    (F,)
      velocity_x_psd_l2_error.npy           scalar
      (same for velocity_y, and 'full' averaged over channels)
    """
    epoch_dir = os.path.join(
        viz_folder,
        dataset_name,
        "rollout_losses",
        f"epoch_{epoch}_rank{rank}_{valid_or_test}",
    )
    os.makedirs(epoch_dir, exist_ok=True)

    pred_np = psd_pred_mean.cpu().numpy()   # (F, C)
    ref_np  = psd_ref_mean.cpu().numpy()    # (F, C)
    l2_np   = psd_l2_error(psd_pred_mean, psd_ref_mean).cpu().numpy()  # (C,)

    # Per-channel
    for c, fname in enumerate(field_names):
        tag = fname.replace(" ", "_")
        np.save(os.path.join(epoch_dir, f"{tag}_psd_pred_rollout_mean.npy"), pred_np[:, c])
        np.save(os.path.join(epoch_dir, f"{tag}_psd_gt_rollout_mean.npy"),   ref_np[:, c])
        np.save(os.path.join(epoch_dir, f"{tag}_psd_l2_error.npy"),          l2_np[c])

    # Full (mean over channels) — useful as a single scalar
    np.save(os.path.join(epoch_dir, "full_psd_pred_rollout_mean.npy"), pred_np.mean(axis=1))
    np.save(os.path.join(epoch_dir, "full_psd_gt_rollout_mean.npy"),   ref_np.mean(axis=1))
    np.save(os.path.join(epoch_dir, "full_psd_l2_error.npy"),          float(l2_np.mean()))


# ── 2. Delta VRMSE ──────────────────────────────────────────────────────────

def compute_delta_vrmse(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-timestep VRMSE of frame-to-frame differences.

    A frozen/collapsed prediction has delta_pred ≈ 0, so the numerator equals
    the denominator and delta_VRMSE ≈ 1 everywhere.  A model that learned
    dynamics has delta_VRMSE << 1.

    Args:
        y_pred, y_ref: (B, T, H, W, C)
        eps: denominator stabiliser
    Returns:
        delta_vrmse: (T-1, C) — one value per temporal step per channel
    """
    B, T, C = y_pred.shape[0], y_pred.shape[1], y_pred.shape[-1]
    yp = y_pred.reshape(B, T, -1, C)              # (B, T, H_flat, C)
    yr = y_ref.reshape(B, T, -1, C)

    delta_pred = yp[:, 1:] - yp[:, :-1]           # (B, T-1, H_flat, C)
    delta_ref  = yr[:, 1:] - yr[:, :-1]
    err2 = (delta_pred - delta_ref).float() ** 2
    ref2 = delta_ref.float() ** 2

    # Mean over (B, H_flat) for each timestep and channel
    num = err2.mean(dim=(0, 2)).sqrt()             # (T-1, C)
    den = ref2.mean(dim=(0, 2)).sqrt() + eps       # (T-1, C)
    return num / den                               # (T-1, C)


def save_delta_vrmse(
    delta_vrmse_mean: torch.Tensor,
    viz_folder: str,
    dataset_name: str,
    epoch: int,
    rank: int,
    valid_or_test: str,
    field_names: list,
):
    """Save delta VRMSE arrays.

    Saves per-channel:
      velocity_x_delta_VRMSE_rollout_mean.npy  (T-1,)
      full_delta_VRMSE_rollout_mean.npy         (T-1,) — mean over channels
      full_delta_VRMSE_scalar.npy               scalar — time & channel mean
    """
    epoch_dir = os.path.join(
        viz_folder,
        dataset_name,
        "rollout_losses",
        f"epoch_{epoch}_rank{rank}_{valid_or_test}",
    )
    os.makedirs(epoch_dir, exist_ok=True)

    arr = delta_vrmse_mean.cpu().numpy()   # (T-1, C)
    for c, fname in enumerate(field_names):
        tag = fname.replace(" ", "_")
        np.save(os.path.join(epoch_dir, f"{tag}_delta_VRMSE_rollout_mean.npy"), arr[:, c])

    full = arr.mean(axis=1)               # (T-1,)
    np.save(os.path.join(epoch_dir, "full_delta_VRMSE_rollout_mean.npy"), full)
    np.save(os.path.join(epoch_dir, "full_delta_VRMSE_scalar.npy"),       float(full.mean()))


# ── 3. Fluctuation VRMSE (u'VRMSE) ─────────────────────────────────────────

def compute_fluctuation_vrmse(
    y_pred: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-timestep VRMSE of the fluctuating (time-mean-subtracted) components.

    For a frozen model: y_pred ≈ constant → y_pred_fluc ≈ 0.
    Numerator = sqrt(mean(y_ref_fluc^2)) = denominator → fluct_VRMSE ≈ 1.
    For a model that captures wake dynamics: fluct_VRMSE << 1.

    Both pred and ref are demeaned w.r.t. their own time-mean before
    computing VRMSE so we measure oscillation accuracy independently of
    any mean-field bias.

    Args:
        y_pred, y_ref: (B, T, H, W, C)
        eps: denominator stabiliser
    Returns:
        fluct_vrmse: (T, C)
    """
    B, T, C = y_pred.shape[0], y_pred.shape[1], y_pred.shape[-1]
    yp = y_pred.reshape(B, T, -1, C).float()      # (B, T, H_flat, C)
    yr = y_ref.reshape(B, T, -1, C).float()

    pred_fluc = yp - yp.mean(dim=1, keepdim=True)  # (B, T, H_flat, C)
    ref_fluc  = yr - yr.mean(dim=1, keepdim=True)

    err2 = (pred_fluc - ref_fluc) ** 2
    ref2 = ref_fluc ** 2

    num = err2.mean(dim=(0, 2)).sqrt()             # (T, C)
    den = ref2.mean(dim=(0, 2)).sqrt() + eps       # (T, C)
    return num / den                               # (T, C)


def save_fluctuation_vrmse(
    fluct_vrmse_mean: torch.Tensor,
    viz_folder: str,
    dataset_name: str,
    epoch: int,
    rank: int,
    valid_or_test: str,
    field_names: list,
):
    """Save fluctuation VRMSE arrays.

    Saves per-channel:
      velocity_x_fluct_VRMSE_rollout_mean.npy  (T,)
      full_fluct_VRMSE_rollout_mean.npy         (T,) — mean over channels
      full_fluct_VRMSE_scalar.npy               scalar — time & channel mean
    """
    epoch_dir = os.path.join(
        viz_folder,
        dataset_name,
        "rollout_losses",
        f"epoch_{epoch}_rank{rank}_{valid_or_test}",
    )
    os.makedirs(epoch_dir, exist_ok=True)

    arr = fluct_vrmse_mean.cpu().numpy()   # (T, C)
    for c, fname in enumerate(field_names):
        tag = fname.replace(" ", "_")
        np.save(os.path.join(epoch_dir, f"{tag}_fluct_VRMSE_rollout_mean.npy"), arr[:, c])

    full = arr.mean(axis=1)               # (T,)
    np.save(os.path.join(epoch_dir, "full_fluct_VRMSE_rollout_mean.npy"), full)
    np.save(os.path.join(epoch_dir, "full_fluct_VRMSE_scalar.npy"),       float(full.mean()))
