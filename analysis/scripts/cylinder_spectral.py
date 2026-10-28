#!/usr/bin/env python3
"""
cylinder_spectral.py

Spectral comparison of real (PIV) vs numerical (CFD) cylinder wake data
from the RealPDEBench cylinder dataset.

Analyses
--------
1. Temporal power spectral density (PSD) of cross-stream velocity v at a
   downstream wake probe → Strouhal number extraction per trajectory.
2. Strouhal number vs Reynolds number for all matched real/numerical pairs.
3. Time-averaged 2-D isotropic spatial energy spectrum of kinetic energy
   (u² + v²)/2 → comparison of energy cascade structure.
4. PSD panel plots for a selection of matched Reynolds numbers.

Outputs (all saved to OUTPUT_DIR)
----------------------------------
  shedding_freq_data.npz         — arrays: re, fs_real, fs_num
  shedding_freq_vs_re.pdf        — f_s vs Re for all matched trajectories
  temporal_psd_comparison.pdf    — full bandwidth PSD (shows comb + noise gap)
  temporal_psd_lowfreq.pdf       — 0-15 Hz zoom (trustworthy PIV band)
  spatial_spectrum_comparison.pdf
  spatial_spectrum_panel.pdf
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from datasets import load_from_disk

# ── paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = f"{ROOT}/realpdebench/data/cylinder/hf_dataset"
OUTPUT_DIR = f"{ROOT}/realpdebench/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# cylinder geometry (from dataset metadata)
D_GRID = 18   # cylinder diameter in grid points

# ── matplotlib style ────────────────────────────────────────────────────────
def nature_style():
    plt.rcParams.update({
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset":     "dejavusans",
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "axes.linewidth":       0.8,
        "axes.labelsize":       9,
        "axes.titlesize":       9,
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.size":     3.0,
        "ytick.major.size":     3.0,
        "xtick.major.width":    0.8,
        "ytick.major.width":    0.8,
        "lines.linewidth":      1.5,
        "legend.frameon":       False,
        "legend.fontsize":      8,
    })

# ── 2-D isotropic spatial energy spectrum ──────────────────────────────────
def _prepare_shells(nx, ny):
    kx = (np.fft.fftfreq(nx) * nx).astype(int)
    ky = (np.fft.fftfreq(ny) * ny).astype(int)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    shell  = np.floor(np.sqrt(KX**2 + KY**2) + 0.5).astype(int)
    mmax   = shell.max()
    counts = np.bincount(shell.ravel(), minlength=mmax + 1).astype(np.float64)
    return shell, counts, mmax


def spatial_spectrum_2d(field2d, subtract_mean=True):
    """
    Isotropic 2-D power spectrum of a scalar field (nx, ny).
    Returns (m, E_per_mode) with DC excluded.
    """
    f = field2d - field2d.mean() if subtract_mean else field2d.copy()
    nx, ny  = f.shape
    shell, counts, mmax = _prepare_shells(nx, ny)
    norm = float(nx * ny) ** 2
    F    = np.fft.fft2(f)
    P    = 0.5 * np.abs(F) ** 2 / norm
    E    = np.bincount(shell.ravel(), weights=P.ravel(), minlength=mmax + 1)
    m    = np.arange(mmax + 1)
    den  = np.maximum(counts, 1.0)
    good = (m > 0) & np.isfinite(E)
    return m[good], (E / den)[good]


def avg_ke_spectrum(u4d, v4d, t_indices):
    """
    Time-averaged 2-D isotropic spectrum of kinetic energy (u²+v²)/2.
    u4d, v4d : (T, H, W) arrays — note spatial axes are (H, W) from raw data.
    t_indices : iterable of time indices to average over.
    """
    E_acc = None
    m_ref = None
    for ti in t_indices:
        ke = 0.5 * (u4d[ti].astype(np.float64) ** 2
                  + v4d[ti].astype(np.float64) ** 2)  # (H, W)
        m, E = spatial_spectrum_2d(ke)
        if E_acc is None:
            E_acc = E.copy()
            m_ref = m
        else:
            # interpolate onto common grid if sizes differ (shouldn't happen)
            E_acc += E
    return m_ref, E_acc / len(t_indices)


# ── temporal PSD ─────────────────────────────────────────────────────────────
def temporal_psd(signal_1d, dt, nperseg=None):
    """
    Welch PSD of a 1-D time series.  nperseg defaults to T//4.
    Returns (freq [Hz], psd).
    """
    T = len(signal_1d)
    if nperseg is None:
        nperseg = T // 4
    freq, psd = welch(
        signal_1d, fs=1.0 / dt,
        nperseg=nperseg, window="hann",
        detrend="linear", scaling="density",
    )
    return freq, psd


def shedding_peak(freq, psd, f_max=5.0, f_min=0.1):
    """
    Find the vortex shedding fundamental in [f_min, f_max] Hz via argmax.
    The dominant peak in this band is the fundamental shedding frequency.
    """
    mask = (freq >= f_min) & (freq <= f_max)
    if mask.sum() == 0:
        return np.nan, np.nan
    f_band   = freq[mask]
    psd_band = psd[mask]
    idx = np.argmax(psd_band)
    return float(f_band[idx]), float(psd_band[idx])


# ── data loading ─────────────────────────────────────────────────────────────
def load_trajectory(row):
    """
    Decode one HuggingFace Arrow row into numpy arrays.
    Returns a dict with u, v, t, x (1-D), y (1-D), re, and optionally vo / p.
    """
    T, H, W = row["shape_t"], row["shape_h"], row["shape_w"]
    u  = np.frombuffer(row["u"], dtype=np.float32).reshape(T, H, W)
    v  = np.frombuffer(row["v"], dtype=np.float32).reshape(T, H, W)
    t  = np.frombuffer(row["t"], dtype=np.float64)
    x2 = np.frombuffer(row["x"], dtype=np.float64).reshape(H, W)
    y2 = np.frombuffer(row["y"], dtype=np.float64).reshape(H, W)
    out = {
        "u":  u, "v": v, "t": t,
        "x":  x2[0, :],   # (W,) streamwise
        "y":  y2[:, 0],   # (H,) cross-stream
        "re": float(row["sim_id"].replace(".h5", "")),
        "T": T, "H": H, "W": W,
    }
    if "vo" in row and row["vo"] is not None:
        out["vo"] = np.frombuffer(row["vo"], dtype=np.float32).reshape(T, H, W)
    if "p" in row and row["p"] is not None:
        out["p"] = np.frombuffer(row["p"],  dtype=np.float32).reshape(T, H, W)
    return out


def build_re_index(dataset):
    """Return {reynolds_number: dataset_index} for all rows."""
    return {
        float(dataset[i]["sim_id"].replace(".h5", "")): i
        for i in range(len(dataset))
    }


# ── wake probe ────────────────────────────────────────────────────────────────
def wake_v_probe(traj, x_frac=(0.35, 0.75), y_band_frac=0.12):
    """
    Cross-stream velocity v spatially averaged over a downstream wake box.

    x_frac       : (start, end) as fraction of domain width
    y_band_frac  : half-width of the cross-stream band as fraction of H

    Returns 1-D time series of length T.
    """
    W, H = traj["W"], traj["H"]
    xi0  = int(x_frac[0] * W)
    xi1  = int(x_frac[1] * W)
    yc   = H // 2
    dy   = max(1, int(y_band_frac * H))
    return traj["v"][:, yc - dy : yc + dy, xi0:xi1].mean(axis=(1, 2))




# ── plotting helpers ──────────────────────────────────────────────────────────
COL_REAL = "#000000"   # black  → real / PIV
COL_NUM  = "#1f77b4"   # blue   → numerical / CFD


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    nature_style()

    print("Loading datasets …")
    ds_real = load_from_disk(f"{DATA_DIR}/real")
    ds_num  = load_from_disk(f"{DATA_DIR}/numerical")
    print(f"  Real: {len(ds_real)} trajectories")
    print(f"  Numerical: {len(ds_num)} trajectories")

    re_idx_real = build_re_index(ds_real)
    re_idx_num  = build_re_index(ds_num)
    common_re   = sorted(set(re_idx_real) & set(re_idx_num))
    print(f"  Matched Re values: {len(common_re)}")

    # ── 1. Shedding frequency vs Re (all matched trajectories) ───────────────
    print("\nComputing shedding frequencies …")
    re_list, fs_real_list, fs_num_list = [], [], []
    psd_cache = {}

    n_show  = 6
    re_show = np.array(common_re)[
        np.linspace(0, len(common_re) - 1, n_show, dtype=int)
    ]

    for re in common_re:
        tr = load_trajectory(ds_real[re_idx_real[re]])
        tn = load_trajectory(ds_num [re_idx_num [re]])

        dt_r = float(tr["t"][1] - tr["t"][0])
        dt_n = float(tn["t"][1] - tn["t"][0])
        fr, pr = temporal_psd(wake_v_probe(tr), dt_r)
        fn, pn = temporal_psd(wake_v_probe(tn), dt_n)
        fs_r, _ = shedding_peak(fr, pr)
        fs_n, _ = shedding_peak(fn, pn)

        re_list.append(re)
        fs_real_list.append(fs_r)
        fs_num_list.append(fs_n)

        if re in re_show:
            psd_cache[re] = {
                "fr": fr, "pr": pr, "fn": fn, "pn": pn,
                "fp_r": fs_r, "fp_n": fs_n,
            }

        print(f"  Re={re:>6.0f}  f_real={fs_r:.3f} Hz  f_num={fs_n:.3f} Hz")

    np.savez(
        f"{OUTPUT_DIR}/shedding_freq_data.npz",
        re=re_list, fs_real=fs_real_list, fs_num=fs_num_list,
    )

    # ── Plot 1: Shedding frequency vs Re ──────────────────────────────────
    re_arr  = np.array(re_list)
    fsr_arr = np.array(fs_real_list)
    fsn_arr = np.array(fs_num_list)

    fig, ax = plt.subplots(figsize=(88 / 25.4, 72 / 25.4), constrained_layout=True)
    ax.scatter(re_arr, fsr_arr, s=22, color=COL_REAL, label="Real (PIV)",
               zorder=3, linewidths=0)
    ax.scatter(re_arr, fsn_arr, s=22, color=COL_NUM,  label="Numerical (CFD)",
               marker="^", zorder=3, linewidths=0)

    # Annotate outliers: points > 2 MAD from the real-data median trend
    diff = np.abs(fsr_arr - fsn_arr)
    mad  = np.median(np.abs(diff - np.median(diff[np.isfinite(diff)])))
    thresh = np.median(diff[np.isfinite(diff)]) + 3 * mad
    for re, fr, fn in zip(re_arr, fsr_arr, fsn_arr):
        if np.isfinite(fr) and np.isfinite(fn) and abs(fr - fn) > thresh:
            ax.annotate(f"{re:.0f}", (re, max(fr, fn)),
                        fontsize=6, color="red",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Reynolds number")
    ax.set_ylabel(r"Shedding frequency $f_s$ (Hz)")
    ax.legend()
    ax.set_title("Vortex shedding frequency: real vs numerical")
    fig.savefig(f"{OUTPUT_DIR}/shedding_freq_vs_re.pdf")
    plt.close(fig)
    print("\nSaved shedding_freq_vs_re.pdf")

    # ── Plot 2a: Full PSD panel (full bandwidth, shows the gap + comb) ────
    ncols = 3
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(180 / 25.4, nrows * 55 / 25.4),
        constrained_layout=True,
    )
    axes = np.array(axes).ravel()

    for ax, re in zip(axes, re_show):
        if re not in psd_cache:
            ax.set_visible(False)
            continue
        c = psd_cache[re]
        # exclude DC (freq=0) for log x-axis
        mr = c["fr"] > 0; mn = c["fn"] > 0
        ax.loglog(c["fr"][mr], c["pr"][mr], color=COL_REAL, label="Real (PIV)")
        ax.loglog(c["fn"][mn], c["pn"][mn], color=COL_NUM,  label="Numerical (CFD)",
                    linestyle="--")
        if np.isfinite(c["fp_r"]):
            ax.axvline(c["fp_r"], color=COL_REAL, linewidth=0.7, linestyle=":")
        if np.isfinite(c["fp_n"]):
            ax.axvline(c["fp_n"], color=COL_NUM,  linewidth=0.7, linestyle=":")
        ax.set_title(f"Re = {re:.0f}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"PSD $[(\mathrm{m/s})^2/\mathrm{Hz}]$")
        ax.legend(fontsize=7)

    for ax in axes[len(re_show):]:
        ax.set_visible(False)

    fig.savefig(f"{OUTPUT_DIR}/temporal_psd_comparison.pdf")
    plt.close(fig)
    print("Saved temporal_psd_comparison.pdf")

    # ── Plot 2b: Zoomed PSD panel (0–15 Hz, trustworthy PIV band) ────────
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(180 / 25.4, nrows * 55 / 25.4),
        constrained_layout=True,
    )
    axes = np.array(axes).ravel()

    for ax, re in zip(axes, re_show):
        if re not in psd_cache:
            ax.set_visible(False)
            continue
        c    = psd_cache[re]
        fmin_plot, fmax = 0.05, 15.0
        mr   = (c["fr"] >= fmin_plot) & (c["fr"] <= fmax)
        mn   = (c["fn"] >= fmin_plot) & (c["fn"] <= fmax)
        ax.loglog(c["fr"][mr], c["pr"][mr], color=COL_REAL, label="Real (PIV)")
        ax.loglog(c["fn"][mn], c["pn"][mn], color=COL_NUM,  label="Numerical (CFD)",
                    linestyle="--")
        if np.isfinite(c["fp_r"]) and c["fp_r"] <= fmax:
            ax.axvline(c["fp_r"], color=COL_REAL, linewidth=0.8, linestyle=":",
                       label=f"$f_s$ real = {c['fp_r']:.2f} Hz")
        if np.isfinite(c["fp_n"]) and c["fp_n"] <= fmax:
            ax.axvline(c["fp_n"], color=COL_NUM,  linewidth=0.8, linestyle=":",
                       label=f"$f_s$ CFD = {c['fp_n']:.2f} Hz")
        ax.set_xlim(fmin_plot, fmax)
        ax.set_title(f"Re = {re:.0f}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"PSD $[(\mathrm{m/s})^2/\mathrm{Hz}]$")
        ax.legend(fontsize=6)

    for ax in axes[len(re_show):]:
        ax.set_visible(False)

    fig.savefig(f"{OUTPUT_DIR}/temporal_psd_lowfreq.pdf")
    plt.close(fig)
    print("Saved temporal_psd_lowfreq.pdf")

    # ── Plot 3: Spatial energy spectrum (time-averaged KE) ─────────────────
    # Use the median-Re trajectory for illustration; skip first quarter (transient)
    re_mid = common_re[len(common_re) // 2]
    print(f"\nComputing spatial spectra for Re={re_mid:.0f} …")
    tr = load_trajectory(ds_real[re_idx_real[re_mid]])
    tn = load_trajectory(ds_num [re_idx_num [re_mid]])

    t_start  = tr["T"] // 4
    n_avg    = 60
    t_inds_r = np.linspace(t_start, tr["T"] - 1, n_avg, dtype=int)
    t_inds_n = np.linspace(t_start, tn["T"] - 1, n_avg, dtype=int)

    m_r, E_r = avg_ke_spectrum(tr["u"], tr["v"], t_inds_r)
    m_n, E_n = avg_ke_spectrum(tn["u"], tn["v"], t_inds_n)

    fig, ax = plt.subplots(figsize=(88 / 25.4, 72 / 25.4), constrained_layout=True)
    ax.loglog(m_r, E_r, color=COL_REAL, label="Real (PIV)")
    ax.loglog(m_n, E_n, color=COL_NUM,  label="Numerical (CFD)", linestyle="--")

    ax.set_xlabel("Wavenumber mode $m$")
    ax.set_ylabel("KE / mode  $(\\mathrm{m/s})^2$")
    ax.set_title(f"Spatial KE spectrum — Re = {re_mid:.0f}")
    ax.legend()
    fig.savefig(f"{OUTPUT_DIR}/spatial_spectrum_comparison.pdf")
    plt.close(fig)
    print("Saved spatial_spectrum_comparison.pdf")

    # ── Plot 4: Spatial spectra for all Re_show panels ─────────────────────
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(180 / 25.4, nrows * 55 / 25.4),
        constrained_layout=True,
    )
    axes = np.array(axes).ravel()

    for ax, re in zip(axes, re_show):
        tr = load_trajectory(ds_real[re_idx_real[re]])
        tn = load_trajectory(ds_num [re_idx_num [re]])
        t_start = tr["T"] // 4
        t_inds  = np.linspace(t_start, tr["T"] - 1, n_avg, dtype=int)
        m_r, E_r = avg_ke_spectrum(tr["u"], tr["v"], t_inds)
        m_n, E_n = avg_ke_spectrum(tn["u"], tn["v"], t_inds)
        ax.loglog(m_r, E_r, color=COL_REAL, label="Real (PIV)")
        ax.loglog(m_n, E_n, color=COL_NUM,  label="Numerical (CFD)", linestyle="--")
    
        ax.set_title(f"Re = {re:.0f}")
        ax.set_xlabel("Mode $m$")
        ax.set_ylabel("KE / mode")
        ax.legend(fontsize=7)

    for ax in axes[len(re_show):]:
        ax.set_visible(False)

    fig.savefig(f"{OUTPUT_DIR}/spatial_spectrum_panel.pdf")
    plt.close(fig)
    print("Saved spatial_spectrum_panel.pdf")
    print("\nAll done. Outputs in:", OUTPUT_DIR)
    print("Key outputs:")
    print("  temporal_psd_comparison.pdf  — full bandwidth (shows comb + noise gap)")
    print("  temporal_psd_lowfreq.pdf     — 0-15 Hz zoom (trustworthy PIV band, shedding peak)")
    print("  shedding_freq_vs_re.pdf      — f_s vs Re for all matched trajectories")


if __name__ == "__main__":
    main()
