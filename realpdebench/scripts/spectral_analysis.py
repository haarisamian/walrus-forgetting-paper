import numpy as np
import matplotlib.pyplot as plt

#ML_path = "/mnt/home/polymathic/ceph/RTI_runs/Walrus_ft_all_long_finetune_golden_ckpt_globalnorm_3D_normset_first_128x3_lowres_100Epochs_200samples-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/finetune/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_normsetlowres_rollout_test_y_pred_epoch_101.npy"
#DNS_path = "/mnt/home/polymathic/ceph/RTI_runs/Walrus_ft_all_long_finetune_golden_ckpt_globalnorm_3D_normset_first_128x3_lowres_100Epochs_200samples-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/finetune/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_normsetlowres_rollout_test_y_ref_epoch_101.npy"
#ML_projected_path = "/mnt/ceph/users/polymathic/MPPX_logging/runs/Walrus_ft_all_long_finetune_golden_ckpt_globalnorm_3D_normset_first_128x3_lowres-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/finetune/0/viz/trial_plots_ML_projected.npy"

ML_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_128X3_DOWNSAMPLED_VALIDATION_25Feb_2026_epoch100-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_rollout_test_y_pred_epoch_101.npy"
DNS_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_128X3_DOWNSAMPLED_VALIDATION_25Feb_2026_epoch100-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_rollout_test_y_ref_epoch_101.npy"
#ML_projected_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_128X3_DOWNSAMPLED_VALIDATION_25Feb_2026_epoch100-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/trial_plots_ML_projected_128x3_downsampled.npy"

ML_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_128X3_DOWNSAMPLED_VALIDATION_25Feb_2026_epoch80_a100_40gb-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_rollout_test_y_pred_epoch_101.npy"
DNS_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_128X3_DOWNSAMPLED_VALIDATION_25Feb_2026_epoch80_a100_40gb-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_rollout_test_y_ref_epoch_101.npy"

orig_sim = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/NowSet_normalized_256x3/data/test/normSet[20250820]_256norm.h5"

#ML_path = "/mnt/home/polymathic/ceph/MPPX_logging/runs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_3D_data_MAE_gpu_vfull_500epochs-wella-full-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/finetune/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_rollout_valid_y_pred_epoch_500.npy"
#DNS_path = "/mnt/home/polymathic/ceph/MPPX_logging/runs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_3D_data_MAE_gpu_vfull_500epochs-wella-full-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/finetune/0/viz/batch_0_rank_0_rt_simulated_stefan_128x3_normalized_rollout_valid_y_ref_epoch_240.npy"

ML_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_256X3_DOWNSAMPLED_V2_4Apr_2026_VALIDATION_step100-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/batch_0_rank_0_rt_simulated_stefan_256x3_normalized_normsetv2_rollout_test_y_pred_epoch_101.npy"
DNS_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_256X3_DOWNSAMPLED_V2_4Apr_2026_VALIDATION_step100-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0001/0/viz/batch_0_rank_0_rt_simulated_stefan_256x3_normalized_normsetv2_rollout_test_y_ref_epoch_101.npy"

ML_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_256X3_DOWNSAMPLED_V2_4Apr_2026-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0003/finetune/1/viz/batch_0_rank_1_rt_simulated_stefan_256x3_normalized_normsetv2_rollout_test_y_pred_epoch_100.npy"
DNS_path = "/mnt/home/polymathic/ceph/Stefan3D_normalized_sims/outputs/Walrus_ft_all_long_finetune_old_ckpt_globalnorm_RTI_256X3_DOWNSAMPLED_V2_4Apr_2026-wella-delta-Isotr[Space-Adapt-Space-FullA-Axial]-AdamW-0.0003/finetune/1/viz/batch_0_rank_3_rt_simulated_stefan_256x3_normalized_normsetv2_rollout_test_y_ref_epoch_100.npy"

def mm_to_in(mm):
    return mm / 25.4

# ---------------- style ----------------
def nature():
    plt.rcParams.update({
        # fonts (Nature commonly uses a clean sans; Arial/Helvetica are fine)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,

        # axes / ticks
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,

        # lines / legend
        "lines.linewidth": 1.7,
        "legend.frameon": False,
        "legend.fontsize": 8,
    })

# ---------------- spectrum ----------------
def _prepare_shells(nx, ny):
    nx_i = (np.fft.fftfreq(nx) * nx).astype(int)
    ny_i = (np.fft.fftfreq(ny) * ny).astype(int)
    NXI, NYI = np.meshgrid(nx_i, ny_i, indexing="ij")

    k = np.sqrt(NXI**2 + NYI**2)
    m_shell = np.floor(k + 0.5).astype(int)  # stable near high k

    mmax = m_shell.max()
    counts = np.bincount(m_shell.ravel(), minlength=mmax + 1).astype(np.float64)
    return m_shell, counts

def spectrum_xy_components(
    filepath,
    tid=60,
    dx=1.0,
    dy=1.0,
    mode="midplane",          # "midplane" or "zavg"
    iz0=None,
    n_crop_z=0,
    subtract_xy_mean=True,
    per_mode=True,
):
    data = np.load(filepath)  # (T,nx,ny,nz,4): [c(or rho), vx, vy, vz]
    rho, vx, vy, vz = data[..., 0], data[..., 1], data[..., 2], data[..., 3]

    rho = rho[tid]
    vy = vy[tid]
    vz = vz[tid]

    if n_crop_z > 0:
        rho = rho[:, :, n_crop_z:-n_crop_z]
        vy  = vy[:, :, n_crop_z:-n_crop_z]
        vz  = vz[:, :, n_crop_z:-n_crop_z]

    nx, ny, nz = rho.shape
    Lx, Ly = nx * dx, ny * dy
    if abs(Lx - Ly) > 1e-12:
        raise ValueError("Assumes square x-y box (Lx==Ly).")

    m_shell, counts = _prepare_shells(nx, ny)
    mmax = len(counts) - 1
    norm = (nx * ny) ** 2

    def plane_spectrum(rho2, vy2, vz2):
        if subtract_xy_mean:
            rho2 = rho2 - rho2.mean()
            vy2  = vy2  - vy2.mean()
            vz2  = vz2  - vz2.mean()

        RHO = np.fft.fft2(rho2)
        VY  = np.fft.fft2(vy2)
        VZ  = np.fft.fft2(vz2)

        Prho = 0.5 * (np.abs(RHO) ** 2) / norm
        Py   = 0.5 * (np.abs(VY)  ** 2) / norm
        Pz   = 0.5 * (np.abs(VZ)  ** 2) / norm

        Erho = np.bincount(m_shell.ravel(), weights=Prho.ravel(), minlength=mmax + 1)
        Ey   = np.bincount(m_shell.ravel(), weights=Py.ravel(),   minlength=mmax + 1)
        Ez   = np.bincount(m_shell.ravel(), weights=Pz.ravel(),   minlength=mmax + 1)
        return Erho, Ey, Ez

    if mode == "midplane":
        if iz0 is None:
            iz0 = nz // 2
        Erho, Ey, Ez = plane_spectrum(rho[:, :, iz0], vy[:, :, iz0], vz[:, :, iz0])

    elif mode == "zavg":
        Erho = np.zeros(mmax + 1, dtype=np.float64)
        Ey   = np.zeros(mmax + 1, dtype=np.float64)
        Ez   = np.zeros(mmax + 1, dtype=np.float64)

        for iz in range(nz):
            erho, ey, ez = plane_spectrum(rho[:, :, iz], vy[:, :, iz], vz[:, :, iz])
            Erho += erho
            Ey   += ey
            Ez   += ez

        Erho /= max(nz, 1)
        Ey   /= max(nz, 1)
        Ez   /= max(nz, 1)
    else:
        raise ValueError('mode must be "midplane" or "zavg"')

    if per_mode:
        den = np.maximum(counts, 1.0)
        Erho = Erho / den
        Ey   = Ey   / den
        Ez   = Ez   / den

    m = np.arange(mmax + 1)
    good = (m > 0) & np.isfinite(Erho) & np.isfinite(Ey) & np.isfinite(Ez)
    return m[good], Erho[good], Ey[good], Ez[good]

# ---------------- main ----------------
nature()

dx = dy = 1.0
n_crop_z = 0
tid = 82

MODE = "zavg"
PER_MODE = True

M_MIN, M_MAX = 1, 64

# restrained color pair (edit these two hexes to taste)
COL_ML  = "#1f77b4"  # blue
COL_DNS = "#000000"  # black

m_ml, Erho_ml, Ey_ml, Ez_ml = spectrum_xy_components(
    ML_path, tid=tid, dx=dx, dy=dy,
    mode=MODE, n_crop_z=n_crop_z,
    subtract_xy_mean=True,
    per_mode=PER_MODE,
)

m_dn, Erho_dn, Ey_dn, Ez_dn = spectrum_xy_components(
    DNS_path, tid=tid, dx=dx, dy=dy,
    mode=MODE, n_crop_z=n_crop_z,
    subtract_xy_mean=True,
    per_mode=PER_MODE,
)

mmax = min(M_MAX, m_ml.max(), m_dn.max())
sel_ml = (m_ml >= M_MIN) & (m_ml <= mmax)
sel_dn = (m_dn >= M_MIN) & (m_dn <= mmax)

fig, axes = plt.subplots(
    1, 3,
    figsize=(mm_to_in(180), mm_to_in(60)),
    constrained_layout=True,
    sharex=True,
    sharey=True
)

for ax, title, Eml, Edn in [
    (axes[0], r"$\rho$", Erho_ml, Erho_dn),
    (axes[1], r"$v_y$", Ey_ml, Ey_dn),
    (axes[2], r"$v_z$", Ez_ml, Ez_dn),
]:
    ax.loglog(m_ml[sel_ml], Eml[sel_ml], label="Walrus", color=COL_ML)
    ax.loglog(m_dn[sel_dn], Edn[sel_dn], label=r"DNS (Downsampled 128$^3$)", color=COL_DNS)
    ax.set_ylim([1e-11, 5e-3])
    # -5/3 guide: thin, gray-ish (keep it subtle)
    mr = m_dn[sel_dn]
    if len(mr) >= 2:
        anchor = min(10, len(mr) - 1)
        guide = 5 * Edn[sel_dn][anchor] * (mr / mr[anchor]) ** (-5/3)
        ax.loglog(mr, guide, linestyle="--", linewidth=1.0, color="0.4")
        ax.text(mr[anchor], 6 * Edn[sel_dn][anchor], r"$m^{-5/3}$", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(r"Mode $m = kL/2\pi$")
    ax.minorticks_on()

ylabel = "Energy / mode" if PER_MODE else "Energy (shell sum)"
axes[0].set_ylabel(ylabel)

# single legend (cleaner than repeating)
axes[0].legend(loc="lower left")

out = f"cascade_curve_modes_ML_vs_DNS_{MODE}_{'permode' if PER_MODE else 'shellsum'}_rho_vy_vz_128x3_downsampled_epoch100_final_runs.pdf"
plt.savefig(out)
plt.show()
print("Saved:", out)