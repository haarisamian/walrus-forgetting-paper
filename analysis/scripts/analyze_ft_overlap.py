"""
Does representational overlap between the FINE-TUNING data and a held-out
dataset predict how much that dataset is forgotten?

Motivation: d_eff and R are properties of the forgotten dataset and of dW.
Neither names the fine-tuning distribution explicitly. But what we train ON
should matter -- and we have two different fine-tuning targets (cylinder
simulation in Phase I, real PIV in Phase II) driving two different damage
orderings, which is the natural experiment.

Measures, all in the PRETRAINED model's representation so every dataset is
described in one common frame:

  CKA(A,B) = <C_A, C_B>_F / (||C_A||_F ||C_B||_F)
      Linear centered kernel alignment on the activation covariances. Symmetric,
      scale-free, 1 iff the two covariances are proportional. This is "how
      similarly does the model represent these two datasets".

  ovl_k(D <- F) = tr(V_F,k^T C_D V_F,k) / tr(C_D)
      Fraction of dataset D's activation energy that lies in the top-k
      eigendirections of the fine-tuning set F. Asymmetric and the more
      mechanistic of the two: it asks how much of D lives where F will be
      rewritten.

RAW OVERLAP MUST NOT BE REGRESSED DIRECTLY. Measured on the ten Well datasets
against each other, a dataset's MEAN overlap with everything else tracks d_eff
at rho = -0.964 (CKA: -0.976) -- concentrated datasets have high overlap with
everything for purely geometric reasons. Regressing the raw value would
re-derive the d_eff factor and present it as a new one. A two-way
decomposition of log ovl over that 10x10 matrix gives roughly 25% row effect
(the forgotten dataset), 41% column effect (the fine-tuning set) and 28%
irreducibly pairwise interaction. So we divide out each dataset's own baseline:

  novl(D <- F) = ovl_k(D <- F) / mean_G ovl_k(D <- G)

with G running over the ten Well datasets. This removes D's geometric baseline
and leaves the part that is specific to the pair, which is the only part d_eff
and persistence cannot already express.

Two competing hypotheses, and they predict opposite signs:
  (a) overlap -> PROTECTION. Shared directions keep being trained, so they stay
      functional and D survives.
  (b) overlap -> DAMAGE. Shared directions get overwritten with an F-specific
      mapping, so D is hit hardest exactly where it overlaps.
We do not assume one; the sign is the result.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob, io, pickle, re, sys
import numpy as np
import torch
from scipy.stats import spearmanr

torch.storage._load_from_bytes = lambda b: torch.load(
    io.BytesIO(b), map_location="cpu", weights_only=False)

EXP = f"{ROOT}/walrus/experiments"
FE = f"{EXP}/forget_eval_results"
GRAM = f"{EXP}/activation_grams_v2"
BLOCKS = [0, 5, 10, 15, 20, 25, 30]
TOPK = 10

DS = [("acoustic_scattering_discontinuous", "ac.disc"),
      ("acoustic_scattering_inclusions", "ac.incl"),
      ("acoustic_scattering_maze", "ac.maze"),
      ("helmholtz_staircase", "helmholtz"), ("planetswe", "planetSWE"),
      ("shear_flow", "shear"), ("active_matter", "act.mat"),
      ("MHD_64", "MHD-64"), ("trl2d", "TRL-2D"), ("turbcool", "turbcool")]

PERS = {"ac.disc": .1614, "ac.incl": .2142, "ac.maze": .9003, "act.mat": .3501,
        "helmholtz": .1283, "MHD-64": .2273, "planetSWE": .0246, "shear": .0282,
        "TRL-2D": .2189, "turbcool": .1387}
DEFF = {"ac.disc": 1.62, "ac.incl": 1.65, "ac.maze": 1.86, "planetSWE": 1.89,
        "helmholtz": 2.02, "TRL-2D": 2.24, "turbcool": 2.94, "act.mat": 5.07,
        "shear": 5.65, "MHD-64": 18.91}
RPH1 = {"ac.disc": .08321, "ac.incl": .08234, "ac.maze": .07792,
        "helmholtz": .07182, "planetSWE": .08634, "shear": .06102,
        "act.mat": .04503, "MHD-64": .03896, "TRL-2D": .07206, "turbcool": .06552}


def cov(arm, ds, b):
    """Centred activation covariance. See capture_activations.py -- the first
    moment is stored precisely so this centring is exact rather than a
    deflate-the-top-eigenvector approximation."""
    f = f"{GRAM}/{arm}/gram_{ds}.npz"
    Z = np.load(f)
    i = list(Z["blocks"]).index(b)
    C = torch.from_numpy(Z[f"block{b}"]).double()
    C = (C + C.T) / 2
    s = torch.from_numpy(Z[f"sum{b}"]).double()
    N = float(Z["ntok"][i])
    mu = s / N
    return (C - N * torch.outer(mu, mu)) / N


def vrmse(ck, ds):
    fs = glob.glob(f"{FE}/{ck}/{ds}/**/valid_loss_dict_*.pkl", recursive=True)
    if not fs:
        return None
    D = pickle.load(open(fs[0], "rb"))
    v = [float(x.item() if hasattr(x, "item") else x)
         for k, x in D.items()
         if (m := re.match(r".*/(.+)_VRMSE_T=all_mean$", k)) and m.group(1) != "full"]
    return float(np.mean(v)) if v else None


def cka(A, B):
    return float((A * B).sum() / (A.norm() * B.norm()))


def energy_in(C_d, C_f, k=TOPK):
    ev, V = torch.linalg.eigh(C_f)
    V = V[:, torch.argsort(ev, descending=True)[:k]]
    return float(torch.trace(V.T @ C_d @ V) / torch.trace(C_d))


def block_mean(fn, arm, ds, ft):
    return float(np.mean([fn(cov(arm, ds, b), cov(arm, ft, b)) for b in BLOCKS]))


def ols(y, *cols):
    X = np.c_[np.ones(len(y)), *cols]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    k = X.shape[1] - 1
    return r2, 1 - (1 - r2) * (len(y) - 1) / (len(y) - k - 1), b


def main():
    for f in ("cylinder_numerical", "cylinder_real_cl"):
        if not glob.glob(f"{GRAM}/pretrained/gram_{f}.npz"):
            sys.exit(f"missing {f} capture -- job 282324 not finished")

    # Per-dataset geometric baseline: mean overlap / CKA against the ten Well
    # datasets. Divides out the "concentrated datasets overlap everything"
    # confound so what remains is pair-specific.
    base_o, base_c = {}, {}
    for ds, l in DS:
        others = [o for o, _ in DS if o != ds]
        base_o[l] = float(np.mean([block_mean(energy_in, "pretrained", ds, o)
                                   for o in others]))
        base_c[l] = float(np.mean([block_mean(cka, "pretrained", ds, o)
                                   for o in others]))

    lab, d1, d2, cka_s, cka_r, ovl_s, ovl_r = [], [], [], [], [], [], []
    for ds, l in DS:
        p = vrmse("pretrained", ds)
        a = vrmse("phase1_gn_lr5e5", ds)
        b = vrmse("gn_to_gn_last_ep90", ds)
        lab.append(l)
        d1.append(None if None in (p, a) else a - p)
        d2.append(None if None in (p, b) else b - p)
        cka_s.append(block_mean(cka, "pretrained", ds, "cylinder_numerical") / base_c[l])
        cka_r.append(block_mean(cka, "pretrained", ds, "cylinder_real_cl") / base_c[l])
        ovl_s.append(block_mean(energy_in, "pretrained", ds, "cylinder_numerical") / base_o[l])
        ovl_r.append(block_mean(energy_in, "pretrained", ds, "cylinder_real_cl") / base_o[l])

    print("Representational overlap with the fine-tuning data "
          f"(pretrained frame, blocks {BLOCKS}, top-{TOPK}, row-normalised)\n")
    print(f"  {'dataset':11}{'CKA sim':>9}{'CKA PIV':>9}{'ovl sim':>9}"
          f"{'ovl PIV':>9}{'dmg I':>9}{'dmg II':>9}")
    order = np.argsort([-(x if x is not None else -1) for x in d1])
    for i in order:
        s1 = f"{d1[i]:9.4f}" if d1[i] is not None else f"{'--':>9}"
        s2 = f"{d2[i]:9.4f}" if d2[i] is not None else f"{'--':>9}"
        print(f"  {lab[i]:9}{cka_s[i]:9.3f}{cka_r[i]:9.3f}"
              f"{ovl_s[i]:9.3f}{ovl_r[i]:9.3f}{s1}{s2}")

    A = {k: np.array(v) for k, v in
         [("cka_sim", cka_s), ("cka_piv", cka_r),
          ("ovl_sim", ovl_s), ("ovl_piv", ovl_r)]}
    # sanity: normalisation should have broken the d_eff collinearity
    _de = np.array([DEFF[l] for l in lab])
    print("  (normalised features vs d_eff -- should now be weak: " +
          ", ".join(f"{k} {spearmanr(v, _de)[0]:+.2f}" for k, v in A.items()) + ")\n")
    dmg1 = np.array([x if x is not None else np.nan for x in d1])
    dmg2 = np.array([x if x is not None else np.nan for x in d2])
    de = np.array([DEFF[l] for l in lab])
    pe = np.array([PERS[l] for l in lab])
    rr = np.array([RPH1[l] for l in lab])

    print("\nMarginal rank correlation with Phase-I damage "
          "(sign is the hypothesis test):")
    for k, v in A.items():
        rho, p = spearmanr(v, dmg1)
        print(f"  {k:9} rho={rho:+.3f}  p={p:.4f}{'  *' if p < .05 else ''}")
    print(f"  {'d_eff':9} rho={spearmanr(de, dmg1)[0]:+.3f}   "
          f"(reference)  R rho={spearmanr(rr, dmg1)[0]:+.3f}")

    print("\nIs overlap just a restatement of d_eff / R?")
    for k, v in A.items():
        print(f"  {k:9} vs d_eff rho={spearmanr(v, de)[0]:+.3f}   "
              f"vs R rho={spearmanr(v, rr)[0]:+.3f}")

    print("\nDoes overlap add anything to the two-factor law?")
    y = np.log10(dmg1)
    base = ols(y, np.log10(de), np.log10(pe))
    print(f"  d_eff + persistence            R2={100*base[0]:5.1f}%  "
          f"adj={100*base[1]:5.1f}%")
    for k, v in A.items():
        r2, adj, _ = ols(y, np.log10(de), np.log10(pe), v)
        print(f"  + {k:26} R2={100*r2:5.1f}%  adj={100*adj:5.1f}%  "
              f"{'GAIN' if adj > base[1] else 'no gain'}")

    print("\nThe natural experiment: fine-tuning target CHANGES between phases.")
    print("  If overlap-with-target drives damage, the sim->PIV shift in overlap")
    print("  should track the Phase-I -> Phase-II shift in damage.")
    m = ~np.isnan(dmg2)
    if m.sum() >= 4:
        dr = dmg2[m] / dmg1[m]
        for nm, num, den in [("CKA", A["cka_piv"], A["cka_sim"]),
                             ("ovl", A["ovl_piv"], A["ovl_sim"])]:
            ratio = num[m] / den[m]
            rho, p = spearmanr(ratio, dr)
            print(f"  {nm} PIV/sim  vs  damage II/I:  rho={rho:+.3f}  "
                  f"p={p:.4f}   (n={m.sum()})")
    else:
        print(f"  only {m.sum()} datasets have Phase-II damage yet -- "
              "needs job 282281")


if __name__ == "__main__":
    main()
