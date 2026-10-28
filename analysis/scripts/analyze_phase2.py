"""
Phase-2 forgetting across all 10 Well datasets.

Answers three things the n=4 set could not:
  1. Does the Phase-1 law (damage ~ persistence / d_eff) transfer to Phase 2?
  2. Is peak-and-recover unique to turbcool, or a property of a class?
  3. If a class, what distinguishes it -- turbulent physics, dimensionality,
     dynamism, or overlap with the Phase-2 update subspace?

Damage is measured in VRMSE (offset-invariant). Recovery is (peak - end) as a
fraction of the rise from the ep51 baseline, so it is comparable across
datasets with different damage magnitudes.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob, io, pickle, re
import numpy as np, torch
from scipy.stats import spearmanr
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                                      weights_only=False)
FE = f"{ROOT}/walrus/experiments/forget_eval_results"
GR = f"{ROOT}/walrus/experiments/activation_grams_v2"
PERS = {"ac.disc":0.1614,"ac.incl":0.2142,"ac.maze":0.9003,"act.mat":0.3501,
        "helmholtz":0.1283,"MHD-64":0.2273,"planetSWE":0.0246,"shear":0.0282,
        "TRL-2D":0.2189,"turbcool":0.1387}
# turbulent = genuinely turbulent hydrodynamics; wave = propagation-dominated
CLASS = {"ac.disc":"wave","ac.incl":"wave","ac.maze":"wave","helmholtz":"wave",
         "planetSWE":"geo","shear":"turb","act.mat":"active","MHD-64":"turb",
         "TRL-2D":"turb","turbcool":"turb"}
DS = [("acoustic_scattering_discontinuous","ac.disc"),("acoustic_scattering_inclusions","ac.incl"),
      ("acoustic_scattering_maze","ac.maze"),("helmholtz_staircase","helmholtz"),
      ("planetswe","planetSWE"),("shear_flow","shear"),("active_matter","act.mat"),
      ("MHD_64","MHD-64"),("trl2d","TRL-2D"),("turbcool","turbcool")]
CK = [("PT","pretrained"),("ep51","phase1_gn_lr5e5"),
      ("ep111","gn_to_gn_step60"),("ep151","gn_to_gn_last_ep90")]

def met(ck, ds, M="VRMSE"):
    fs = glob.glob(f"{FE}/{ck}/{ds}/**/valid_loss_dict_*.pkl", recursive=True)
    if not fs: return None
    D = pickle.load(open(fs[0], "rb")); v = []
    for k, x in D.items():
        m = re.match(rf".*/(.+)_{M}_T=all_mean$", k)
        if m and m.group(1) != "full":
            v.append(float(x.item() if hasattr(x, "item") else x))
    return float(np.mean(v)) if v else None

def effdim(ds, arm="phase1", b=10):
    Z = np.load(f"{GR}/{arm}/gram_{ds}.npz"); i = list(Z["blocks"]).index(b)
    C = torch.from_numpy(Z[f"block{b}"]).double(); C = (C + C.T) / 2
    s = torch.from_numpy(Z[f"sum{b}"]).double(); N = float(Z["ntok"][i]); mu = s / N
    C = (C - N * torch.outer(mu, mu)) / N
    ev = torch.clip(torch.linalg.eigvalsh(C), 0, None); p = ev / ev.sum()
    return float((p.sum() ** 2) / (p ** 2).sum())

rows, missing = [], []
for ds, lab in DS:
    vals = {n: met(ck, ds) for n, ck in CK}
    r = {n: met(ck, ds, "PearsonR") for n, ck in CK}
    if any(v is None for v in vals.values()):
        missing.append((lab, [n for n, v in vals.items() if v is None])); continue
    p2 = vals["ep151"] - vals["ep51"]
    rise = max(vals["ep111"] - vals["ep51"], 1e-9)
    rec = (vals["ep111"] - vals["ep151"]) / rise      # >0 means it recovered
    rows.append(dict(lab=lab, cls=CLASS[lab], p1=vals["ep51"]-vals["PT"], p2=p2,
                     rec=rec, dr=r["ep151"]-r["ep111"],
                     eff=effdim(ds), pers=PERS[lab], **{f"v_{n}": vals[n] for n, _ in CK}))
if missing:
    print("MISSING (job still running):")
    for l, m in missing: print(f"   {l}: {m}")
    print()
if len(rows) < 4: raise SystemExit("too few datasets yet")

print(f"PHASE-2 FORGETTING, n={len(rows)}\n")
print(f"  {'dataset':11}{'class':>8}{'PT':>8}{'ep51':>8}{'ep111':>8}{'ep151':>8}"
      f"{'P1 dam':>9}{'P2 dam':>9}{'recov':>8}{'dRho':>9}")
for r in sorted(rows, key=lambda r: -r["p2"]):
    print(f"  {r['lab']:9}{r['cls']:>8}{r['v_PT']:8.4f}{r['v_ep51']:8.4f}"
          f"{r['v_ep111']:8.4f}{r['v_ep151']:8.4f}{r['p1']:9.4f}{r['p2']:9.4f}"
          f"{100*r['rec']:7.1f}%{r['dr']:+9.4f}")

d2 = [r["p2"] for r in rows]
print(f"\nDoes the Phase-1 law transfer? (predictors vs Phase-2 damage)")
for nm, x in [("d_eff at Phase-1 ckpt", [r["eff"] for r in rows]),
              ("persistence", [r["pers"] for r in rows]),
              ("Phase-1 damage", [r["p1"] for r in rows])]:
    rho, p = spearmanr(x, d2)
    print(f"  {nm:24} rho={rho:+.3f}  p={p:.4f}{'  *' if p<0.05 else ''}")
lD = np.log10(np.clip(d2, 1e-6, None))
X = np.c_[np.ones(len(rows)), np.log10([r["eff"] for r in rows]),
          np.log10([r["pers"] for r in rows])]
beta, *_ = np.linalg.lstsq(X, lD, rcond=None); pr = X @ beta
r2 = 1 - ((lD - pr) ** 2).sum() / ((lD - lD.mean()) ** 2).sum()
print(f"  R^2 of the two-factor form refit on Phase 2 = {100*r2:.0f}%")

print(f"\nIs peak-and-recover a class property?")
for c in sorted({r["cls"] for r in rows}):
    g = [r for r in rows if r["cls"] == c]
    print(f"  {c:8} n={len(g)}  mean recovery {100*np.mean([x['rec'] for x in g]):6.1f}%"
          f"   mean dPearsonR {np.mean([x['dr'] for x in g]):+.4f}")
rec = [r["rec"] for r in rows]
for nm, x in [("persistence", [r["pers"] for r in rows]),
              ("d_eff", [r["eff"] for r in rows]),
              ("Phase-1 damage", [r["p1"] for r in rows])]:
    rho, p = spearmanr(x, rec)
    print(f"  recovery vs {nm:18} rho={rho:+.3f}  p={p:.4f}")
