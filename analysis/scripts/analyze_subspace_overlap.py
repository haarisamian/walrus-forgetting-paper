"""
Causal representational test of forgetting.

Phase 1's weight update dW is low-rank and lands on the HIGHEST-variance
directions of the pretrained weights (median W-rank 57/1408 in block 0). If
that overwrite is what causes forgetting, a dataset's damage should scale with
how much of its activation energy lives in the overwritten subspace.

For each probe block:
    V_k = top-k right singular vectors of dW      ("the cylinder subspace")
    C   = activation Gram matrix for a dataset    (from capture_activations.py)
    overlap = tr(V_k C V_k^T) / tr(C)             in [0,1]

Then correlate overlap against measured Phase-1 damage across datasets.
A NULL control uses k random orthonormal directions, so we can tell a real
alignment from the overlap any k-dim subspace would give.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob, io, pickle, re
import numpy as np, torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from scipy.stats import spearmanr
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu",
                                                      weights_only=False)
EXP = f"{ROOT}/walrus/experiments"
P1  = (f"{EXP}/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]"
       "-AdamW-5e-05/finetune/0/checkpoints/step_51")
GRAM = f"{EXP}/activation_grams"
FE   = f"{EXP}/forget_eval_results"
BLOCKS = [0, 10, 20, 30, 39]
K = 32
DS = [("acoustic_scattering_discontinuous","ac.disc","wave"),
      ("acoustic_scattering_inclusions","ac.incl","wave"),
      ("acoustic_scattering_maze","ac.maze","wave"),
      ("helmholtz_staircase","helmholtz","wave"),
      ("planetswe","planetSWE","geo"),
      ("shear_flow","shear","fluid"),
      ("active_matter","act.mat","fluid"),
      ("MHD_64","MHD-64","mhd"),
      ("trl2d","TRL-2D","fluid"),
      ("turbcool","turbcool","fluid")]

def damage(ds, M="VRMSE"):
    out = []
    for ck in ("pretrained", "phase1_dyn_step20"):
        fs = glob.glob(f"{FE}/{ck}/{ds}/**/valid_loss_dict_*.pkl", recursive=True)
        D = pickle.load(open(fs[0], "rb")); v = []
        for k, x in D.items():
            m = re.match(rf".*/(.+)_{M}_T=all_mean$", k)
            if m and m.group(1) != "full":
                v.append(float(x.item() if hasattr(x, "item") else x))
        out.append(np.mean(v))
    return out[1] - out[0]

# The dW SVD is expensive (~10 min) and does not change between runs, so cache it.
CACHE = "/tmp/dw_subspaces.pt"
import os
if os.path.exists(CACHE):
    _c = torch.load(CACHE)
    Vk, Vrand = _c["Vk"], _c["Vrand"]
    print(f"loaded cached dW subspaces for blocks {sorted(Vk)}", flush=True)
else:
    print("loading pretrained + phase-1 weights ...", flush=True)
    pre = torch.load(f"{ROOT}/walrus_model/checkpoints/walrus.pt",
                     map_location="cpu", weights_only=False, mmap=True)["app"]["model"]
    KEYS = [f"blocks.{b}.space_mixing.fused_ff_qkv.weight" for b in BLOCKS]
    sd = {f"app.model.{k}": torch.empty_like(pre[k].float()) for k in KEYS}
    dcp.load_state_dict(state_dict=sd, storage_reader=FileSystemReader(P1),
                        planner=DefaultLoadPlanner(allow_partial_load=True), no_dist=True)
    Vk, Vrand = {}, {}
    g = torch.Generator().manual_seed(0)
    for b, k in zip(BLOCKS, KEYS):
        W = pre[k].float().flatten(1); dW = sd[f"app.model.{k}"].flatten(1) - W
        _, _, V = torch.linalg.svd(dW, full_matrices=False)
        Vk[b] = V[:K]
        Q, _ = torch.linalg.qr(torch.randn(V.shape[1], K, generator=g))
        Vrand[b] = Q.T
        print(f"  block {b}: dW subspace ready", flush=True)
    torch.save({"Vk": Vk, "Vrand": Vrand}, CACHE)
    print(f"cached to {CACHE}", flush=True)

rows = []
for ds, lab, grp in DS:
    f = f"{GRAM}/gram_{ds}.npz"
    try: Z = np.load(f)
    except FileNotFoundError:
        print(f"  [missing] {lab}"); continue
    ov, ovr, ovd = {}, {}, {}
    for b in BLOCKS:
        key = f"block{b}"
        if key not in Z: continue
        C = torch.from_numpy(Z[key]).float()
        C = (C + C.T) / 2
        tr = torch.trace(C).item()
        ov[b]  = float((Vk[b] @ C @ Vk[b].T).trace() / tr)
        ovr[b] = float((Vrand[b] @ C @ Vrand[b].T).trace() / tr)
        # DEFLATED: the capture saved sum(x x^T), not the covariance, so the
        # top eigendirection is essentially the DC/mean component (63% of the
        # energy at block 0). Project it out as a proxy for centring, since
        # the concepts of interest live in the fluctuating part.
        ev, U = torch.linalg.eigh(C)
        u = U[:, -1:]
        Cd = C - u @ (u.T @ C @ u) @ u.T
        trd = torch.trace(Cd).item()
        ovd[b] = float((Vk[b] @ Cd @ Vk[b].T).trace() / trd) if trd > 0 else float("nan")
    rows.append(dict(lab=lab, grp=grp, ov=ov, ovr=ovr, ovd=ovd, dam=damage(ds)))

if not rows:
    print("\nNo gram files yet -- rerun once job 282022 has written them."); raise SystemExit

print(f"\n{'dataset':11}{'grp':>6}{'damage':>9}" +
      "".join(f"{'b'+str(b):>8}" for b in BLOCKS) +
      f"{'mean':>8}{'null':>8}{'ratio':>7}{'deflated':>10}")
for r in sorted(rows, key=lambda r: -r["dam"]):
    m  = np.mean([r["ov"][b]  for b in r["ov"]])
    mr = np.mean([r["ovr"][b] for b in r["ovr"]])
    md = np.nanmean([r["ovd"][b] for b in r["ovd"]])
    print(f"  {r['lab']:9}{r['grp']:>6}{r['dam']:9.4f}" +
          "".join(f"{r['ov'].get(b,float('nan')):8.3f}" for b in BLOCKS) +
          f"{m:8.3f}{mr:8.3f}{m/mr:7.2f}{md:10.3f}")

print("\nDoes overlap with the overwritten subspace predict damage?")
d = [r["dam"] for r in rows]
for b in BLOCKS:
    x = [r["ov"].get(b, np.nan) for r in rows]
    if np.isnan(x).any(): continue
    rho, p = spearmanr(x, d); print(f"  block {b:<3} rho={rho:+.3f}  p={p:.4f}")
mean_ov = [np.mean(list(r["ov"].values())) for r in rows]
rho, p = spearmanr(mean_ov, d)
print(f"  mean      rho={rho:+.3f}  p={p:.4f}   (n={len(rows)})")
rho, p = spearmanr([np.mean(list(r["ovr"].values())) for r in rows], d)
print(f"  NULL      rho={rho:+.3f}  p={p:.4f}   <- random subspace control")
rho, p = spearmanr([np.nanmean(list(r["ovd"].values())) for r in rows], d)
print(f"  DEFLATED  rho={rho:+.3f}  p={p:.4f}   <- DC direction projected out")
print("""
  Read: if 'mean' tracks damage but 'deflated' does not, the apparent signal was
  the shared DC component, not dataset-specific structure. If both track, the
  result is robust to the missing centring. 'ratio' vs the random-subspace null
  says whether dW's directions are special at all -- with activation effective
  dimension of only 2-7, some overlap is guaranteed for any subspace.""")
