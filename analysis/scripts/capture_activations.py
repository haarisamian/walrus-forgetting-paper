"""
Capture per-dataset activation Gram matrices, by INSTRUMENTING the standard
eval path rather than reimplementing it.

Rationale: the model's forward needs the trainer's revin normalisation and
metadata pipeline. Rebuilding that by hand is exactly how silent errors get in
(an earlier attempt died on checkpoint field alignment). So this wrapper
monkeypatches SpaceTimeSplitBlock.forward to record activations, then calls
walrus.train.main() unchanged -- identical data, normalisation and checkpoint
loading to forget_eval_new_well_pretrained.sh.

Block input x is (T, B, C, H, W, D) with the 1408-dim hidden axis at index 2.
We accumulate C_block = sum_tokens x x^T so the offline step can compute what
fraction of a dataset's activation energy lies in the subspace Phase 1
overwrote:  overlap = tr(V_k C V_k^T) / tr(C).

Records BOTH the second moment sum(x x^T) and the first moment sum(x) with the
token count, so the covariance can be formed properly:
    Cov = (sum(x x^T) - N mu mu^T) / N,   mu = sum(x)/N
An earlier version saved only the second moment, forcing a deflate-the-top-
eigenvector proxy for centring. The DC direction carries ~63% of the energy at
block 0, so that proxy was doing real work and had to be checked separately.

Also records field_to_index_map, so the 67 embedding slots can be named.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import atexit, os, sys
import numpy as np
import torch

sys.path.insert(0, f"{ROOT}/walrus")
import walrus.models.spatiotemporal_blocks.space_time_split as sts
import walrus.train as wtrain

PROBE = {int(x) for x in os.environ.get("ACT_BLOCKS", "0,10,20,30,39").split(",")}
OUT   = os.environ["ACT_OUT"]
LABEL = os.environ["ACT_LABEL"]
MAXB  = int(os.environ.get("ACT_MAXB", "6"))
MAXTOK = 4096

_idx, _grams, _n = {}, {}, {}
_sums, _tok = {}, {}          # first moment and token count -> proper centring
_f2i = {}

_orig_fwd = sts.SpaceTimeSplitBlock.forward
def _patched_fwd(self, x, bcs, return_att=False):
    i = _idx.setdefault(id(self), len(_idx))
    if i in PROBE and _n.get(i, 0) < MAXB:
        with torch.no_grad():
            C = x.shape[2]
            v = x.detach().float().permute(0, 1, 3, 4, 5, 2).reshape(-1, C)
            if v.shape[0] > MAXTOK:
                v = v[torch.randperm(v.shape[0], device=v.device)[:MAXTOK]]
            g = (v.T @ v).cpu()
            sm = v.sum(0).cpu()
            _grams[i] = _grams.get(i, torch.zeros_like(g)) + g
            _sums[i]  = _sums.get(i, torch.zeros_like(sm)) + sm
            _tok[i]   = _tok.get(i, 0) + v.shape[0]
            _n[i] = _n.get(i, 0) + 1
    return _orig_fwd(self, x, bcs, return_att=return_att)
sts.SpaceTimeSplitBlock.forward = _patched_fwd

_orig_load = wtrain.load_from_coalesced_checkpoint
def _patched_load(*a, **k):
    m = k.get("field_to_index_map")
    if m: _f2i.update(m)
    return _orig_load(*a, **k)
wtrain.load_from_coalesced_checkpoint = _patched_load

@atexit.register
def _dump():
    if not _grams:
        print("[act] NO ACTIVATIONS CAPTURED -- patch did not fire", flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"gram_{LABEL}.npz")
    np.savez(p,
             blocks=np.array(sorted(_grams)),
             ncalls=np.array([_n[i] for i in sorted(_grams)]),
             field_names=np.array(list(_f2i.keys())),
             field_idx=np.array(list(_f2i.values())),
             ntok=np.array([_tok[i] for i in sorted(_grams)]),
             **{f"block{i}": _grams[i].numpy() for i in _grams},
             **{f"sum{i}": _sums[i].numpy() for i in _sums})
    print(f"[act] wrote {p} blocks={sorted(_grams)} calls={_n} "
          f"fields={len(_f2i)}", flush=True)

wtrain.main()
