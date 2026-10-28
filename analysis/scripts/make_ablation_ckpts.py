"""
Build surgical ablation checkpoints to isolate WHICH Phase-1 weight changes
cause forgetting.

Each arm starts from the Phase-1 checkpoint and RESTORES a chosen subset of
parameters to their pretrained values. Output is written as a standard
distributed checkpoint directory so the normal train.py + validation_mode
pipeline loads it via checkpoint.load_checkpoint_path -- no custom loading.

Arms
  restore_knorm     blocks.*.time_mixing.knorm.*     (the extreme outlier:
                                                      mean rel change 0.22)
  restore_timemix   blocks.*.time_mixing.*           (whole temporal operator)
  restore_spacemix  blocks.*.space_mixing.*          (CONTROL: comparable size,
                                                      different mechanism)

Prediction if the temporal-operator hypothesis is right: restore_knorm and
restore_timemix recover wave/acoustic datasets far more than fluid ones, while
restore_spacemix does not show that asymmetry.
"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import re, shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

EXP  = Path(f"{ROOT}/walrus/experiments")
P1   = EXP/("walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]"
            "-AdamW-5e-05/finetune/0/checkpoints/step_51")
PRE  = f"{ROOT}/walrus_model/checkpoints/walrus.pt"
OUT  = EXP/"ablation_ckpts"

ARMS = {
    "restore_knorm":    lambda k: re.match(r"blocks\.\d+\.time_mixing\.knorm\.", k),
    "restore_timemix":  lambda k: re.match(r"blocks\.\d+\.time_mixing\.", k),
    "restore_spacemix": lambda k: re.match(r"blocks\.\d+\.space_mixing\.", k),
}

print("loading pretrained ...")
pre = torch.load(PRE, map_location="cpu", weights_only=False, mmap=True)["app"]["model"]
md  = FileSystemReader(str(P1)).read_metadata().state_dict_metadata
mkeys = [k for k in md if k.startswith("app.model.") and k[len("app.model."):] in pre]
print(f"loading phase-1 ({len(mkeys)} model tensors) ...")
p1 = {k: torch.empty(md[k].size, dtype=torch.float32) for k in mkeys}
dcp.load_state_dict(state_dict=p1, storage_reader=FileSystemReader(str(P1)),
                    planner=DefaultLoadPlanner(allow_partial_load=True), no_dist=True)

OUT.mkdir(exist_ok=True)
for arm, match in ARMS.items():
    sd, n, npar = {}, 0, 0
    for fk, t in p1.items():
        k = fk[len("app.model."):]
        if match(k) and pre[k].shape == t.shape:
            sd[fk] = pre[k].float().clone(); n += 1; npar += t.numel()
        else:
            sd[fk] = t.clone()
    d = OUT/arm
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    dcp.save_state_dict(state_dict=sd, storage_writer=FileSystemWriter(str(d)), no_dist=True)
    torch.save({"epoch": 51, "val_loss": 0.0, "best_val_loss": 0.0}, d/"metadata.pt")
    # verify the surgery actually landed
    chk = {k: torch.empty_like(v) for k, v in sd.items()}
    dcp.load_state_dict(state_dict=chk, storage_reader=FileSystemReader(str(d)),
                        planner=DefaultLoadPlanner(allow_partial_load=True), no_dist=True)
    restored = sum(1 for fk in sd
                   if match(fk[len("app.model."):])
                   and torch.equal(chk[fk], pre[fk[len("app.model."):]].float()))
    moved = sum(1 for fk in sd
                if not match(fk[len("app.model."):])
                and not torch.equal(chk[fk], p1[fk]))
    print(f"  {arm:18} restored {n:4d} tensors / {npar/1e6:6.1f}M params  "
          f"| verify: {restored}/{n} match pretrained, {moved} unintended changes")
print(f"\nwritten to {OUT}")
