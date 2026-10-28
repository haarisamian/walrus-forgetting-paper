"""
tab_appendix_well_dyn_nrmse.py
-------------------------------
Appendix table: mean per-field NRMSE for all Well benchmark datasets across
the Phase I forgetting-dynamics checkpoints:
  pretrained | step 5 | step 10 | step 15 | step 20

Uses the same metric as forgetting_figures.py:
  valid_loss_dict_*.pkl  →  valid_{ds_pkl_name}/{field}_NRMSE_T=all_mean
  mean across fields, excluding full_NRMSE entries.

Outputs:
  tab_appendix_well_dyn_nrmse.tex
  tab_appendix_well_dyn_nrmse.csv
"""

import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1]))
from analysis._paths import ROOT
import glob
import io
import pickle
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT    = Path(f"{ROOT}/realpdebench/analysis/plots")
FEBASE = Path(f"{ROOT}/walrus/experiments/forget_eval_results")

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
CKPTS = [
    ("pretrained", "pretrained"),
    ("step 5",     "phase1_dyn_step5"),
    ("step 10",    "phase1_dyn_step10"),
    ("step 15",    "phase1_dyn_step15"),
    ("step 20",    "phase1_dyn_step20"),
]

# ---------------------------------------------------------------------------
# Dataset registry:
#   folder_name  — subdir in forget_eval_results/{ckpt}/
#   pkl_ds_name  — prefix in valid_loss_dict keys: valid_{pkl_ds_name}/
#   tex_display  — LaTeX display name
# ---------------------------------------------------------------------------
DATASETS = [
    ("acoustic_scattering_discontinuous", "acoustic_scattering_discontinuous",
     r"acoustic\_scattering\_disc."),
    ("acoustic_scattering_inclusions",    "acoustic_scattering_inclusions",
     r"acoustic\_scattering\_incl."),
    ("acoustic_scattering_maze",          "acoustic_scattering_maze",
     r"acoustic\_scattering\_maze"),
    ("active_matter",                     "active_matter",
     r"active\_matter"),
    ("helmholtz_staircase",               "helmholtz_staircase",
     r"helmholtz\_staircase"),
    ("MHD_64",                            "MHD_64",
     r"MHD\_64"),
    ("planetswe",                         "planetswe",
     r"planetswe"),
    ("shear_flow",                        "shear_flow",
     r"shear\_flow"),
    ("trl2d",                             "turbulent_radiative_layer_2D",
     r"turbulent\_radiative\_layer\_2D"),
    ("turbcool",                          "turbulence_gravity_cooling",
     r"turbulence\_gravity\_cooling"),
]

# ---------------------------------------------------------------------------
# CPU unpickler
# ---------------------------------------------------------------------------
class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)

def load_pkl(path):
    with open(path, "rb") as f:
        return _CPUUnpickler(f).load()

def find_valid_pkl(folder_path):
    """Find valid_loss_dict_*.pkl (not rollout_valid_*)."""
    pattern = str(folder_path / "finetune/0/viz/loss_dicts/valid_loss_dict_*.pkl")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

def mean_nrmse(d, pkl_ds_name):
    """Mean NRMSE across fields from valid_loss_dict."""
    prefix = f"valid_{pkl_ds_name}/"
    vals = [
        float(v) for k, v in d.items()
        if k.startswith(prefix)
        and "NRMSE_T=all_mean" in k
        and not k.endswith("full_NRMSE_T=all_mean")
    ]
    return float(np.mean(vals)) if vals else None

# ---------------------------------------------------------------------------
# Collect results
# ---------------------------------------------------------------------------
results = {}   # (folder_name, ckpt_label) → float | None

for folder_name, pkl_ds_name, _ in DATASETS:
    for ckpt_label, ckpt_tag in CKPTS:
        folder_path = FEBASE / ckpt_tag / folder_name
        pkl = find_valid_pkl(folder_path)
        if pkl is None:
            print(f"  MISSING: {ckpt_tag}/{folder_name}")
            results[(folder_name, ckpt_label)] = None
            continue
        try:
            d = load_pkl(pkl)
            v = mean_nrmse(d, pkl_ds_name)
            results[(folder_name, ckpt_label)] = v
            tag = f"{v:.4f}" if v is not None else "NO KEY"
            print(f"  {ckpt_tag}/{folder_name}: {tag}")
        except Exception as e:
            print(f"  ERROR {ckpt_tag}/{folder_name}: {e}")
            results[(folder_name, ckpt_label)] = None

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
col_labels = [cl for cl, _ in CKPTS]

def fmt_val(v):
    if v is None:
        return "—"
    return f"{v:.4f}"

def fmt_tex(v, is_min):
    s = fmt_val(v)
    if s == "—":
        return r"\textemdash"
    if is_min:
        return r"\textbf{" + s + "}"
    return s

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
csv_lines = ["dataset," + ",".join(col_labels)]
for folder_name, _, _ in DATASETS:
    row = [folder_name]
    for cl in col_labels:
        v = results.get((folder_name, cl))
        row.append(fmt_val(v))
    csv_lines.append(",".join(row))
csv_path = OUT / "tab_appendix_well_dyn_nrmse.csv"
csv_path.write_text("\n".join(csv_lines))
print(f"\nCSV written.")
print("\n" + "\n".join(csv_lines))

# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
header_cols = " & ".join([r"\textbf{" + cl + "}" for cl in col_labels])
tex = (
    r"\begin{table}[h]" + "\n"
    r"\centering" + "\n"
    r"\caption{Well benchmark mean per-field NRMSE at each Phase I forgetting-dynamics"
    r" checkpoint. Metric: mean of per-field \texttt{NRMSE\_T=all\_mean} from the"
    r" one-step validation pass, matching the metric used in the main forgetting figure."
    r" Bold = lowest (best) per row."
    r" \emph{Pretrained}: Walrus-1.3B before any cylinder finetuning."
    r" \emph{Step $k$}: after $k$ epochs of Phase I cylinder finetuning (200 gradient steps/epoch).}"
    + "\n"
    r"\label{tab:appendix_well_dyn_nrmse}" + "\n"
    r"\small" + "\n"
    r"\setlength{\tabcolsep}{5pt}" + "\n"
    r"\begin{tabular}{l" + "r" * len(col_labels) + "}\n"
    r"\toprule" + "\n"
    r"\textbf{Dataset} & " + header_cols + r" \\" + "\n"
    r"\midrule" + "\n"
)

for folder_name, _, tex_display in DATASETS:
    vals = [results.get((folder_name, cl)) for cl in col_labels]
    valid = [v for v in vals if v is not None]
    minv  = min(valid) if valid else None
    cells = []
    for v in vals:
        is_min = (v is not None and minv is not None
                  and abs(v - minv) < 1e-8 * max(abs(minv), 1e-9))
        cells.append(fmt_tex(v, is_min))
    tex += f"  {tex_display} & " + " & ".join(cells) + r" \\" + "\n"

tex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"

tex_path = OUT / "tab_appendix_well_dyn_nrmse.tex"
tex_path.write_text(tex)
print(f"\nTeX written: {tex_path}")
print("\nDone.")
