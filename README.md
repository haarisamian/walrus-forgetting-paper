# Frozen in Time: How a Physics Foundation Model Forgets Under Sim-to-Real Adaptation

Code release for our NeurIPS 2026 workshop paper "Frozen in Time: How a Physics Foundation Model
Forgets Under Sim-to-Real Adaptation." Everything here reproduces the paper's experiments and figures:
Phase I fine-tuning of Walrus-1.3B on simulated cylinder flow, Phase II adaptation to real PIV, the
replay sweep, forgetting evaluation on ten held-out Well datasets, and the mechanistic analysis
(representational similarity + persistence) that predicts which datasets are forgotten.

This repository is a fork of [Polymathic AI's Walrus](https://github.com/PolymathicAI/walrus)
(MIT; original README preserved as `README_walrus_upstream.md`), based on upstream commit `73f2cf6`.

## Layout

| path | contents |
|---|---|
| `walrus/` | the model + trainer. Our changes vs upstream: per-dataset `sampling_weight` in `walrus/data/` (the replay mixture knob) and dynamics metrics in `walrus/trainer/psd_metrics.py` |
| `walrus/run_scripts/` | every Slurm script used for the paper, grouped by prefix: `finetune_cylinder_*` (Phase I), `gn_to_gn_*` / `sw_to_gn_*` / `pretrained_to_gn_*` (Phase II), `replay_*` / `cl_*` (replay sweep), `forget_eval_*` (held-out Well evaluation), `capture_activations_*` (CKA), `download_*` (data) |
| `scripts/collect_*.py` | aggregate evaluation outputs into the paper's tables |
| `analysis/scripts/` | the forgetting-mechanism analysis: `capture_activations.py`, `persistence_fullsplit.py`, `analyze_ft_overlap.py` |
| `analysis/plots/` | figure and table scripts; `make_figures.py` and `forgetting_figures.py` are the entry points |
| `tests/` | unit tests, including the replay-weighting test (`test_sampling_weights.py`) |

## Paths

All scripts resolve data and checkpoints under one root, `WALRUS_PAPER_ROOT`
(default: the Cambridge cluster path the paper was produced on). Set it to relocate:

```bash
export WALRUS_PAPER_ROOT=/your/root
```

Expected under the root: `walrus/experiments/` (training runs + `forget_eval_results/`),
`realpdebench/well/` (cylinder sim + PIV in Well format), `walrus_model/checkpoints/walrus.pt`,
`the_well/datasets/`. Python code imports the root from `analysis/_paths.py`; shell scripts read the
same variable. The `#SBATCH --output/--error` log paths are the one place the literal cluster path
remains, because Slurm directives cannot expand variables — edit them for your cluster.

## Data and pretrained model

- Cylinder simulation and real PIV: [RealPDEBench](https://github.com/AI4Science-WestlakeU/RealPDEBench) (CC BY-NC 4.0). Converted to Well format with `analysis/scripts/convert_realpde.py`; the train/valid/test split with `create_cylinder_real_cl_splits.py`.
- Held-out evaluation datasets: [The Well](https://huggingface.co/polymathic-ai) (CC-BY-4.0), fetched by `walrus/run_scripts/download_*.sh`.
- Pretrained checkpoint: [polymathic-ai/walrus](https://huggingface.co/polymathic-ai/walrus) (MIT).

## Environment

`environment.yml` (conda, Python 3.11) is the environment the paper was run with. ffmpeg is required
for rollout videos.

## Reproducing the headline analysis

1. Phase I / Phase II fine-tuning: `walrus/run_scripts/finetune_cylinder_swirles_lr5e5.sh`, then `gn_to_gn_real100pct.sh`.
2. Forgetting evaluation on the ten Well datasets: `forget_eval_pretrained.sh`, `forget_eval_phase1_dynamics.sh`, `forget_eval_gn_to_gn.sh`.
3. Predictors: `capture_activations_v3.sh` (activation covariances, 64 windows), then `analysis/scripts/persistence_fullsplit.py` and `analysis/scripts/analyze_ft_overlap.py` (CKA at the input to the final block + log–log regression).
4. Figures and tables: `python analysis/plots/make_figures.py`.

Compute for the reported experiments was ≈4,400 GPU-hours (L40S and A100); see the paper appendix.

## Citation

If you use this code or the forgetting analysis, please cite the paper:

```bibtex
@inproceedings{
mian2026frozen,
title={Frozen in Time: How a Physics Foundation Model Forgets Under Sim-to-Real Adaptation},
author={Haaris Mian and Miles Cranmer and Alessandro Favero and Payel Mukhopadhyay},
booktitle={Sim2Science: ML with Imperfect Scientific Models},
year={2026},
url={https://openreview.net/forum?id=9OGz7wjj5j}
}
```

Walrus itself is Polymathic AI's model; please also cite their paper (see `README_walrus_upstream.md`).
