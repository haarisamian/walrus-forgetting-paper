#!/bin/bash
#SBATCH --partition=lovelace-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=1:00:00
#SBATCH --job-name=phase1_piv_eval
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/phase1_piv_eval_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/phase1_piv_eval_%j.err

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
TORCHRUN=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/torchrun

# Base config: copy of gn_to_gn extended_config in bracket-free path (Hydra can't handle [...] in --config-path)
GN_CFG_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/phase1_piv_eval_cfg"

# Phase-I checkpoint (epoch 51, best val loss on cyl_sim)
CKPT="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"

# Output folder — use folder_override to bypass validation_mode "no previous runs" check
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
FOLDER="$EXPBASE/forget_eval_results/phase1_piv_eval/piv_real/finetune/0"

echo "=== phase1_piv_eval | ckpt: $CKPT ==="

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$GN_CFG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT}'" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=phase1_piv_eval" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
