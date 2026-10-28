#!/bin/bash
#SBATCH --partition=lovelace-mc,ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=gn2gn_cyl_intermed
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/gn2gn_cyl_intermed_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/gn2gn_cyl_intermed_%A_%a.err
#SBATCH --array=0-3

# Cylinder sim (cylinder_numerical) forgetting eval for gn_to_gn intermediate checkpoints:
#   task 0 → step_20
#   task 1 → step_40
#   task 2 → step_60
#   task 3 → step_80
# (step_100 / last already evaluated in forget_eval_cyl_sim.sh)
#
# Uses GlobalRevNormalization (model was Phase 1 GN trained on cyl_sim).
# Config dir: phase1_gn (walrus_finetune_cylinder_lr5e5) has cylinder_numerical.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

# Phase 1 GN config has cylinder_numerical dataset configured
CONFIG_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

GN_TO_GN_DIR="$EXPBASE/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints"

STEPS=(20 40 60 80)
IDX=$SLURM_ARRAY_TASK_ID
STEP="${STEPS[$IDX]}"

CKPT_PATH="${GN_TO_GN_DIR}/step_${STEP}"
CKPT_NAME="gn_to_gn_step${STEP}"
FOLDER="$EXPBASE/forget_eval_results/${CKPT_NAME}/cyl_sim/finetune/0"

echo "=== gn2gn_cyl_intermed task ${IDX}: step_${STEP} (cyl_sim) ==="
echo "CKPT: $CKPT_PATH"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# gn_to_gn is a Phase 2 model whose Phase 1 was GN-trained on cyl_sim.
# Use GN (no override) to match the normalization used during Phase 1 cyl_sim training.

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT_PATH}'" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=forget_cyl_sim_${CKPT_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
