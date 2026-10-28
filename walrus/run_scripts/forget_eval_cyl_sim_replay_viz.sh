#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=cyl_sim_viz
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/cyl_sim_viz_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/cyl_sim_viz_%A_%a.err
#SBATCH --array=0-8

# Cyl_sim rollout VISUALIZATION for replay experiments using their native GlobalRevNorm.
#
# WHY SEPARATE FROM forget_eval_cyl_sim_replay.sh:
#   That script forces SamplewiseRevNorm to produce a normalized forgetting metric (step-1 VRMSE).
#   For replay experiments trained with GlobalRevNorm, the SW override breaks autoregressive
#   rollout in delta-prediction mode (normalization mismatch → error compounding → divergence).
#   This script uses no norm override so videos show the model's true behaviour.
#
# Does NOT override trainer.revin → uses GlobalRevNorm from each experiment's own config.
# PKLs written to: forget_eval_results/{label}/cyl_sim_viz/finetune/0/
# (separate from the metric PKLs in cyl_sim/ to avoid overwriting)

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

# Base config: phase1_gn has cylinder_numerical in its data config
CONFIG_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

CKPT_NAMES=(
    "replay10pct_sim"
    "replay20pct_sim"
    "replay30pct_sim"
    "replay40pct_sim"
    "replay50pct_sim"
    "replay60pct_sim"
    "replay70pct_sim"
    "replay80pct_sim"
    "replay90pct_sim"
)
CKPT_PATHS=(
    "$EXPBASE/walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay20pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay30pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay40pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay50pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay60pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay70pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay80pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
)

IDX=$SLURM_ARRAY_TASK_ID
CKPT_NAME="${CKPT_NAMES[$IDX]}"
CKPT_PATH="${CKPT_PATHS[$IDX]}"
FOLDER="$EXPBASE/forget_eval_results/${CKPT_NAME}/cyl_sim_viz/finetune/0"

echo "=== cyl_sim_viz task ${IDX}: ${CKPT_NAME} (GlobalRevNorm) ==="
echo "CKPT: $CKPT_PATH"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# NOTE: no ++trainer.revin override — uses GlobalRevNorm from extended_config.yaml
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
    "++name=cyl_sim_viz_${CKPT_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
