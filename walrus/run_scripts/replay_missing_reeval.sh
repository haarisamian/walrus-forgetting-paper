#!/bin/bash
#SBATCH --partition=lovelace-mc,ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --job-name=replay_reeval
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_missing_reeval_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_missing_reeval_%A_%a.err
#SBATCH --array=0-5

# GN-correct cyl_sim re-evals for 6 arms that had SW normalization errors:
# 0=phase1_gn_lr5e5  1=gn_to_gn(0%replay)  2=10%  3=30%  4=40%  5=90%

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
# Base config: Phase-1 GN training run (confirmed GlobalRevNormalization)
CONFIG_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

IDX=$SLURM_ARRAY_TASK_ID

case $IDX in
  0)
    LABEL="phase1_gn_lr5e5_reeval"
    CKPT="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"
    ;;
  1)
    LABEL="gn_to_gn_reeval"
    CKPT="$EXPBASE/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/step_100"
    ;;
  2)
    LABEL="replay10pct_sim_reeval"
    CKPT=$(ls -d ${EXPBASE}/walrus_replay10pct_sim*/finetune/0/checkpoints/step_100 2>/dev/null | head -1)
    ;;
  3)
    LABEL="replay30pct_sim_reeval"
    CKPT=$(ls -d ${EXPBASE}/walrus_replay30pct_sim*/finetune/0/checkpoints/step_100 2>/dev/null | head -1)
    ;;
  4)
    LABEL="replay40pct_sim_reeval"
    CKPT=$(ls -d ${EXPBASE}/walrus_replay40pct_sim*/finetune/0/checkpoints/step_100 2>/dev/null | head -1)
    ;;
  5)
    LABEL="replay90pct_sim_reeval"
    CKPT=$(ls -d ${EXPBASE}/walrus_replay90pct_sim*/finetune/0/checkpoints/step_100 2>/dev/null | head -1)
    ;;
esac

FOLDER="$EXPBASE/forget_eval_results/${LABEL}/cyl_sim/finetune/0"

echo "=== $LABEL | ckpt: $CKPT ==="

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT}'" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=forget_cyl_sim_${LABEL}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
