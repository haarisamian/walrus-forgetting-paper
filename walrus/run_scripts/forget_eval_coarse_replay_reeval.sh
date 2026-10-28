#!/bin/bash
#SBATCH --partition=lovelace-mc,ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --job-name=replay_cyl_reeval
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_cyl_reeval_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_cyl_reeval_%A_%a.err
#SBATCH --array=0-4

# Re-eval cyl-sim for coarse replay arms that were evaluated before training finished.
# Bad arms: 20%, 50%, 60%, 70%, 80%
# Writes to new subfolder reeval/ to avoid overwriting the old (stale) pkls.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
CONFIG_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

PCTS=(20 50 60 70 80)
IDX=$SLURM_ARRAY_TASK_ID
PCT=${PCTS[$IDX]}

CKPT_DIR=$(ls -d ${EXPBASE}/walrus_replay${PCT}pct_sim*/finetune/0/checkpoints 2>/dev/null | head -1)
# Prefer step_100 explicitly to guarantee we have the fully-trained checkpoint
if [ -d "${CKPT_DIR}/step_100" ]; then
    CKPT_PATH="${CKPT_DIR}/step_100"
else
    CKPT_PATH="${CKPT_DIR}/last"
fi

CKPT_NAME="replay${PCT}pct_sim"
# Write to separate reeval dir so old stale pkl is not overwritten until we've confirmed
FOLDER="$EXPBASE/forget_eval_results/${CKPT_NAME}_reeval/cyl_sim/finetune/0"

echo "=== coarse replay reeval: ${PCT}% | ckpt: $CKPT_PATH ==="

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
    "++name=forget_cyl_sim_${CKPT_NAME}_reeval" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
