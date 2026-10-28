#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --job-name=dump_vx_fields
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dump_vx_fields_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dump_vx_fields_%A_%a.err
#SBATCH --array=0-1

# Dump raw prediction npy files for GN (task 0) and SW (task 1) models.
# Uses epoch-40 checkpoints (closest saved to the epoch-30 videos).
# Saves:
#   {viz_folder}/realpdebench_cylinder_real_cl/full_trajectory_dumps/
#       ypred_*_{j}.npy  — shape (1, T, H, W, C), j=0,1,2 for batches 0-2
#       yref_*_{j}.npy   — same shape (GT)

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE=${WALRUS_PAPER_ROOT}/walrus/experiments

GN_EXP="walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
SW_EXP="phase2_sw_real/walrus_cl_v2_real100pct_sim0pct_fixed-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

case $SLURM_ARRAY_TASK_ID in
    0)
        LABEL="gn"
        CONFIG_DIR="$EXPBASE/$GN_EXP/finetune/0"
        CKPT="$EXPBASE/$GN_EXP/finetune/0/checkpoints/step_40"
        ;;
    1)
        LABEL="sw"
        CONFIG_DIR="$EXPBASE/$SW_EXP/finetune/0"
        CKPT="$EXPBASE/$SW_EXP/finetune/0/checkpoints/step_40"
        ;;
esac

VIZ_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/dump_vx_fields_${LABEL}/finetune/0"

echo "=== dump_vx_fields task ${SLURM_ARRAY_TASK_ID}: ${LABEL} ==="
echo "CKPT: $CKPT"
echo "VIZ_DIR: $VIZ_DIR"

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    "++folder_override=$VIZ_DIR" \
    "checkpoint.load_checkpoint_path='$CKPT'" \
    "checkpoint.coalesced_checkpoint_path=null" \
    "++checkpoint.prioritize_resume=False" \
    ++start_rollout_valid_output_at_t=-1 \
    ++trainer.dump_prediction_to_disk=True \
    ++trainer.num_detailed_logs=4 \
    ++trainer.video_validation=False \
    ++trainer.debug_mode=True \
    data_workers=4
