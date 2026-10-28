#!/bin/bash
#SBATCH --partition=lovelace-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200gb
#SBATCH --time=2:00:00
#SBATCH --job-name=gn_eval_orig_ft
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/gn_eval_orig_ft_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/gn_eval_orig_ft_%A_%a.err
#SBATCH --array=0-5

# Array: 0=lr1e4_gn  1=lr1e4_sw  2=lr5e5_gn  3=lr5e5_sw  4=lr2e5_gn  5=lr2e5_sw

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXP_BASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

# CLv2 50% symlink as config base — already has cylinder_real_cl + correct eval settings
V2_DIR="$EXP_BASE/walrus_cl_v2_50pct_symlink/finetune/0"

CKPT_DIRS=(
    "walrus_finetune_cylinder_lr1e4-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001"
    "walrus_finetune_cylinder_lr1e4_samplewise-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001"
    "walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
    "walrus_finetune_cylinder_lr5e5_samplewise-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
    "walrus_finetune_cylinder_lr2e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-2e-05"
    "walrus_finetune_cylinder_lr2e5_samplewise-cylin-delta-Isotr[Space-Adapt-]-AdamW-2e-05"
)
NAMES=(
    "ft_lr1e4_gn_eval"
    "ft_lr1e4_sw_gninf_eval"
    "ft_lr5e5_gn_eval"
    "ft_lr5e5_sw_gninf_eval"
    "ft_lr2e5_gn_eval"
    "ft_lr2e5_sw_gninf_eval"
)

CKPT_DIR="$EXP_BASE/${CKPT_DIRS[$SLURM_ARRAY_TASK_ID]}/finetune/0/checkpoints/best"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
FOLDER="$EXP_BASE/walrus_${NAME}/finetune/0"

echo "=== gn_eval_orig_ft task ${SLURM_ARRAY_TASK_ID}: ${NAME} ==="
echo "CKPT: $CKPT_DIR"

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$V2_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT_DIR}'" \
    ++logger.wandb_project_name="walrus_cylinder_cl_eval" \
    "++name=${NAME}" \
    "++folder_override=${FOLDER}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.RMSGlobalRevNormalization" \
    "++trainer.debug_mode=True"
