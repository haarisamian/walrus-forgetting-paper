#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_pretrained
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_pretrained_well_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_pretrained_well_%A_%a.err
#SBATCH --array=0-3

# Pretrained walrus.pt baseline eval on 4 Well forgetting datasets.
# Array layout: 0=MHD_64, 1=trl2d, 2=turbcool, 3=active_matter
# well_base_path differs: active_matter lives in the_well/datasets/ (not datasets/datasets/)
# Required to anchor the forgetting curves — without this, finetuned checkpoint
# VRMSEs have no reference point.
#
# MHD_64 uses step_downsample_factor:0.5 (walrus.pt was trained with this).
# learnable_rope=False: correct for walrus.pt which has no learnable RoPE.
# SamplewiseRevNormalization matches the finetuned evals in phase2/gn_to_gn.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

DS_NAMES=("MHD_64" "trl2d" "turbcool" "active_matter")
DS_CONFIG_DIRS=(
    "$EXPBASE/walrus_MHD64_eval/finetune/0"
    "$EXPBASE/walrus_trl2d_eval/finetune/0"
    "$EXPBASE/walrus_activ_pretrained/finetune/0"
    "$EXPBASE/walrus_activ_pretrained/finetune/0"
)
DS_WELL_PATHS=(
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets"
)

DS_IDX=$SLURM_ARRAY_TASK_ID
DS_NAME="${DS_NAMES[$DS_IDX]}"
CONFIG_DIR="${DS_CONFIG_DIRS[$DS_IDX]}"
WELL_PATH="${DS_WELL_PATHS[$DS_IDX]}"
EVAL_NAME="forget_pretrained_walrus_${DS_NAME}"
FOLDER="$EXPBASE/forget_eval_results/pretrained/${DS_NAME}/finetune/0"

if [ "$DS_IDX" -eq 0 ]; then
    mhd_extra_args=(
        "~data.module_parameters.well_dataset_info.MHD_64"
        "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
    )
elif [ "$DS_IDX" -eq 2 ]; then
    mhd_extra_args=(
        "~data.module_parameters.well_dataset_info.active_matter"
        "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}"
    )
else
    mhd_extra_args=()
fi

echo "=== forget_eval_pretrained_well task ${SLURM_ARRAY_TASK_ID}: walrus.pt on ${DS_NAME} ==="
echo "CONFIG: $CONFIG_DIR"

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=${WALRUS_PAPER_ROOT}/walrus_model/checkpoints/walrus.pt \
    checkpoint.load_checkpoint_path=null \
    ++finetuning_mods.learnable_rope=False \
    ++finetuning_mods.rope_per_axis=False \
    "++data.well_base_path=${WELL_PATH}" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=${EVAL_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    "${mhd_extra_args[@]}"
