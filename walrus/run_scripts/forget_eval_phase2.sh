#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_p2
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_phase2_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_phase2_%A_%a.err
#SBATCH --array=0-43

# Forgetting eval for phase-2 models on 4 Well datasets (MHD_64, trl2d, turbcool, active_matter).
# Array layout: 11 checkpoints × 4 datasets = 44 tasks (ckpt_idx*4 + ds_idx)
# well_base_path differs: active_matter lives in the_well/datasets/ (not datasets/datasets/)
#
# Checkpoint index map:
#   0  phase1_gn_lr5e5          5  cl_v2_real0pct
#   1  sw_to_gn_5e5_ep100       6  cl_v2_real25pct
#   2  sw_to_gn_1e4_ep100       7  cl_v2_real50pct
#   3  sw_lr1e4_ep100           8  cl_v2_real75pct
#   4  cl_v2_stride4_ep50       9  cl_v2_real100pct
#                              10  gn_to_gn_lr1e4

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

CKPT_NAMES=(
    "phase1_gn_lr5e5"
    "sw_to_gn_5e5_ep100"
    "sw_to_gn_1e4_ep100"
    "sw_lr1e4_ep100"
    "cl_v2_stride4_ep50"
    "cl_v2_real0pct"
    "cl_v2_real25pct"
    "cl_v2_real50pct"
    "cl_v2_real75pct"
    "cl_v2_real100pct"
    "gn_to_gn_lr1e4"
)
CKPT_PATHS=(
    "$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"
    "$EXPBASE/walrus_sw_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"
    "$EXPBASE/walrus_sw_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_sw_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_cl_v2_real50pct_stride4-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_cl_v2_real0pct_sim100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"
    "$EXPBASE/walrus_cl_v2_real25pct_sim75pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_cl_v2_real50pct_sim50pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_cl_v2_real75pct_sim25pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_cl_v2_real100pct_sim0pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"
)

DS_NAMES=("MHD_64" "trl2d" "turbcool" "active_matter")
DS_CONFIG_DIRS=(
    "$EXPBASE/walrus_MHD64_eval/finetune/0"
    "$EXPBASE/walrus_trl2d_eval/finetune/0"
    "$EXPBASE/walrus_activ_pretrained/finetune/0"
    "$EXPBASE/walrus_activ_pretrained/finetune/0"
)
# active_matter lives in the_well/datasets/ ; others in the_well/datasets/datasets/
DS_WELL_PATHS=(
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets"
)

CKPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))  # 0..10
DS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

CKPT_NAME="${CKPT_NAMES[$CKPT_IDX]}"
CKPT_PATH="${CKPT_PATHS[$CKPT_IDX]}"
DS_NAME="${DS_NAMES[$DS_IDX]}"
CONFIG_DIR="${DS_CONFIG_DIRS[$DS_IDX]}"
WELL_PATH="${DS_WELL_PATHS[$DS_IDX]}"

EVAL_BASE="$EXPBASE/forget_eval_results"
EVAL_NAME="forget_${CKPT_NAME}_${DS_NAME}"
FOLDER="$EVAL_BASE/${CKPT_NAME}/${DS_NAME}/finetune/0"

# Dataset-specific args
if [ "$DS_IDX" -eq 0 ]; then
    # MHD_64: step_downsample_factor:0.5 to match walrus.pt pretraining
    mhd_extra_args=(
        "~data.module_parameters.well_dataset_info.MHD_64"
        "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
    )
elif [ "$DS_IDX" -eq 2 ]; then
    # turbcool: swap active_matter for turbulence_gravity_cooling with log transforms
    mhd_extra_args=(
        "~data.module_parameters.well_dataset_info.active_matter"
        "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}"
    )
else
    mhd_extra_args=()
fi

echo "=== forget_eval_phase2 task ${SLURM_ARRAY_TASK_ID}: ${CKPT_NAME} on ${DS_NAME} ==="
echo "CKPT: $CKPT_PATH"
echo "CONFIG: $CONFIG_DIR"

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
    "++name=${EVAL_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_PATH}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "${mhd_extra_args[@]}"
