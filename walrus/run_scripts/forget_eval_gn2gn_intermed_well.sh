#!/bin/bash
#SBATCH --partition=lovelace-mc,ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=gn2gn_well_intermed
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/gn2gn_well_intermed_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/gn2gn_well_intermed_%A_%a.err
#SBATCH --array=0-15

# Well dataset forgetting eval for gn_to_gn intermediate checkpoints:
#   step_20, step_40, step_60, step_80
# (step_100 / best already have results from forget_eval_gn_to_gn.sh)
#
# Array task layout (4 datasets × 4 steps = 16 tasks):
#   task_idx = step_idx * 4 + ds_idx
#   step_idx  0→step_20  1→step_40  2→step_60  3→step_80
#   ds_idx    0→MHD_64   1→trl2d    2→turbcool 3→active_matter
#
# Uses SamplewiseRevNormalization (fair cross-norm comparison).

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

GN_TO_GN_DIR="$EXPBASE/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints"

STEPS=(20 40 60 80)
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

IDX=$SLURM_ARRAY_TASK_ID
STEP_IDX=$((IDX / 4))
DS_IDX=$((IDX % 4))

STEP="${STEPS[$STEP_IDX]}"
DS_NAME="${DS_NAMES[$DS_IDX]}"
CONFIG_DIR="${DS_CONFIG_DIRS[$DS_IDX]}"
WELL_PATH="${DS_WELL_PATHS[$DS_IDX]}"

CKPT_PATH="${GN_TO_GN_DIR}/step_${STEP}"
EVAL_NAME="forget_gn_to_gn_step${STEP}_${DS_NAME}"
FOLDER="$EXPBASE/forget_eval_results/gn_to_gn_step${STEP}/${DS_NAME}/finetune/0"

echo "=== gn2gn_well_intermed task ${IDX}: step_${STEP} on ${DS_NAME} ==="
echo "CKPT: $CKPT_PATH"
echo "CONFIG: $CONFIG_DIR"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Dataset-specific overrides (mirrors forget_eval_gn_to_gn.sh logic)
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
