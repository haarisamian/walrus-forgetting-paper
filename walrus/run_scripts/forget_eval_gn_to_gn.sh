#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_gn2gn
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_gn_to_gn_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_gn_to_gn_%A_%a.err
#SBATCH --array=0-3

# Forgetting eval for gn_to_gn checkpoint on 4 Well datasets.
# Array layout: 0=MHD_64, 1=trl2d, 2=turbcool, 3=active_matter
# well_base_path differs: active_matter lives in the_well/datasets/ (not datasets/datasets/)
# Uses SamplewiseRevNormalization for fair comparison across norm strategies.
# Submit after job 213475 (gn_to_gn ep100) completes so best/ checkpoint exists.

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

# Use best if it exists (ep100 complete), otherwise fall back to last (ep90)
if [ -d "$GN_TO_GN_DIR/best" ]; then
    CKPT_PATH="$GN_TO_GN_DIR/best"
    CKPT_LABEL="best"
else
    CKPT_PATH="$GN_TO_GN_DIR/last"
    CKPT_LABEL="last_ep90"
fi

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
EVAL_NAME="forget_gn_to_gn_${CKPT_LABEL}_${DS_NAME}"
FOLDER="$EXPBASE/forget_eval_results/gn_to_gn_${CKPT_LABEL}/${DS_NAME}/finetune/0"

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

echo "=== forget_eval_gn_to_gn task ${SLURM_ARRAY_TASK_ID}: ${CKPT_LABEL} on ${DS_NAME} ==="
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
