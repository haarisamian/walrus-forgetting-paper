#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_replay
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_replay_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_replay_%A_%a.err
#SBATCH --array=0-35

# Forgetting eval for replay experiments on 4 Well datasets.
# Array layout: 9 replay checkpoints × 4 datasets = 36 tasks (ckpt_idx*4 + ds_idx)
# Replay checkpoints: 10%..90% simulation data fraction mixed during cylinder finetuning.
# All have learnable_rope=True (initialized from phase1_gn).
#
# Checkpoint index:
#   0  replay10pct_sim    5  replay60pct_sim
#   1  replay20pct_sim    6  replay70pct_sim
#   2  replay30pct_sim    7  replay80pct_sim
#   3  replay40pct_sim    8  replay90pct_sim
#   4  replay50pct_sim
#
# Dataset index (ds_idx = task_id % 4):
#   0  MHD_64    1  trl2d    2  turbcool    3  active_matter

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

CKPT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

CKPT_NAME="${CKPT_NAMES[$CKPT_IDX]}"
CKPT_PATH="${CKPT_PATHS[$CKPT_IDX]}"
DS_NAME="${DS_NAMES[$DS_IDX]}"
CONFIG_DIR="${DS_CONFIG_DIRS[$DS_IDX]}"
WELL_PATH="${DS_WELL_PATHS[$DS_IDX]}"

EVAL_BASE="$EXPBASE/forget_eval_results"
EVAL_NAME="forget_${CKPT_NAME}_${DS_NAME}"
FOLDER="$EVAL_BASE/${CKPT_NAME}/${DS_NAME}/finetune/0"

if [ "$DS_IDX" -eq 0 ]; then
    ds_extra_args=(
        "~data.module_parameters.well_dataset_info.MHD_64"
        "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
    )
elif [ "$DS_IDX" -eq 2 ]; then
    ds_extra_args=(
        "~data.module_parameters.well_dataset_info.active_matter"
        "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}"
    )
else
    ds_extra_args=()
fi

echo "=== forget_eval_replay task ${SLURM_ARRAY_TASK_ID}: ${CKPT_NAME} on ${DS_NAME} ==="
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
    "${ds_extra_args[@]}"
