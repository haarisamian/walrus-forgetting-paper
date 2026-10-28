#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_gn1e4
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_gn_to_gn_lr1e4_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_gn_to_gn_lr1e4_%A_%a.err
#SBATCH --array=0-3

# Forgetting eval for gn_to_gn_lr1e4 (epoch-100 last checkpoint) on 4 Well datasets.
# Previous evals (Jul 31) used an epoch-52 checkpoint — training completed Aug 4 09:27.
# Array layout: 0=MHD_64, 1=trl2d, 2=turbcool, 3=active_matter
# PKL written to: forget_eval_results/gn_to_gn_lr1e4/{ds_name}/finetune/0/
# New epoch-99 PKLs will supersede the stale epoch-52 ones in collect_forgetting_results.py.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

GN1E4_DIR="$EXPBASE/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0"
CKPT_PATH="$GN1E4_DIR/checkpoints/last"

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
EVAL_NAME="forget_gn_to_gn_lr1e4_${DS_NAME}"
FOLDER="$EXPBASE/forget_eval_results/gn_to_gn_lr1e4/${DS_NAME}/finetune/0"

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

echo "=== forget_eval_gn_to_gn_lr1e4 task ${SLURM_ARRAY_TASK_ID}: on ${DS_NAME} ==="
echo "CKPT: $CKPT_PATH"
echo "CONFIG: $CONFIG_DIR"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
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
    "${ds_extra_args[@]}"
