#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_new_well_dyn
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_new_well_phase1_dyn_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_new_well_phase1_dyn_%A_%a.err
#SBATCH --array=0-23

# Forgetting eval for phase1_forgetting_dynamics intermediate checkpoints on 6 new Well datasets.
# Measures how quickly Well pretraining is lost during early cyl_sim finetuning.
#
# Layout: 4 checkpoints × 6 datasets = 24 tasks (ckpt_idx*6 + ds_idx)
#
# Checkpoint index (ckpt_idx = task_id / 6):
#   0  step_5   (epoch 5)
#   1  step_10  (epoch 10)
#   2  step_15  (epoch 15)
#   3  step_20  (epoch 20)
#
# Dataset index (ds_idx = task_id % 6):
#   0  acoustic_scattering_discontinuous   (2D 256×256, 102t)
#   1  acoustic_scattering_inclusions      (2D 256×256, 102t)
#   2  acoustic_scattering_maze            (2D 256×256, 202t)
#   3  planetswe                           (2D 256×512, 1008t)
#   4  helmholtz_staircase                 (2D 1024×256, 50t)
#   5  shear_flow                          (2D 256×512, 200t)
#
# PKL written to: forget_eval_results/phase1_dyn_step{N}/{ds_name}/finetune/0/

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
WELL_BASE="${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
DYN_BASE="$EXPBASE/walrus_phase1_forgetting_dynamics-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints"
CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"

CKPT_STEPS=(5 10 15 20)
CKPT_NAMES=("phase1_dyn_step5" "phase1_dyn_step10" "phase1_dyn_step15" "phase1_dyn_step20")

DS_NAMES=(
    "acoustic_scattering_discontinuous"
    "acoustic_scattering_inclusions"
    "acoustic_scattering_maze"
    "planetswe"
    "helmholtz_staircase"
    "shear_flow"
)
STEP_DOWNSAMPLE=(1 1 1 0.5 1 1)

CKPT_IDX=$((SLURM_ARRAY_TASK_ID / 6))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 6))

STEP="${CKPT_STEPS[$CKPT_IDX]}"
CKPT_NAME="${CKPT_NAMES[$CKPT_IDX]}"
CKPT_PATH="$DYN_BASE/step_${STEP}"
DS_NAME="${DS_NAMES[$DS_IDX]}"
DOWNSAMPLE="${STEP_DOWNSAMPLE[$DS_IDX]}"
FOLDER="$EXPBASE/forget_eval_results/${CKPT_NAME}/${DS_NAME}/finetune/0"

# acoustic_scattering datasets: density is a static background field zeroed during pretraining.
# Must match pretraining: field_transforms={density:torch.zeros_like}.
# All other new well datasets: no field transforms.
if [ "$DS_IDX" -le 2 ]; then
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:${DOWNSAMPLE},field_transforms:{density:torch.zeros_like}}"
    echo "FIELD_TRANSFORM: density→zeros_like (acoustic_scattering background field)"
else
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:${DOWNSAMPLE}}"
fi

echo "=== forget_new_well_phase1_dyn task ${SLURM_ARRAY_TASK_ID}: ${CKPT_NAME} on ${DS_NAME} ==="
echo "CKPT: $CKPT_PATH"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

if [ ! -d "$WELL_BASE/$DS_NAME/data/valid" ]; then
    echo "ERROR: dataset not found: $WELL_BASE/$DS_NAME"
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
    "++name=forget_phase1_dyn_step${STEP}_${DS_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_BASE}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "~data.module_parameters.well_dataset_info.active_matter" \
    "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
