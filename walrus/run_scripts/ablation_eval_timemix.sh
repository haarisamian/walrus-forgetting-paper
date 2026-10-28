#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=abl_timemix
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/abl_timemix_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/abl_timemix_%A_%a.err
#SBATCH --array=0-29

# Which Phase-1 weight changes actually cause forgetting?
#
# Each arm = the Phase-1 checkpoint with one parameter group RESTORED to its
# pretrained values (built by realpdebench/scripts/make_ablation_ckpts.py):
#   restore_knorm     blocks.*.time_mixing.knorm.*   80 tensors, 0.04M params
#   restore_timemix   blocks.*.time_mixing.*        400 tensors, 317.5M params
#   restore_spacemix  blocks.*.space_mixing.*       400 tensors, 793.5M params  [CONTROL]
#
# HYPOTHESIS: Phase 1 overwhelmingly rewrote the temporal operator
# (time_mixing.knorm.bias, mean relative change 0.22 vs 0.08 for the next
# parameter class). Wave/acoustic physics depends on that temporal structure and
# takes 3.9-7.4x more absolute damage than fluid datasets. If restoring
# time_mixing recovers wave datasets far more than fluid ones -- and restoring
# the larger space_mixing does not -- the mechanism is established.
#
# Layout: 3 arms x 10 datasets = 30 tasks (arm_idx*10 + ds_idx)
# PKL -> forget_eval_results/{arm}/{ds}/finetune/0/

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
WELL_DD="${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
WELL_D="${WALRUS_PAPER_ROOT}/the_well/datasets"

ARMS=("restore_knorm" "restore_timemix" "restore_spacemix")
DS_NAMES=("acoustic_scattering_discontinuous" "acoustic_scattering_inclusions" \
          "acoustic_scattering_maze" "helmholtz_staircase" "planetswe" \
          "shear_flow" "active_matter" "MHD_64" "trl2d" "turbcool")

ARM_IDX=$((SLURM_ARRAY_TASK_ID / 10))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 10))
ARM="${ARMS[$ARM_IDX]}"
DS_NAME="${DS_NAMES[$DS_IDX]}"
CKPT_PATH="$EXPBASE/ablation_ckpts/${ARM}"

# Per-dataset config dir / well path / dataset entry. These differ per dataset
# and getting them wrong silently produces wrong numbers.
CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
WELL_PATH="$WELL_DD"
ds_extra_args=()
case "$DS_NAME" in
  acoustic_scattering_*)
    ds_extra_args=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.${DS_NAME}={include_filters:[],exclude_filters:[],step_downsample_factor:1,field_transforms:{density:torch.zeros_like}}") ;;
  helmholtz_staircase|shear_flow)
    ds_extra_args=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.${DS_NAME}={include_filters:[],exclude_filters:[],step_downsample_factor:1}") ;;
  planetswe)
    ds_extra_args=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.planetswe={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}") ;;
  active_matter)
    WELL_PATH="$WELL_D" ;;                       # active_matter lives one level up
  MHD_64)
    CONFIG_DIR="$EXPBASE/walrus_MHD64_eval/finetune/0"
    ds_extra_args=("~data.module_parameters.well_dataset_info.MHD_64"
      "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}") ;;
  trl2d)
    CONFIG_DIR="$EXPBASE/walrus_trl2d_eval/finetune/0" ;;
  turbcool)
    ds_extra_args=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}") ;;
esac

FOLDER="$EXPBASE/forget_eval_results/${ARM}/${DS_NAME}/finetune/0"

echo "=== ablation task ${SLURM_ARRAY_TASK_ID}: ${ARM} on ${DS_NAME} ==="
echo "CKPT:   $CKPT_PATH"
echo "CONFIG: $CONFIG_DIR"
echo "WELL:   $WELL_PATH"

if [ ! -d "$CKPT_PATH" ]; then echo "ERROR: checkpoint missing: $CKPT_PATH"; exit 1; fi
if [ ! -f "$CKPT_PATH/metadata.pt" ]; then echo "ERROR: metadata.pt missing"; exit 1; fi
if [ ! -d "$CONFIG_DIR" ]; then echo "ERROR: config dir missing: $CONFIG_DIR"; exit 1; fi

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
    "++name=abl_${ARM}_${DS_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_PATH}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "${ds_extra_args[@]}"
