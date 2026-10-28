#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=act_v3
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/act_v3_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/act_v3_%A_%a.err
#SBATCH --array=0-10

# v3: v2 used ACT_MAXB=6 with batch_size=1, i.e. SIX trajectory windows per
# dataset. The 4096 tokens drawn per window are correlated within that window,
# so the effective sample was six draws, not 24k. This raises it to 64 windows
# and writes to a separate directory so v2 is not clobbered.
# Pretrained arm only -- that is the frame the reported CKA is measured in.
# Task 10 is the cylinder simulation, the reference the CKA is taken against.

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
SIM_CFG="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

DS_NAMES=("acoustic_scattering_discontinuous" "acoustic_scattering_inclusions" \
          "acoustic_scattering_maze" "helmholtz_staircase" "planetswe" \
          "shear_flow" "active_matter" "MHD_64" "trl2d" "turbcool" "cylinder_numerical")
DS_NAME="${DS_NAMES[$SLURM_ARRAY_TASK_ID]}"

CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
WELL_PATH="$WELL_DD"
extra=()
case "$DS_NAME" in
  acoustic_scattering_*)
    extra=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.${DS_NAME}={include_filters:[],exclude_filters:[],step_downsample_factor:1,field_transforms:{density:torch.zeros_like}}") ;;
  helmholtz_staircase|shear_flow)
    extra=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.${DS_NAME}={include_filters:[],exclude_filters:[],step_downsample_factor:1}") ;;
  planetswe)
    extra=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.planetswe={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}") ;;
  active_matter) WELL_PATH="$WELL_D" ;;
  MHD_64)
    CONFIG_DIR="$EXPBASE/walrus_MHD64_eval/finetune/0"
    extra=("~data.module_parameters.well_dataset_info.MHD_64"
      "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}") ;;
  trl2d) CONFIG_DIR="$EXPBASE/walrus_trl2d_eval/finetune/0" ;;
  turbcool)
    extra=("~data.module_parameters.well_dataset_info.active_matter"
      "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}") ;;
  cylinder_numerical)
    CONFIG_DIR="$SIM_CFG"
    WELL_PATH="${WALRUS_PAPER_ROOT}/realpdebench/well" ;;
esac

export ACT_OUT="$EXPBASE/activation_grams_v3/pretrained"
export ACT_LABEL="$DS_NAME"
export ACT_BLOCKS="0,5,10,15,20,25,30,39"
export ACT_MAXB=64
FOLDER="$EXPBASE/act_capture_runs_v3/pretrained/${DS_NAME}/finetune/0"

echo "=== act_v3 task ${SLURM_ARRAY_TASK_ID}: ${DS_NAME} (MAXB=${ACT_MAXB}) ==="

srun $PYTHON ${WALRUS_PAPER_ROOT}/realpdebench/scripts/capture_activations.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=act_v3_${DS_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_PATH}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "checkpoint.load_checkpoint_path=null" \
    "checkpoint.coalesced_checkpoint_path=${WALRUS_PAPER_ROOT}/walrus_model/checkpoints/walrus.pt" \
    "${extra[@]}"
