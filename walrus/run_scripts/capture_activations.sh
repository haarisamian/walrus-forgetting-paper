#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=3:00:00
#SBATCH --job-name=act_capture
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/act_capture_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/act_capture_%A_%a.err
#SBATCH --array=0-9

# Instrumented copy of the PRETRAINED forgetting eval. Identical invocation to
# forget_eval_new_well_pretrained.sh / forget_eval_checkpoints.sh (same config
# dirs, well paths, field transforms, and NO checkpoint overrides -- the config's
# coalesced_checkpoint_path already points at walrus.pt). The only difference is
# the entrypoint, which registers activation hooks and then calls train.main().

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

DS_NAMES=("acoustic_scattering_discontinuous" "acoustic_scattering_inclusions" \
          "acoustic_scattering_maze" "helmholtz_staircase" "planetswe" \
          "shear_flow" "active_matter" "MHD_64" "trl2d" "turbcool")
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
esac

export ACT_OUT="$EXPBASE/activation_grams"
export ACT_LABEL="$DS_NAME"
export ACT_BLOCKS="0,10,20,30,39"
export ACT_MAXB=6
FOLDER="$EXPBASE/act_capture_runs/${DS_NAME}/finetune/0"

echo "=== act_capture task ${SLURM_ARRAY_TASK_ID}: ${DS_NAME} ==="
echo "CONFIG: $CONFIG_DIR"; echo "WELL:   $WELL_PATH"

srun $PYTHON ${WALRUS_PAPER_ROOT}/realpdebench/scripts/capture_activations.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=act_capture_${DS_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_PATH}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "${extra[@]}"
