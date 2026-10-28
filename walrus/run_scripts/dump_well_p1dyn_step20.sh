#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --job-name=dump_well_p1dyn
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dump_well_p1dyn_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dump_well_p1dyn_%A_%a.err
#SBATCH --array=0-9

# Dump clean prediction arrays (ypred_*.npy + yref_*.npy) for the Phase-I step_20
# checkpoint on all 10 Well benchmark datasets.
#
# Used to generate paper-quality appendix rollout field visualisations:
#   GT / Pred / Error at t=20, t=40, t=60 for each field, per dataset.
#
# Outputs written to:
#   experiments/dump_well_p1dyn_step20/{ds_name}/finetune/0/viz/{ds_viz}/
#       full_trajectory_dumps/ypred_*_0.npy
#       full_trajectory_dumps/yref_*_0.npy
#
# Array layout (task_id → dataset):
#   0  acoustic_scattering_discontinuous  (walrus_activ_pretrained config)
#   1  acoustic_scattering_inclusions     (walrus_activ_pretrained config)
#   2  acoustic_scattering_maze           (walrus_activ_pretrained config)
#   3  active_matter                      (walrus_activ_pretrained config; dataset already in config)
#   4  helmholtz_staircase                (walrus_activ_pretrained config)
#   5  MHD_64                             (walrus_MHD64_eval config; remove+re-add for downsample)
#   6  planetswe                          (walrus_activ_pretrained config)
#   7  shear_flow                         (walrus_activ_pretrained config)
#   8  turbulent_radiative_layer_2D       (walrus_trl2d_eval config; dataset already in config)
#   9  turbulence_gravity_cooling         (walrus_activ_pretrained config; log transforms)

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
DYN_CKPT="$EXPBASE/walrus_phase1_forgetting_dynamics-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/step_20"

WELL_A="${WALRUS_PAPER_ROOT}/the_well/datasets"           # active_matter
WELL_B="${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"  # all others

case $SLURM_ARRAY_TASK_ID in
    0)
        DS_NAME="acoustic_scattering_discontinuous"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1,field_transforms:{density:torch.zeros_like}}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    1)
        DS_NAME="acoustic_scattering_inclusions"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1,field_transforms:{density:torch.zeros_like}}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    2)
        DS_NAME="acoustic_scattering_maze"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1,field_transforms:{density:torch.zeros_like}}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    3)
        # active_matter is already in walrus_activ_pretrained config — no DS_EXTRA needed
        DS_NAME="active_matter"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_A"
        DS_EXTRA=()
        ;;
    4)
        DS_NAME="helmholtz_staircase"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    5)
        # MHD_64 has its own eval config; re-add with step_downsample_factor=0.5
        DS_NAME="MHD_64"
        CONFIG_DIR="$EXPBASE/walrus_MHD64_eval/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.MHD_64"
            "+data.module_parameters.well_dataset_info.MHD_64=${DS_ENTRY}"
        )
        ;;
    6)
        DS_NAME="planetswe"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    7)
        DS_NAME="shear_flow"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    8)
        # trl2d has its own eval config and is already set up correctly there
        DS_NAME="turbulent_radiative_layer_2D"
        CONFIG_DIR="$EXPBASE/walrus_trl2d_eval/finetune/0"
        WELL_BASE="$WELL_B"
        DS_EXTRA=()
        ;;
    9)
        # turbcool: uses activ_pretrained config; remove active_matter, add turbcool with log transforms
        DS_NAME="turbulence_gravity_cooling"
        CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
        WELL_BASE="$WELL_B"
        DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}"
        DS_EXTRA=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
        )
        ;;
    *)
        echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

FOLDER="$EXPBASE/dump_well_p1dyn_step20/${DS_NAME}/finetune/0"

echo "=== dump_well_p1dyn task ${SLURM_ARRAY_TASK_ID}: step_20 on ${DS_NAME} ==="
echo "CKPT:       $DYN_CKPT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "FOLDER:     $FOLDER"
echo "WELL_BASE:  $WELL_BASE"

if [ ! -d "$DYN_CKPT" ]; then
    echo "ERROR: checkpoint not found: $DYN_CKPT"; exit 1
fi

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${DYN_CKPT}'" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_BASE}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    ++trainer.dump_prediction_to_disk=True \
    ++trainer.num_detailed_logs=4 \
    ++trainer.video_validation=False \
    "${DS_EXTRA[@]}"
