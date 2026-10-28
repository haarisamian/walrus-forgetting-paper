#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --job-name=dump_well_pre
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dump_well_pre_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dump_well_pre_%A_%a.err
#SBATCH --array=0-9

# Dump clean prediction arrays for the PRETRAINED Walrus-1.3B model
# (before any cylinder finetuning) on all 10 Well benchmark datasets.
#
# Outputs: experiments/dump_well_pretrained/{ds_name}/finetune/0/viz/{ds_name}/
#              full_trajectory_dumps/ypred_*_0.npy
#              full_trajectory_dumps/yref_*_0.npy
#
# Array: same 10-dataset layout as dump_well_p1dyn_step20.sh
#   0  acoustic_scattering_discontinuous
#   1  acoustic_scattering_inclusions
#   2  acoustic_scattering_maze
#   3  active_matter
#   4  helmholtz_staircase
#   5  MHD_64
#   6  planetswe
#   7  shear_flow
#   8  turbulent_radiative_layer_2D
#   9  turbulence_gravity_cooling

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
PRETRAINED_CKPT="${WALRUS_PAPER_ROOT}/walrus_model/checkpoints/walrus.pt"

WELL_A="${WALRUS_PAPER_ROOT}/the_well/datasets"
WELL_B="${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"

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
        DS_NAME="turbulent_radiative_layer_2D"
        CONFIG_DIR="$EXPBASE/walrus_trl2d_eval/finetune/0"
        WELL_BASE="$WELL_B"
        DS_EXTRA=()
        ;;
    9)
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
        echo "ERROR: unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

FOLDER="$EXPBASE/dump_well_pretrained/${DS_NAME}/finetune/0"

echo "=== dump_well_pretrained task ${SLURM_ARRAY_TASK_ID}: pretrained on ${DS_NAME} ==="
echo "CKPT:       $PRETRAINED_CKPT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "FOLDER:     $FOLDER"

if [ ! -f "$PRETRAINED_CKPT" ]; then
    echo "ERROR: checkpoint not found: $PRETRAINED_CKPT"; exit 1
fi

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.load_checkpoint_path=null \
    "checkpoint.coalesced_checkpoint_path='${PRETRAINED_CKPT}'" \
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
