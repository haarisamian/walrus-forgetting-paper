#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --job-name=forget_video
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_video_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_video_%A_%a.err
#SBATCH --array=0-8

# Video rollout evaluation for visual plausibility / catastrophic failure detection.
# Each task saves mp4s to a unique folder under walrus_video_eval/.
# Videos: 3 rollout batches × 1 trajectory each → 3 side-by-side (pred|truth|error) mp4s per task.
#
# Array layout (9 tasks):
#   0  pretrained walrus.pt   active_matter
#   1  pretrained walrus.pt   MHD_64
#   2  pretrained walrus.pt   turbcool
#   3  pretrained walrus.pt   TRL_2D
#   4  phase1_gn_lr5e5        active_matter
#   5  phase1_gn_lr5e5        MHD_64
#   6  phase1_gn_lr5e5        turbcool
#   7  phase1_gn_lr5e5        TRL_2D
#   8  sw_to_gn_1e4_ep100     MHD_64  (catastrophic forgetter — rollout diverges)

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
VIDBASE="${WALRUS_PAPER_ROOT}/walrus/video_evals"

# Per-dataset base config dirs
CONFIG_ACTIV="$EXPBASE/walrus_activ_pretrained/finetune/0"
CONFIG_MHD="$EXPBASE/walrus_MHD64_eval/finetune/0"
CONFIG_TURBCOOL="$EXPBASE/walrus_activ_pretrained/finetune/0"  # reuse activ config, override dataset below
CONFIG_TRL2D="$EXPBASE/walrus_trl2d_eval/finetune/0"

# Checkpoint args per row: 0=pretrained, 1=phase1_gn, 2=sw_to_gn_1e4
PHASE1_GN="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"
SW_TO_GN_1E4="$EXPBASE/walrus_sw_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"

case $SLURM_ARRAY_TASK_ID in
    0) CKPT_LABEL="pretrained"; DS="active_matter"; CONFIG_DIR="$CONFIG_ACTIV" ;;
    1) CKPT_LABEL="pretrained"; DS="MHD_64";        CONFIG_DIR="$CONFIG_MHD" ;;
    2) CKPT_LABEL="pretrained"; DS="turbcool";      CONFIG_DIR="$CONFIG_TURBCOOL" ;;
    3) CKPT_LABEL="pretrained"; DS="TRL_2D";        CONFIG_DIR="$CONFIG_TRL2D" ;;
    4) CKPT_LABEL="phase1_gn";  DS="active_matter"; CONFIG_DIR="$CONFIG_ACTIV" ;;
    5) CKPT_LABEL="phase1_gn";  DS="MHD_64";        CONFIG_DIR="$CONFIG_MHD" ;;
    6) CKPT_LABEL="phase1_gn";  DS="turbcool";      CONFIG_DIR="$CONFIG_TURBCOOL" ;;
    7) CKPT_LABEL="phase1_gn";  DS="TRL_2D";        CONFIG_DIR="$CONFIG_TRL2D" ;;
    8) CKPT_LABEL="sw_gn_1e4";  DS="MHD_64";        CONFIG_DIR="$CONFIG_MHD" ;;
esac

VIZ_DIR="$VIDBASE/${CKPT_LABEL}_${DS}/finetune/0"

echo "=== forget_eval_video task ${SLURM_ARRAY_TASK_ID}: ${CKPT_LABEL} on ${DS} ==="
echo "CONFIG: $CONFIG_DIR"
echo "VIZ_DIR: $VIZ_DIR"

# Checkpoint args
if [ "$CKPT_LABEL" = "pretrained" ]; then
    ckpt_args=(
        "checkpoint.coalesced_checkpoint_path=${WALRUS_PAPER_ROOT}/walrus_model/checkpoints/walrus.pt"
        "checkpoint.load_checkpoint_path=null"
        "++checkpoint.prioritize_resume=False"
        "++finetuning_mods.learnable_rope=False"
        "++finetuning_mods.rope_per_axis=False"
    )
elif [ "$CKPT_LABEL" = "phase1_gn" ]; then
    ckpt_args=(
        "checkpoint.coalesced_checkpoint_path=null"
        "checkpoint.load_checkpoint_path='$PHASE1_GN'"
        "++checkpoint.prioritize_resume=False"
        "++finetuning_mods.learnable_rope=True"
        "++finetuning_mods.rope_per_axis=True"
    )
elif [ "$CKPT_LABEL" = "sw_gn_1e4" ]; then
    ckpt_args=(
        "checkpoint.coalesced_checkpoint_path=null"
        "checkpoint.load_checkpoint_path='$SW_TO_GN_1E4'"
        "++checkpoint.prioritize_resume=False"
        "++finetuning_mods.learnable_rope=True"
        "++finetuning_mods.rope_per_axis=True"
    )
fi

# Dataset-specific args
case $DS in
    active_matter)
        ds_args=()
        ;;
    MHD_64)
        ds_args=(
            "~data.module_parameters.well_dataset_info.MHD_64"
            "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
        )
        ;;
    turbcool)
        # Override active_matter dataset → turbcool with required transforms
        ds_args=(
            "~data.module_parameters.well_dataset_info.active_matter"
            "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}"
        )
        ;;
    TRL_2D)
        ds_args=()
        ;;
esac

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    "++folder_override=$VIZ_DIR" \
    "++data.well_base_path=${WALRUS_PAPER_ROOT}/the_well/datasets/datasets" \
    ++logger.wandb_project_name="walrus_forgetting_video" \
    "++name=video_${CKPT_LABEL}_${DS}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++trainer.video_validation=True \
    ++trainer.video_size_multiplier=0.8 \
    ++trainer.num_detailed_logs=3 \
    "${ckpt_args[@]}" \
    "${ds_args[@]}"
