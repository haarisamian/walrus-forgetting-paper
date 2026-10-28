#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200gb
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_turbcool
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_turbcool_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_turbcool_%A_%a.err
#SBATCH --array=0-4

# Array: 0=pretrained walrus.pt, 1=best, 2=step_20, 3=step_40, 4=step_51
# turbulence_gravity_cooling rollout has 270 batches × 139s each = 10.4h untruncated.
# short_validation_length=20 caps rollout to 20 batches × 139s × 2 (valid+test) = ~1.5h per task.
# min(270, 300)=270 was the bug — the cap was larger than the dataloader, so nothing was skipped.
# SamplewiseRevNormalization added for fair comparison with phase-2 models.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
ACTIV_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_activ_pretrained/finetune/0"
CKPT_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_ft_lr5e5_samplewise_best/finetune/0/checkpoints"
EVAL_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_turbcool_eval/finetune/0"
DATASET_KEY="turbulence_gravity_cooling"

CKPT_NAMES=("pretrained" "best" "step_20" "step_40" "step_51")
CKPT_NAME=${CKPT_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "=== forget eval ${DATASET_KEY} task ${SLURM_ARRAY_TASK_ID}: ${CKPT_NAME} ==="

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    extra_args=(
        "++checkpoint.prioritize_resume=False"
        "++finetuning_mods.learnable_rope=False"
        "++finetuning_mods.rope_per_axis=False"
    )
else
    extra_args=(
        "checkpoint.coalesced_checkpoint_path=null"
        "checkpoint.load_checkpoint_path=$CKPT_DIR/$CKPT_NAME"
        "++checkpoint.prioritize_resume=False"
    )
fi

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$ACTIV_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    "++folder_override=$EVAL_DIR" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=walrus_forget_eval_turbcool_${CKPT_NAME}" \
    "${extra_args[@]}" \
    data_workers=4 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    data.module_parameters.batch_size=1 \
    "~data.module_parameters.well_dataset_info.active_matter" \
    "+data.module_parameters.well_dataset_info.${DATASET_KEY}={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}" \
    "++data.well_base_path=${WALRUS_PAPER_ROOT}/the_well/datasets/datasets" \
    ++trainer.image_validation=false \
    ++trainer.video_validation=false \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++trainer.debug_mode=True \
    ++trainer.short_validation_length=20
