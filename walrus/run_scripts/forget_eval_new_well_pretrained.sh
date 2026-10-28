#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_new_well_pre
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_new_well_pretrained_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_new_well_pretrained_%A_%a.err
#SBATCH --array=0-5

# Pretrained-baseline forgetting eval on 6 new Well datasets.
# These datasets were not in the original Well pretraining suite — they provide a
# reference VRMSE denominator for forgetting ratios across all downstream checkpoints.
#
# Dataset index (task_id):
#   0  acoustic_scattering_discontinuous   (2D 256×256, 102t)
#   1  acoustic_scattering_inclusions      (2D 256×256, 102t)
#   2  acoustic_scattering_maze            (2D 256×256, 202t)
#   3  planetswe                           (2D 256×512, 1008t) — step_downsample=0.5
#   4  helmholtz_staircase                 (2D 1024×256, 50t)
#   5  shear_flow                          (2D 256×512, 200t) — submit after download complete
#
# Uses walrus_activ_pretrained config base:
#   - video_validation: true (all are 2D, videos will work)
#   - SamplewiseRevNorm (all pretrained with SW)
#
# PKL written to: forget_eval_results/pretrained/{ds_name}/finetune/0/

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

# walrus_activ_pretrained config already has coalesced_checkpoint_path pointing to walrus.pt.
# Do NOT override checkpoint paths here — let the config load walrus.pt via coalesced path.
CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"

DS_NAMES=(
    "acoustic_scattering_discontinuous"
    "acoustic_scattering_inclusions"
    "acoustic_scattering_maze"
    "planetswe"
    "helmholtz_staircase"
    "shear_flow"
)

# step_downsample_factor: planetswe has 1008 timesteps, use 0.5; others are fine at 1
STEP_DOWNSAMPLE=(1 1 1 0.5 1 1)

IDX=$SLURM_ARRAY_TASK_ID
DS_NAME="${DS_NAMES[$IDX]}"
DOWNSAMPLE="${STEP_DOWNSAMPLE[$IDX]}"
FOLDER="$EXPBASE/forget_eval_results/pretrained/${DS_NAME}/finetune/0"

# acoustic_scattering datasets: density is a static background field zeroed during pretraining.
# Must match pretraining: field_transforms={density:torch.zeros_like}.
# All other new well datasets: no field transforms.
if [ "$IDX" -le 2 ]; then
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:${DOWNSAMPLE},field_transforms:{density:torch.zeros_like}}"
    echo "FIELD_TRANSFORM: density→zeros_like (acoustic_scattering background field)"
else
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:${DOWNSAMPLE}}"
fi

echo "=== forget_new_well_pretrained task ${IDX}: ${DS_NAME} (downsample=${DOWNSAMPLE}) ==="
echo "CKPT: walrus.pt (via walrus_activ_pretrained config coalesced_checkpoint_path)"

# Verify dataset exists
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
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=forget_pretrained_${DS_NAME}" \
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
