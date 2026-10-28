#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=p2_traj
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/p2_traj_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/p2_traj_%A_%a.err
#SBATCH --array=0-5

# Phase-2 forgetting for the six Well datasets never evaluated after Phase 2.
# Brings Phase-2 coverage from 4 datasets to 10, matching Phase 1.
#
# WHY: Phase-2 damage is currently unexplained. turbcool is the sole anomaly --
# it peaks at ep111 and partially RECOVERS by ep151, in both VRMSE and PearsonR,
# while all others degrade monotonically. With n=4 the cause is unidentifiable.
# If recovery appears in turbulent datasets but not wave ones, that supports
# positive transfer from real PIV turbulence.
#
# 3 checkpoints x 6 datasets = 18 tasks (ckpt*6 + ds)
#   ep51  = Phase-1 end        (baseline for Phase-2 damage)
#   ep111 = gn_to_gn step_60   (where turbcool peaks)
#   ep151 = gn_to_gn step_100  (end of Phase 2)
# Output dirs reuse existing names so the analysis scripts work unchanged.

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
P1B="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints"
G2B="$EXPBASE/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints"

CK_PATHS=("$G2B/step_20" "$G2B/step_40" "$G2B/step_80")
CK_NAMES=("gn_to_gn_step20" "gn_to_gn_step40" "gn_to_gn_step80")
DS_NAMES=("helmholtz_staircase" "planetswe")

CK_IDX=$((SLURM_ARRAY_TASK_ID / 2))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 2))
CKPT="${CK_PATHS[$CK_IDX]}"
CKNAME="${CK_NAMES[$CK_IDX]}"
DS_NAME="${DS_NAMES[$DS_IDX]}"

CONFIG_DIR="$EXPBASE/walrus_activ_pretrained/finetune/0"
case "$DS_NAME" in
  acoustic_scattering_*)
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1,field_transforms:{density:torch.zeros_like}}" ;;
  planetswe)
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:0.5}" ;;
  *)
    DS_ENTRY="{include_filters:[],exclude_filters:[],step_downsample_factor:1}" ;;
esac

FOLDER="$EXPBASE/forget_eval_results/${CKNAME}/${DS_NAME}/finetune/0"

echo "=== p2_missing task ${SLURM_ARRAY_TASK_ID}: ${CKNAME} on ${DS_NAME} ==="
echo "CKPT: $CKPT"
if [ ! -d "$CKPT" ]; then echo "ERROR: checkpoint missing: $CKPT"; exit 1; fi
if [ ! -d "$WELL_DD/$DS_NAME/data/valid" ]; then echo "ERROR: dataset missing"; exit 1; fi

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT}'" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=p2_missing_${CKNAME}_${DS_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_DD}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "~data.module_parameters.well_dataset_info.active_matter" \
    "+data.module_parameters.well_dataset_info.${DS_NAME}=${DS_ENTRY}"
