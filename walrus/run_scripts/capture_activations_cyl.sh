#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --job-name=act_cyl
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/act_cyl_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/act_cyl_%A_%a.err
#SBATCH --array=0-3

# Activation covariances for the two FINE-TUNING datasets, so representational
# overlap between what we train ON and what we forget can be measured directly.
#   cylinder_numerical -> Phase I target (simulation)
#   cylinder_real_cl   -> Phase II target (real PIV)
# Same instrument, blocks and token budget as capture_activations_v2.sh, so the
# resulting covariances are directly comparable to the ten Well datasets.
# Layout: arm*2 + ds_idx    arms = (pretrained, phase1)

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
P1CKPT="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/step_51"
SIM_CFG="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"
REAL_CFG="$EXPBASE/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

DS_NAMES=("cylinder_numerical" "cylinder_real_cl")
CFGS=("$SIM_CFG" "$REAL_CFG")
ARMS=("pretrained" "phase1")
ARM_IDX=$((SLURM_ARRAY_TASK_ID / 2))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 2))
DS_NAME="${DS_NAMES[$DS_IDX]}"
CONFIG_DIR="${CFGS[$DS_IDX]}"
ARM="${ARMS[$ARM_IDX]}"

# pretrained arm: force walrus.pt and ignore whatever the run resumed from.
# phase1 arm: load step_51 instead.
if [ "$ARM" = "phase1" ]; then
  ckpt=("checkpoint.coalesced_checkpoint_path=null"
        "checkpoint.load_checkpoint_path='${P1CKPT}'")
else
  ckpt=("checkpoint.load_checkpoint_path=null"
        "checkpoint.coalesced_checkpoint_path=${WALRUS_PAPER_ROOT}/walrus_model/checkpoints/walrus.pt")
fi

export ACT_OUT="$EXPBASE/activation_grams_v2/${ARM}"
export ACT_LABEL="$DS_NAME"
export ACT_BLOCKS="0,5,10,15,20,25,30,39"
export ACT_MAXB=6
FOLDER="$EXPBASE/act_capture_runs_v2/${ARM}/${DS_NAME}/finetune/0"

echo "=== act_cyl task ${SLURM_ARRAY_TASK_ID}: ${ARM} / ${DS_NAME} ==="

srun $PYTHON ${WALRUS_PAPER_ROOT}/realpdebench/scripts/capture_activations.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=act_cyl_${ARM}_${DS_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WALRUS_PAPER_ROOT}/realpdebench/well" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "${ckpt[@]}"
