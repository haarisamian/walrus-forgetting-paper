#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --exclude=sw-amp01,sw-amp02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200gb
#SBATCH --time=2:00:00
#SBATCH --job-name=dyn_metrics_eval
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dyn_metrics_eval_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/dyn_metrics_eval_%A_%a.err
#SBATCH --array=0-3

# Array comparison: expose delta collapse via PSD L2, Delta VRMSE, Fluct VRMSE
#   0: samplewise best (ep50)             — SamplewiseRevNorm → delta collapse baseline
#   1: CLv2 100%real samplewise (ep44)    — SamplewiseRevNorm → ceteris paribus collapse
#   2: sw_to_gn real100 last (ep47)       — RMSGlobalRevNorm  → ceteris paribus fix
#   3: original GN finetune best (ep51)   — RMSGlobalRevNorm  → GN baseline

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXP_BASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

V2_DIR="$EXP_BASE/walrus_cl_v2_50pct_symlink/finetune/0"

CKPT_DIRS=(
    "$EXP_BASE/walrus_ft_lr5e5_samplewise_best/finetune/0/checkpoints/best"
    "$EXP_BASE/walrus_cl_v2_real100pct_sim0pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXP_BASE/walrus_sw_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXP_BASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"
)
NAMES=(
    "dyneval_sw_best_ep50"
    "dyneval_clv2_real100_sw_ep44"
    "dyneval_sw_to_gn_ep47"
    "dyneval_orig_gn_ep51"
)
NORMS=(
    "walrus.trainer.normalization_strat.SamplewiseRevNormalization"
    "walrus.trainer.normalization_strat.SamplewiseRevNormalization"
    "walrus.trainer.normalization_strat.RMSGlobalRevNormalization"
    "walrus.trainer.normalization_strat.RMSGlobalRevNormalization"
)

CKPT="${CKPT_DIRS[$SLURM_ARRAY_TASK_ID]}"
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
NORM="${NORMS[$SLURM_ARRAY_TASK_ID]}"
FOLDER="$EXP_BASE/walrus_${NAME}/finetune/0"

echo "=== dyn_metrics_eval task ${SLURM_ARRAY_TASK_ID}: ${NAME} ==="
echo "CKPT: $CKPT"
echo "NORM: $NORM"

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$V2_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT}'" \
    ++logger.wandb_project_name="walrus_cylinder_dynamics_eval" \
    "++name=${NAME}" \
    "++folder_override=${FOLDER}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=${NORM}" \
    ++trainer.compute_psd_metric=True \
    ++trainer.compute_delta_vrmse_metric=True \
    ++trainer.compute_fluct_vrmse_metric=True \
    ++trainer.debug_mode=True
