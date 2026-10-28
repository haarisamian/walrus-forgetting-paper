#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_eval_cyl_sim
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_cyl_sim_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_cyl_sim_%A_%a.err
#SBATCH --array=0-4

# Cylinder sim (cylinder_numerical) forgetting eval for phase-2 checkpoints
# that were trained ONLY on real data and have no cylinder_sim VRMSE in training logs.
#
# Checkpoints:
#   0  sw_to_gn_5e5_ep100   (SW→GN lr=5e-5)
#   1  sw_to_gn_1e4_ep100   (SW→GN lr=1e-4)
#   2  sw_lr1e4_ep100        (SW lr=1e-4)
#   3  gn_to_gn              (GN→GN lr=5e-5)
#   4  gn_to_gn_lr1e4        (GN→GN lr=1e-4)
#
# Uses phase1_gn config dir as base (has cylinder_numerical configured).
# PKL written to: forget_eval_results/{ckpt_name}/cyl_sim/finetune/0/

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

# Base config: phase1_gn has cylinder_numerical in its data config
CONFIG_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"

CKPT_NAMES=(
    "sw_to_gn_5e5_ep100"
    "sw_to_gn_1e4_ep100"
    "sw_lr1e4_ep100"
    "gn_to_gn"
    "gn_to_gn_lr1e4"
)
CKPT_PATHS=(
    "$EXPBASE/walrus_sw_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_sw_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_sw_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/last"
    "$EXPBASE/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001/finetune/0/checkpoints/last"
)

IDX=$SLURM_ARRAY_TASK_ID
CKPT_NAME="${CKPT_NAMES[$IDX]}"
CKPT_PATH="${CKPT_PATHS[$IDX]}"
FOLDER="$EXPBASE/forget_eval_results/${CKPT_NAME}/cyl_sim/finetune/0"

echo "=== forget_eval_cyl_sim task ${IDX}: ${CKPT_NAME} ==="
echo "CKPT: $CKPT_PATH"

if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

# Norm must match the model's training norm for cylinder data:
#   idx 0 (sw_to_gn_5e5): SW-pretrained, never trained on cyl_sim → use SW
#   idx 1 (sw_to_gn_1e4): SW-pretrained, never trained on cyl_sim → use SW
#   idx 2 (sw_lr1e4):     SW throughout (pretrain + finetune on cyl_real) → use SW
#   idx 3 (gn_to_gn):     phase1_gn cyl_sim trained with GN → use GN (no override)
#   idx 4 (gn_to_gn_lr1e4): same as gn_to_gn → use GN (no override)
if [ "$IDX" -le 2 ]; then
    NORM_ARGS=("++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization")
    echo "NORM: SamplewiseRevNormalization (idx=$IDX: ${CKPT_NAMES[$IDX]})"
else
    NORM_ARGS=()
    echo "NORM: GlobalRevNormalization from config (idx=$IDX: ${CKPT_NAMES[$IDX]})"
fi

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$CONFIG_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    "++folder_override=$FOLDER" \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='${CKPT_PATH}'" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    "++name=forget_cyl_sim_${CKPT_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "${NORM_ARGS[@]}" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
