#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --job-name=forget_eval_cyl_sim_phase1
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_cyl_sim_phase1_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_cyl_sim_phase1_%j.err

# Cyl_sim forget eval for phase1_gn_lr5e5 — the sim-only trained baseline.
# Provides the "perfect sim retention" reference in SamplewiseRevNorm eval,
# directly comparable to forget_eval_cyl_sim_replay results.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

EXP_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"
CKPT_PATH="$EXP_DIR/checkpoints/last"
CONFIG_DIR="$EXP_DIR"
FOLDER="$EXPBASE/forget_eval_results/phase1_gn_lr5e5/cyl_sim/finetune/0"

echo "=== forget_eval_cyl_sim phase1_gn_lr5e5 ==="
echo "CKPT: $CKPT_PATH"

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
    "++name=forget_cyl_sim_phase1_gn" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
