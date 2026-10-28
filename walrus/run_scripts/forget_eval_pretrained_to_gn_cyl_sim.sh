#!/bin/bash
#SBATCH --partition=lovelace-mc,ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=2:00:00
#SBATCH --job-name=p2gn_cyl_eval
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/p2gn_cyl_eval_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/p2gn_cyl_eval_%j.err

# Cyl-sim rollout NRMSE eval for pretrained_to_gn (walrus.pt → GN → real PIV, no Phase-1 sim).
# Measures whether skipping Phase-1 changes cyl-sim retention.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
CONFIG_DIR="$EXPBASE/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0"
CKPT_PATH="$EXPBASE/walrus_pretrained_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/step_100"
FOLDER="$EXPBASE/forget_eval_results/pretrained_to_gn/cyl_sim/finetune/0"

echo "=== pretrained_to_gn cyl_sim eval | ckpt: $CKPT_PATH ==="

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
    "++name=forget_cyl_sim_pretrained_to_gn" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True
