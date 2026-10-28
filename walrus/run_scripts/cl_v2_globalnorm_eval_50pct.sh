#!/bin/bash
#SBATCH --partition=ampere-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200gb
#SBATCH --time=2:00:00
#SBATCH --job-name=globalnorm_eval_50pct
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/globalnorm_eval_50pct_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/globalnorm_eval_50pct_%j.err

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
# Symlink to v2 50% experiment dir (no brackets) — gives v2 config with cylinder_real_cl,
# short_validation_length=20, and start_rollout_valid_output_at_t=-1 already set
V2_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_cl_v2_50pct_symlink/finetune/0"

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$V2_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='$V2_DIR/checkpoints/step_50'" \
    ++logger.wandb_project_name="walrus_cylinder_cl_eval" \
    ++name=cl_v2_globalnorm_eval_50pct \
    "++folder_override=${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_cl_v2_50pct_globalnorm_eval/finetune/0" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.RMSGlobalRevNormalization" \
    "++trainer.debug_mode=True"
