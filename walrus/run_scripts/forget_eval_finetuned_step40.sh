#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200gb
#SBATCH --time=3:00:00
#SBATCH --job-name=forget_eval_step40
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_finetuned_step40_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_finetuned_step40_%j.err

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

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$ACTIV_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    "++folder_override=$ACTIV_DIR" \
    ++logger.wandb_project_name="walrus_forgetting_eval" \
    ++name=walrus_forget_eval_step40 \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path=$CKPT_DIR/step_40" \
    ++checkpoint.prioritize_resume=False \
    data_workers=4 \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    data.module_parameters.batch_size=1
