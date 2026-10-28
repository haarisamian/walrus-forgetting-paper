#!/bin/bash
#SBATCH --partition=lovelace-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=7:00:00
#SBATCH --job-name=cl_eval_100pct
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/cl_eval_100pct_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/cl_eval_100pct_%j.err

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
TORCHRUN=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/torchrun
BEST_FT_DIR="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_ft_lr5e5_samplewise_best/finetune/0"
# 100pct: step_30 is final (no best symlink since val loss never improved past epoch 5)
CKPT="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_cl_real100pct_sim0pct/finetune/0/checkpoints/step_30"

srun $PYTHON -u $TORCHRUN \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
    ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="$BEST_FT_DIR" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=fsdp \
    ++validation_mode=True \
    ++checkpoint.prioritize_resume=False \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path=$CKPT" \
    ++logger.wandb_project_name="walrus_cylinder_cl_eval" \
    ++name=cl_eval_100pct \
    ++experiment_dir=${WALRUS_PAPER_ROOT}/walrus/experiments \
    "++data.module_parameters.well_dataset_info={cylinder_numerical: {path: ${WALRUS_PAPER_ROOT}/realpdebench/well/cylinder_numerical/}, cylinder_real_cl: {path: ${WALRUS_PAPER_ROOT}/realpdebench/well/cylinder_real_cl/}}" \
    data_workers=10 \
    data.module_parameters.batch_size=1
