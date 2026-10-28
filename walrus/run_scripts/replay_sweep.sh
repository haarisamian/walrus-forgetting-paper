#!/bin/bash
#SBATCH --partition=lovelace-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=replay_sweep
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_sweep_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_sweep_%A_%a.err
#SBATCH --array=1-9

# Replay ratio sweep: vary fraction of sim (cylinder_numerical) vs real (cylinder_real_cl)
# per epoch, using MultisetSampler sampling_weight. max_samples=200 batches total/epoch.
#
# Task i: sim_weight=i, real_weight=10-i
#   task 1 → 10% sim / 90% real  (~20 sim + 180 real batches/epoch)
#   task 5 → 50% sim / 50% real  (~100 sim + 100 real batches/epoch)
#   task 9 → 90% sim / 10% real  (~180 sim + 20 real batches/epoch)
#
# Baseline (0% sim / 100% real) = existing gn_to_gn_real100pct run.
# All arms start from the Phase 1 GN checkpoint (sim-only finetune of walrus.pt).
# gn_to_gn (job 213475) is the 0% replay baseline from the same starting point.

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
GN_CKPT="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0/checkpoints/best"

SIM_WEIGHT=${SLURM_ARRAY_TASK_ID}
REAL_WEIGHT=$((10 - SLURM_ARRAY_TASK_ID))
SIM_PCT=$((SLURM_ARRAY_TASK_ID * 10))

echo "=== replay_sweep task ${SLURM_ARRAY_TASK_ID}: ${SIM_PCT}% sim (sim_w=${SIM_WEIGHT}, real_w=${REAL_WEIGHT}) ==="

srun $PYTHON -u $TORCHRUN \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME:29500 \
    ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    distribution=fsdp \
    model=isotropic_model \
    trainer=globalnorm \
    trainer.grad_acc_steps=1 \
    server=local \
    optimizer=adam \
    optimizer.lr=5e-5 \
    logger.wandb_project_name="walrus_cylinder_cl" \
    trainer.enable_amp=False \
    model.gradient_checkpointing_freq=2 \
    trainer.log_interval=200 \
    trainer.clip_gradient=10 \
    data.module_parameters.batch_size=1 \
    data.module_parameters.n_steps_input=6 \
    data.module_parameters.n_steps_output=1 \
    model.projection_dim=48 \
    model.intermediate_dim=352 \
    model.hidden_dim=1408 \
    model.groups=16 \
    model.processor_blocks=40 \
    model.drop_path=0.0 \
    model/processor/space_mixing=full_spatial_attention \
    model.processor.space_mixing.num_heads=16 \
    model.processor.time_mixing.num_heads=16 \
    ++trainer.epsilon=1e-8 \
    model.causal_in_time=True \
    model.jitter_patches=True \
    data.module_parameters.max_samples=200 \
    trainer.short_validation_length=20 \
    trainer.max_rollout_steps=60 \
    lr_scheduler=inv_sqrt_w_sqrt_ramps_longer \
    ++lr_scheduler.warmup_epochs=3 \
    ++lr_scheduler.cooldown_epochs=5 \
    trainer.val_frequency=5 \
    trainer.rollout_val_frequency=5 \
    data.module_parameters.min_dt_stride=1 \
    data.module_parameters.max_dt_stride=1 \
    trainer.prediction_type="delta" \
    data=cylinder \
    trainer.max_epoch=100 \
    data_workers=10 \
    model.override_dimensionality=0 \
    auto_resume=True \
    checkpoint=finetune_cylinder \
    experiment=finetune_cylinder \
    ++model.use_periodic_fixed_jitter=True \
    ++model.input_field_drop=0 \
    ++trainer.skip_spectral_metrics=True \
    finetuning_mods=all \
    ++experiment_dir=${WALRUS_PAPER_ROOT}/walrus/experiments \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    checkpoint.coalesced_checkpoint_path=null \
    "checkpoint.load_checkpoint_path='$GN_CKPT'" \
    "name=walrus_replay${SIM_PCT}pct_sim" \
    "++data.module_parameters.well_dataset_info.cylinder_numerical.sampling_weight=${SIM_WEIGHT}" \
    "++data.module_parameters.well_dataset_info.cylinder_real_cl={path: ${WALRUS_PAPER_ROOT}/realpdebench/well/cylinder_real_cl/, sampling_weight: ${REAL_WEIGHT}}"
