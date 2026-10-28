#!/bin/bash
#SBATCH --partition=lovelace-mc,ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --job-name=replay_fine
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_fine_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/replay_fine_%j.err

# Self-chaining replay fine sweep.
# SIM_PCT and HOP are hardcoded at submission time (no --export needed).
# Successors are generated with sed and submitted without --export,
# avoiding Slurm's user environment retrieval entirely.

# ── Task identity (hardcoded by submit_replay_fine.sh) ────────────────────────
WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
SIM_PCT=3
HOP=2
MAX_HOPS=8

SCRIPT_DIR="${WALRUS_PAPER_ROOT}/walrus/walrus/run_scripts"
EXP_BASE="${WALRUS_PAPER_ROOT}/walrus/experiments"
GN_CKPT="${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_ft_lr5e5_gn_best/finetune/0/checkpoints/best"

SIM_WEIGHT=${SIM_PCT}
REAL_WEIGHT=$((100 - SIM_PCT))
N_GPUS=${SLURM_GPUS_PER_NODE}

echo "=== replay_fine SIM_PCT=${SIM_PCT}% | hop=${HOP}/${MAX_HOPS} | GPUs=${N_GPUS} | node=${SLURMD_NODENAME} ==="

# ── Completion check ──────────────────────────────────────────────────────────
CKPT_BASE=$(ls -d ${EXP_BASE}/walrus_replay${SIM_PCT}pct_sim*/finetune/0/checkpoints 2>/dev/null | head -1)
if [ -d "${CKPT_BASE}/step_100" ]; then
    echo "step_100 found — training complete. Chain ends."
    exit 0
fi

# ── Safety cap ────────────────────────────────────────────────────────────────
if [ "${HOP}" -ge "${MAX_HOPS}" ]; then
    echo "ERROR: max hops (${MAX_HOPS}) reached without step_100. Stopping."
    exit 1
fi

# ── Submit successor by generating a new script with next HOP baked in ───────
NEXT_HOP=$((HOP + 1))
SUCC_SCRIPT="${SCRIPT_DIR}/replay_${SIM_PCT}pct_hop${NEXT_HOP}.sh"
sed "s/^SIM_PCT=.*/SIM_PCT=${SIM_PCT}/; s/^HOP=.*/HOP=${NEXT_HOP}/" \
    "${BASH_SOURCE[0]}" > "${SUCC_SCRIPT}"
chmod +x "${SUCC_SCRIPT}"
SUCC_OUT=$(sbatch --dependency=afternotok:${SLURM_JOB_ID} "${SUCC_SCRIPT}" 2>&1)
echo "Successor scheduled: ${SUCC_OUT}  (fires only on timeout/crash)"

# ── Environment ───────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
TORCHRUN=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/torchrun

# ── Training ──────────────────────────────────────────────────────────────────
srun $PYTHON -u $TORCHRUN \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=${N_GPUS} \
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
    "checkpoint.load_checkpoint_path=${GN_CKPT}" \
    name=walrus_replay${SIM_PCT}pct_sim \
    "++data.module_parameters.well_dataset_info.cylinder_numerical.sampling_weight=${SIM_WEIGHT}" \
    "++data.module_parameters.well_dataset_info.cylinder_real_cl={path: ${WALRUS_PAPER_ROOT}/realpdebench/well/cylinder_real_cl/, sampling_weight: ${REAL_WEIGHT}}"
