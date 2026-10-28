#!/bin/bash
#SBATCH --partition=ampere-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --job-name=walrus_eval_cyl_real_lr5e5_sw
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/eval_cylinder_real_lr5e5_samplewise_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/eval_cylinder_real_lr5e5_samplewise_%j.err

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPERIMENT=${WALRUS_PAPER_ROOT}/walrus/experiments/walrus_finetune_cylinder_lr5e5_samplewise-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05/finetune/0

srun $PYTHON ${WALRUS_PAPER_ROOT}/walrus/walrus/train.py \
    --config-path="${WALRUS_PAPER_ROOT}/walrus_model/configs/" \
    --config-name="extended_config.yaml" \
    ++distribution.distribution_type=local \
    ++validation_mode=True \
    ++finetune=False \
    ++config_override="${EXPERIMENT}/extended_config.yaml" \
    ++checkpoint_override="" \
    ++folder_override="${EXPERIMENT}/" \
    data=cylinder_real \
    ++data.well_base_path="${WALRUS_PAPER_ROOT}/realpdebench/well" \
    ++data.module_parameters.batch_size=1 \
    ++data.module_parameters.max_dt_stride=1 \
    ++data.module_parameters.max_rollout_steps=60 \
    ++data.module_parameters.start_rollout_valid_output_at_t=17 \
    ++trainer.max_rollout_steps=60 \
    ++trainer.skip_spectral_metrics=True \
    ++trainer.debug_mode=True \
    ++trainer.short_validation_length=500 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    ++logger.wandb_project_name="walrus_cylinder_finetune" \
    "++trainer.validation_suite=[{_target_:the_well.benchmark.metrics.NRMSE},{_target_:the_well.benchmark.metrics.VRMSE},{_target_:the_well.benchmark.metrics.PearsonR}]" \
    "++trainer.batch_aggregation_fns=[torch.mean,torch.median,torch.std]"
