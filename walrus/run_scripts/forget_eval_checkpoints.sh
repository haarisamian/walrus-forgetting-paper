#!/bin/bash
#SBATCH --partition=lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=4:00:00
#SBATCH --job-name=forget_ckpt
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_checkpoints_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/forget_eval_checkpoints_%A_%a.err
#SBATCH --array=0-191

# Checkpoint-level forgetting eval: Well datasets evaluated at every saved checkpoint
# during training, giving Well VRMSE vs. training progress curves.
#
# Covers: replay10-90pct (all step_XX + last), phase1_gn (step_20/40/51),
#         gn_to_gn_lr5e5 (step_20/40/60/80/100), gn_to_gn_lr1e4 (step_20/40/60/80)
#
# Array layout: (exp×ckpt_pair) × 4 datasets = N_PAIRS × 4 tasks
#   task_id = pair_idx * 4 + ds_idx
#
# Results written to: experiments/forget_eval_results/{exp_label}/step_{XX}/{ds}/finetune/0/
# Combine with final-checkpoint results using collect_forgetting_results.py.

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

PYTHON=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python
EXPBASE="${WALRUS_PAPER_ROOT}/walrus/experiments"

# ── Flat list of (exp_label, ckpt_subdir) pairs ────────────────────────────
# exp_label is used in the output folder name
# ckpt_subdir is the checkpoint directory name under checkpoints/
#
# replay experiments: step_20 step_40 [step_60] [step_80] last
# pair indices: 0-based

EXP_LABELS=()
CKPT_SUBDIRS=()
EXP_DIRS=()

add_pair() {
    EXP_LABELS+=("$1")
    CKPT_SUBDIRS+=("$2")
    EXP_DIRS+=("$3")
}

R="$EXPBASE"

# replay10pct: step_20 step_40 step_60 step_80 last  (5 ckpts, pairs 0-4)
add_pair "replay10pct_sim" "step_20" "$R/walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay10pct_sim" "step_40" "$R/walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay10pct_sim" "step_60" "$R/walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay10pct_sim" "step_80" "$R/walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay10pct_sim" "last"    "$R/walrus_replay10pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay20pct: step_20 step_40 last  (3 ckpts, pairs 5-7)
add_pair "replay20pct_sim" "step_20" "$R/walrus_replay20pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay20pct_sim" "step_40" "$R/walrus_replay20pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay20pct_sim" "last"    "$R/walrus_replay20pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay30pct: step_20 step_40 last  (3 ckpts, pairs 8-10)
add_pair "replay30pct_sim" "step_20" "$R/walrus_replay30pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay30pct_sim" "step_40" "$R/walrus_replay30pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay30pct_sim" "last"    "$R/walrus_replay30pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay40pct: step_20 step_40 step_60 last  (4 ckpts, pairs 11-14)
add_pair "replay40pct_sim" "step_20" "$R/walrus_replay40pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay40pct_sim" "step_40" "$R/walrus_replay40pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay40pct_sim" "step_60" "$R/walrus_replay40pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay40pct_sim" "last"    "$R/walrus_replay40pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay50pct: step_20 step_40 step_60 last  (4 ckpts, pairs 15-18)
add_pair "replay50pct_sim" "step_20" "$R/walrus_replay50pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay50pct_sim" "step_40" "$R/walrus_replay50pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay50pct_sim" "step_60" "$R/walrus_replay50pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay50pct_sim" "last"    "$R/walrus_replay50pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay60pct: step_20 step_40 step_60 last  (4 ckpts, pairs 19-22)
add_pair "replay60pct_sim" "step_20" "$R/walrus_replay60pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay60pct_sim" "step_40" "$R/walrus_replay60pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay60pct_sim" "step_60" "$R/walrus_replay60pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay60pct_sim" "last"    "$R/walrus_replay60pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay70pct: step_20 step_40 step_60 last  (4 ckpts, pairs 23-26)
add_pair "replay70pct_sim" "step_20" "$R/walrus_replay70pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay70pct_sim" "step_40" "$R/walrus_replay70pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay70pct_sim" "step_60" "$R/walrus_replay70pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay70pct_sim" "last"    "$R/walrus_replay70pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay80pct: step_20 step_40 step_60 last  (4 ckpts, pairs 27-30)
add_pair "replay80pct_sim" "step_20" "$R/walrus_replay80pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay80pct_sim" "step_40" "$R/walrus_replay80pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay80pct_sim" "step_60" "$R/walrus_replay80pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay80pct_sim" "last"    "$R/walrus_replay80pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# replay90pct: step_20 step_40 step_60 step_80 last  (5 ckpts, pairs 31-35)
add_pair "replay90pct_sim" "step_20" "$R/walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay90pct_sim" "step_40" "$R/walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay90pct_sim" "step_60" "$R/walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay90pct_sim" "step_80" "$R/walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "replay90pct_sim" "last"    "$R/walrus_replay90pct_sim-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# phase1_gn_lr5e5: step_20 step_40 step_51  (pairs 36-38; best already in main eval)
add_pair "phase1_gn_lr5e5" "step_20" "$R/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "phase1_gn_lr5e5" "step_40" "$R/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "phase1_gn_lr5e5" "step_51" "$R/walrus_finetune_cylinder_lr5e5-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# gn_to_gn_lr5e5: step_20 step_40 step_60 step_80 step_100  (pairs 39-43; last already in main eval)
add_pair "gn_to_gn"        "step_20"  "$R/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "gn_to_gn"        "step_40"  "$R/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "gn_to_gn"        "step_60"  "$R/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "gn_to_gn"        "step_80"  "$R/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"
add_pair "gn_to_gn"        "step_100" "$R/walrus_gn_to_gn_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-5e-05"

# gn_to_gn_lr1e4: step_20 step_40 step_60 step_80  (pairs 44-47; last already in main eval)
add_pair "gn_to_gn_lr1e4"  "step_20"  "$R/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001"
add_pair "gn_to_gn_lr1e4"  "step_40"  "$R/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001"
add_pair "gn_to_gn_lr1e4"  "step_60"  "$R/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001"
add_pair "gn_to_gn_lr1e4"  "step_80"  "$R/walrus_gn_to_gn_lr1e4_real100pct-cylin-delta-Isotr[Space-Adapt-]-AdamW-0.0001"

# Total pairs: 5+3+3+4+4+4+4+4+5 + 3 + 5 + 4 = 48 pairs × 4 datasets = 192 tasks (0-191)

# ── Dataset config ──────────────────────────────────────────────────────────
DS_NAMES=("MHD_64" "trl2d" "turbcool" "active_matter")
DS_CONFIG_DIRS=(
    "$EXPBASE/walrus_MHD64_eval/finetune/0"
    "$EXPBASE/walrus_trl2d_eval/finetune/0"
    "$EXPBASE/walrus_activ_pretrained/finetune/0"
    "$EXPBASE/walrus_activ_pretrained/finetune/0"
)
DS_WELL_PATHS=(
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets/datasets"
    "${WALRUS_PAPER_ROOT}/the_well/datasets"
)

# ── Derive task parameters ──────────────────────────────────────────────────
PAIR_IDX=$((SLURM_ARRAY_TASK_ID / 4))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 4))

EXP_LABEL="${EXP_LABELS[$PAIR_IDX]}"
CKPT_SUBDIR="${CKPT_SUBDIRS[$PAIR_IDX]}"
EXP_DIR="${EXP_DIRS[$PAIR_IDX]}"
CKPT_PATH="$EXP_DIR/finetune/0/checkpoints/$CKPT_SUBDIR"

DS_NAME="${DS_NAMES[$DS_IDX]}"
CONFIG_DIR="${DS_CONFIG_DIRS[$DS_IDX]}"
WELL_PATH="${DS_WELL_PATHS[$DS_IDX]}"

EVAL_BASE="$EXPBASE/forget_eval_results"
EVAL_NAME="forget_ckpt_${EXP_LABEL}_${CKPT_SUBDIR}_${DS_NAME}"
FOLDER="$EVAL_BASE/${EXP_LABEL}/${CKPT_SUBDIR}/${DS_NAME}/finetune/0"

if [ "$DS_IDX" -eq 0 ]; then
    ds_extra_args=(
        "~data.module_parameters.well_dataset_info.MHD_64"
        "+data.module_parameters.well_dataset_info.MHD_64={include_filters:[],exclude_filters:[],step_downsample_factor:0.5}"
    )
elif [ "$DS_IDX" -eq 2 ]; then
    ds_extra_args=(
        "~data.module_parameters.well_dataset_info.active_matter"
        "+data.module_parameters.well_dataset_info.turbulence_gravity_cooling={include_filters:[],exclude_filters:[],step_downsample_factor:0.5,field_transforms:{density:torch.log10,temperature:torch.log10}}"
    )
else
    ds_extra_args=()
fi

echo "=== forget_eval_checkpoints task ${SLURM_ARRAY_TASK_ID}: ${EXP_LABEL}/${CKPT_SUBDIR} on ${DS_NAME} ==="
echo "CKPT: $CKPT_PATH"

# Verify checkpoint exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
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
    ++logger.wandb_project_name="walrus_forgetting_ckpt" \
    "++name=${EVAL_NAME}" \
    data_workers=4 \
    data.module_parameters.batch_size=1 \
    "++trainer.revin._target_=walrus.trainer.normalization_strat.SamplewiseRevNormalization" \
    ++data.module_parameters.start_rollout_valid_output_at_t=-1 \
    "++data.well_base_path=${WELL_PATH}" \
    ++trainer.debug_mode=True \
    ++finetuning_mods.learnable_rope=True \
    ++finetuning_mods.rope_per_axis=True \
    "${ds_extra_args[@]}"
