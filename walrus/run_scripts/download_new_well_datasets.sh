#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=download_well
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_well_%A_%a.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_well_%A_%a.err
#SBATCH --array=0-5

# Download test split only for new Well datasets, then symlink test files into
# train/ and valid/ directories so the walrus dataloader sees a complete split structure
# without downloading the full dataset.
#
# Datasets:
#   0  acoustic_scattering_discontinuous
#   1  acoustic_scattering_inclusions
#   2  acoustic_scattering_maze
#   3  planetswe
#   4  helmholtz_staircase
#   5  shear_flow

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export HF_HUB_DISABLE_XET=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

BASE_PATH="${WALRUS_PAPER_ROOT}/the_well/datasets"

DATASETS=(
    "acoustic_scattering_discontinuous"
    "acoustic_scattering_inclusions"
    "acoustic_scattering_maze"
    "planetswe"
    "helmholtz_staircase"
    "shear_flow"
)

DS="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
echo "=== Downloading ${DS} (test only) ==="

# Download test split only
the-well-download --base-path "$BASE_PATH" --dataset "$DS" --split test --parallel

DATA_DIR="$BASE_PATH/datasets/$DS/data"

if [ ! -d "$DATA_DIR/test" ]; then
    echo "ERROR: test dir not found after download: $DATA_DIR/test"
    exit 1
fi

# Symlink test files into train/ and valid/ to satisfy the dataloader's split expectations
for SPLIT in train valid; do
    mkdir -p "$DATA_DIR/$SPLIT"
    for f in "$DATA_DIR/test"/*.hdf5; do
        fname=$(basename "$f")
        target="$DATA_DIR/$SPLIT/$fname"
        if [ ! -e "$target" ]; then
            ln -s "../test/$fname" "$target"
            echo "  symlinked $SPLIT/$fname → test/$fname"
        else
            echo "  already exists: $SPLIT/$fname"
        fi
    done
done

echo "Done: $DS — $(ls $DATA_DIR/test/*.hdf5 | wc -l) test files, symlinked into train/ and valid/"
