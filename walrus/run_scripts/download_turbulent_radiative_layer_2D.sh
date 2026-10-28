#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=download_turbulent_radiative_layer_2D
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_turbulent_radiative_layer_2D_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_turbulent_radiative_layer_2D_%j.err

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export HF_HUB_DISABLE_XET=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

for SPLIT in train valid test; do
    echo "Downloading turbulent_radiative_layer_2D $SPLIT ..."
    the-well-download --base-path ${WALRUS_PAPER_ROOT}/the_well/datasets --dataset turbulent_radiative_layer_2D --split $SPLIT
done
echo "Done: turbulent_radiative_layer_2D"
