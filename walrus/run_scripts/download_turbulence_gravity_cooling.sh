#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=download_turbulence_gravity_cooling
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_turbulence_gravity_cooling_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_turbulence_gravity_cooling_%j.err

WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export HF_HUB_DISABLE_XET=1
export PATH=${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH

for SPLIT in train valid test; do
    echo "Downloading turbulence_gravity_cooling $SPLIT ..."
    the-well-download --base-path ${WALRUS_PAPER_ROOT}/the_well/datasets --dataset turbulence_gravity_cooling --split $SPLIT
done
echo "Done: turbulence_gravity_cooling"
