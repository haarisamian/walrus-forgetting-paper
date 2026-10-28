#!/bin/bash
#SBATCH --partition=spr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --job-name=persist_full
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/persist_full_%j.out
#SBATCH --error=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/persist_full_%j.err
WALRUS_PAPER_ROOT="${WALRUS_PAPER_ROOT:-/cephfs/store/gr-mc2473/ham51}"   # paper was run with this root
export HDF5_USE_FILE_LOCKING=FALSE
export PATH=/usr/bin:${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin:$PATH
export PYTHONUNBUFFERED=1
srun ${WALRUS_PAPER_ROOT}/Conda/envs/realpdebench/bin/python \
  ${WALRUS_PAPER_ROOT}/realpdebench/scripts/persistence_fullsplit.py
