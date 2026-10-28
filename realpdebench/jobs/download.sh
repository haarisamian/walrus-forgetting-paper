#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --job-name=download_cylinder
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/download_cylinder_%j.out

export HF_HUB_DISABLE_XET=1

realpdebench download \
    --dataset-root /cephfs/store/gr-mc2473/ham51/realpdebench/data \
    --scenario cylinder \
    --what hf_dataset