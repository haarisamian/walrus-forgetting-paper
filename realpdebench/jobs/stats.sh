#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=cylinder_stats
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/stats_%j.out

/cephfs/store/gr-mc2473/ham51/Conda/envs/realpdebench/bin/python \
    /cephfs/store/gr-mc2473/ham51/walrus/scripts/get_one_stats.py \
    /cephfs/store/gr-mc2473/ham51/realpdebench/well/cylinder_numerical \
    stats.yaml
