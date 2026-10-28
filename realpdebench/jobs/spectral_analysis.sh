#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=cylinder_spectral
#SBATCH --exclude=sw-gen05
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/cylinder_spectral_%j.out

/cephfs/store/gr-mc2473/ham51/Conda/envs/realpdebench/bin/python /cephfs/store/gr-mc2473/ham51/realpdebench/scripts/cylinder_spectral.py
