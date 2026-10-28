#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --job-name=convert_cylinder
#SBATCH --output=/cephfs/store/gr-mc2473/ham51/realpdebench/logs/convert_cylinder_%j.out

python /cephfs/store/gr-mc2473/ham51/realpdebench/scripts/convert_realpde.py