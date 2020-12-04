#!/bin/bash

#SBATCH --job-name=04-muda_vctk
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=24GB
#SBATCH --output=/home/hh2263/Speaker-Diarization/04-muda_vctk.out


module purge 

python /home/hh2263/Speaker-Diarization/augmentation_ver2.py
