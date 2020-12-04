#!/bin/bash

#SBATCH --job-name=train_vctk
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem=62GB
#SBATCH --gres=gpu:2
#SBATCH --output=/home/hh2263/Speaker-Diarization/03-slurm-train_vctk.out


module purge

module load cuda/9.0.176
module load cudnn/9.0v7.3.0.29
module load ffmpeg/intel/3.2.2

python /home/hh2263/Speaker-Diarization/train.py   


