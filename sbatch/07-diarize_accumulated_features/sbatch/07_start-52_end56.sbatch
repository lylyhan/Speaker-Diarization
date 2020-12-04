#!/bin/bash

#BATCH --job-name=07
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/07_start-52_end56_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ James\ Wright\ I006-11-157_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ James\ Wright\ I006-11-157_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ George\ Bowering\ I006-11-034.1_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ George\ Bowering\ I006-11-034.1_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Michael\ McClure\ I006-11-160.2_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Michael\ McClure\ I006-11-160.2_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/1-01\ Anthony\ Hecht\ I006-11-041_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/1-01\ Anthony\ Hecht\ I006-11-041_enhanced.pkl
