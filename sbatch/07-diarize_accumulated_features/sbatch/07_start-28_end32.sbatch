#!/bin/bash

#BATCH --job-name=07
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/07_start-28_end32_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/1-01\ F.R.\ Scott_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/1-01\ F.R.\ Scott_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ David\ McFadden\ and\ Gerry\ Gilbert\ 2_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ David\ McFadden\ and\ Gerry\ Gilbert\ 2_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Charles\ Simic\ I006-11-115_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Charles\ Simic\ I006-11-115_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Stan\ Persky\ I006-11-137_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Stan\ Persky\ I006-11-137_enhanced.pkl
