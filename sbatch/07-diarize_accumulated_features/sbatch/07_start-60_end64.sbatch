#!/bin/bash

#BATCH --job-name=07
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/07_start-60_end64_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Jerome\ Rothenberg\ I006-11-095.2_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Jerome\ Rothenberg\ I006-11-095.2_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Robert\ Creeley_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Robert\ Creeley_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Kenneth\ Koch\ I006-11-039.21_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Kenneth\ Koch\ I006-11-039.21_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/1-01\ Allen\ Ginsberg\ 3_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/1-01\ Allen\ Ginsberg\ 3_enhanced.pkl
