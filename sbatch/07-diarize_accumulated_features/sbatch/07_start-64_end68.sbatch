#!/bin/bash

#BATCH --job-name=07
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/07_start-64_end68_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Robin\ Blaser\ I086-11-005.2_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Robin\ Blaser\ I086-11-005.2_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ John\ Weiners_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ John\ Weiners_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ Henry\ Beissel_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ Henry\ Beissel_enhanced.pkl
python speakerDiarization_longfiles_ver2.py --save_wavpath=/scratch/hh2263/Spokenweb_data/enhanced/01\ bp\ Nichol\ \&\ Lionel\ kearns\ I086-11-026.1_enhanced.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/enhanced/accumulated/01\ bp\ Nichol\ \&\ Lionel\ kearns\ I086-11-026.1_enhanced.pkl
