#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-4_end8_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Robin\ Blaser\ I086-11-005.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Robin\ Blaser\ I086-11-005.1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Muriel\ Rukeyser.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Muriel\ Rukeyser.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/7-01\ Margaret\ Avison\ \ 1086-11-002.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/7-01\ Margaret\ Avison\ \ 1086-11-002.1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/1-01\ Anthony\ Hecht\ I006-11-041.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/1-01\ Anthony\ Hecht\ I006-11-041.pkl
