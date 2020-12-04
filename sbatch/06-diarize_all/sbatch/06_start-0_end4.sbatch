#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-0_end4_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/4-01\ Margaret\ Atwood\ and\ Alden\ Nowlan\ I006-11-036.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/4-01\ Margaret\ Atwood\ and\ Alden\ Nowlan\ I006-11-036.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ John\ Logan\ \ I006-11-163.2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ John\ Logan\ \ I006-11-163.2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ David\ McFadden\ and\ Gerry\ Gilbert.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ David\ McFadden\ and\ Gerry\ Gilbert.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ David\ McFadden\ and\ Gerry\ Gilbert\ 2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ David\ McFadden\ and\ Gerry\ Gilbert\ 2.pkl
