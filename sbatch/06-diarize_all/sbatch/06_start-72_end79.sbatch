#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-72_end79_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Al\ Purdy\ Vancouver\ I086-11-042.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Al\ Purdy\ Vancouver\ I086-11-042.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Robin\ Blaser\ I086-11-005.2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Robin\ Blaser\ I086-11-005.2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/1-01\ David\ Ball\ and\ Tom\ Raworth\ 2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/1-01\ David\ Ball\ and\ Tom\ Raworth\ 2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/1-01\ Jackson\ MacLow\ 2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/1-01\ Jackson\ MacLow\ 2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/2-01\ Irving\ Layton\ 1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/2-01\ Irving\ Layton\ 1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Mike\ Gnarowski.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Mike\ Gnarowski.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Ron\ Loewinsohn.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Ron\ Loewinsohn.pkl
