#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-24_end28_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Jerome\ Rothenberg\ I006-11-095.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Jerome\ Rothenberg\ I006-11-095.1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Al\ Purdy-\ Of\ All\ People\ I006-11-037.2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Al\ Purdy-\ Of\ All\ People\ I006-11-037.2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Gary\ Snyder\ I006-11-106=AC2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Gary\ Snyder\ I006-11-106=AC2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Henry\ Beissel.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Henry\ Beissel.pkl
