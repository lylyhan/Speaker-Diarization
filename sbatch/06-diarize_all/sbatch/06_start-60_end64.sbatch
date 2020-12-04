#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-60_end64_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Kenneth\ Koch\ I006-11-039.21.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Kenneth\ Koch\ I006-11-039.21.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Jackson\ MacLow\ 3.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Jackson\ MacLow\ 3.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Chris\ Levenson\ I006-11-104.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Chris\ Levenson\ I006-11-104.1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Kenneth\ Koch\ I006-11-039.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Kenneth\ Koch\ I006-11-039.1.pkl
