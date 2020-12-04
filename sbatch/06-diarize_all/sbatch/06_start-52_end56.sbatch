#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-52_end56_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ D.G.\ Jones\ and\ Eli\ Mandel.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ D.G.\ Jones\ and\ Eli\ Mandel.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Robert\ Creeley\ 1006-11-089.2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Robert\ Creeley\ 1006-11-089.2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ George\ Bowering\ I006-11-034.2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ George\ Bowering\ I006-11-034.2.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Earle\ Birney\ I006-11-018.2.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Earle\ Birney\ I006-11-018.2.pkl
