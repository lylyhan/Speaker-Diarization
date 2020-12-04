#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-56_end60_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Barbara\ Howes\ I086-11-024.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Barbara\ Howes\ I086-11-024.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Eli\ Mandel.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Eli\ Mandel.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Phyllis\ Webb.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Phyllis\ Webb.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Bisset,\ Bill.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Bisset,\ Bill.pkl
