#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-28_end32_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/12-01\ Michael\ McClure\ I006-11-160.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/12-01\ Michael\ McClure\ I006-11-160.1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/1-01\ Allen\ Ginsberg\ 3.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/1-01\ Allen\ Ginsberg\ 3.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Charles\ Reznikoff\ I006-11-153.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Charles\ Reznikoff\ I006-11-153.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Robert\ Kelly.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Robert\ Kelly.pkl
