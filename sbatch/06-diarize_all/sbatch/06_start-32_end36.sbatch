#!/bin/bash

#BATCH --job-name=06
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/06_start-32_end36_%j.out

module purge
module load gcc/6.3.0

cd /home/hh2263/Speaker-Diarization
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/1-01\ George\ Oppen.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/1-01\ George\ Oppen.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Roy\ Kiyooka.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Roy\ Kiyooka.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ bp\ Nichol\ \&\ Lionel\ kearns\ I086-11-026.1.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ bp\ Nichol\ \&\ Lionel\ kearns\ I086-11-026.1.pkl
python speakerDiarization_longfiles.py --save_wavpath=/scratch/hh2263/Spokenweb_data/resampled/01\ Victor\ Coleman\ Reading.wav --save_pklpath=/scratch/hh2263/Spokenweb_data/resampled/stream_3min_score/01\ Victor\ Coleman\ Reading.pkl
