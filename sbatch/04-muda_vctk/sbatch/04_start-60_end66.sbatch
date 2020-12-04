#!/bin/bash

#BATCH --job-name=04
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --mem=64GB
#SBATCH --output=../slurm/04_start-60_end66_%j.out

module purge

# The first and second argument is the path to dataset and impulse response files.
# The second and third argument is the range of speaker folders to augment for this job .
python /home/hh2263/Speaker-Diarization/augmentation_ver3.py /scratch/hh2263/VCTK/VCTK-Corpus/wav48 /home/hh2263/Speaker-Diarization/ir_files/ 60 66 /home/hh2263/Speaker-Diarization/bg_files/