#!/bin/bash

#SBATCH --job-name=03-embeddings_muda
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=62GB
#SBATCH --output=./02-generate_embeddings.out

module purge


python /home/hh2263/Speaker-Diarization/ghostvlad/generate_embeddings.py
