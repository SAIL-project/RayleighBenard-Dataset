#!/bin/bash

#SBATCH --job-name=sbsa_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00
#SBATCH --output=/homes/mstraat/Projects/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/mstraat/Projects/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/Projects/RayleighBenard-Dataset

srun python rbcdata/sbsa_eval.py