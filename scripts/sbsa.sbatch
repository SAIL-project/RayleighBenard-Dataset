#!/bin/bash

#SBATCH --job-name=PPO
#SBATCH --partition=study
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/RayleighBenard-Dataset

srun python rbcdata/sbsa.py
