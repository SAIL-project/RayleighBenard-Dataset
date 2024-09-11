#!/bin/bash

#SBATCH --job-name=PPOra104
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:0
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --mail-type=BEGIN,FAIL,END

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/Projects/RayleighBenard-Dataset/rbcdata

echo "hello world"
# srun python sa.py