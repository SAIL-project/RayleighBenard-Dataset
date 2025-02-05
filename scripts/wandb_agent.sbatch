#!/bin/bash

#SBATCH --job-name=PPO
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/mstraat/Projects/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/mstraat/Projects/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=FAIL


# source conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/Projects/RayleighBenard-Dataset

# start job
srun wandb agent --count 50 $1