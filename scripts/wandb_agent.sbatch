#!/bin/bash

#SBATCH --job-name=PPO
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=FAIL


# source conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/RayleighBenard-Dataset

# start job
srun wandb agent --count 50 $1