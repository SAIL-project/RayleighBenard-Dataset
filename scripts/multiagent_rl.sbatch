#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=/homes/tmarkmann/RayleighBenard-Dataset/slurm.log


# source conda env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcmodel
cd ~/RayleighBenard-Dataset

# start job
srun python rbcdata/multiagent_rl.py
