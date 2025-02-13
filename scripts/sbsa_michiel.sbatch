#!/bin/bash

#SBATCH --job-name=sbsa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=22
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/homes/mstraat/Projects/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/mstraat/Projects/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/Projects/RayleighBenard-Dataset

srun python rbcdata/sbsa.py -m tags="['train']" experiment=michiel ra=$1 reward_shaping=$2
#srun python rbcdata/sbsa.py -m tags="['train']" notes="train on different ra for eval later" ra=50000,500000,5000000