#!/bin/bash

#SBATCH --job-name=sbsa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=23
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/RayleighBenard-Dataset

# run with config
#srun python rbcdata/sbsa.py experiment=thorben tags="['trainv2']" ra=10000,1000000
srun python rbcdata/sbsa.py experiment=thorben tags="['trainv2']" ra=100000,5000000

# eval
#srun python rbcdata/sbsa_eval.py

# run ppo with sweep params for different RA
#srun python rbcdata/sbsa.py -m tags="['train','ep_len_50']" notes="train on different ra for eval later" dt=0.05 ra=5000,10000
#srun python rbcdata/sbsa.py -m tags="['train']" notes="train on different ra for eval later" ra=50000,500000,5000000
#srun python rbcdata/sbsa.py -m tags="['train']" notes="train on different ra for eval later" ra=100000,1000000

# run ppo on single episode for multiple times
#srun python rbcdata/sbsa.py -m tags="['single_ep']" notes="run ppo on single episode for multiple times" ra=10000 seed=42,43,44,45 train_env.checkpoint="data/checkpoints/ra10000/train/baseline42.chk.h5"
