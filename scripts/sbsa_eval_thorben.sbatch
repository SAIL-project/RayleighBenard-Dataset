#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.out
#SBATCH --error=/homes/tmarkmann/RayleighBenard-Dataset/slurmout/%x_%j.err
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcdata
cd ~/RayleighBenard-Dataset

# evaluate on different RA using checkpoints
#srun python rbcdata/sbsa_eval.py tags="['evalv3']" ra=5000000

# evaluate generalization
#srun python rbcdata/sbsa_eval.py -m tags="['gen_ra10000']" ra=10000 env.ra=10000,100000,1000000,5000000 env.checkpoint="data/checkpoints/ra10000/test/"
srun python rbcdata/sbsa_eval.py -m tags="['gen_ra100000']" ra=100000 env.ra=10000,100000,1000000,5000000 env.checkpoint="data/checkpoints/ra100000/test/"

# pd and baseline
#srun python rbcdata/run_pd.py -m baseline=false env.ra=10000,100000,1000000,5000000
