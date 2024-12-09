"""
Fetch the latest results for a specific date from the cluster runs
"""
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Inspect a trained agent in the Rayleigh-Benard environment')
parser.add_argument('--date', type=str, help='date of the experiment in M-D format')
args = parser.parse_args()



os.system('rsync -a mstraat@login-1.gpu.cit-ec.de:/homes/mstraat/Projects/RayleighBenard-Dataset/rbcdata/logs/runs/ /PATH/TO/MY/FILE ')