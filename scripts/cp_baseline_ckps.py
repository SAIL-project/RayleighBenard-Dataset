import os
import shutil

import yaml

PATH = "."
OUTPUT = "./data"
dirs = [f.path for f in os.scandir(PATH) if f.is_dir()]

for dir in dirs:
    # Load config
    with open("{dir}/.hydra/config.yml", "r") as file:
        cfg = yaml.safe_load(file)
        ra = cfg["sim"]["ra"]
        seed = cfg["sim"]["seed"]
    # Copy file
    src = f"{dir}/shenfun/checkpoint/RB_2D.chk.h5"
    dest = f"{OUTPUT}/checkpoints/ra{ra}/train/baseline{seed}.chk.h5"
    if os.path.isfile(dest):
        print(f"File '{dest}' already exists! exiting...")
        exit(1)
    shutil.copyfileobj(src, dest)
