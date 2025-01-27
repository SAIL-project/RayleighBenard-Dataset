import os
import shutil

import yaml

PATH = "logs/run_multirun/01-27-02-14-18"
OUTPUT = "data"
dirs = [f.path for f in os.scandir(PATH) if f.is_dir()]
print(dirs)

for dir in dirs:
    # Load config
    with open(f"{dir}/.hydra/config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
        ra = cfg["env"]["ra"]
        seed = cfg["seed"]
    # Copy file
    src = f"{dir}/shenfun/checkpoint.chk.h5"
    dest = f"{OUTPUT}/checkpoints_test/ra{ra}/train/baseline{seed}.chk.h5"
    if os.path.isfile(dest):
        print(f"File '{dest}' already exists! exiting...")
        exit(1)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
