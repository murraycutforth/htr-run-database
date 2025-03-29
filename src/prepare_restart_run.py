import os
import shutil
import subprocess
import sys
import re
import json
from glob import glob
from pathlib import Path

REF_CONFIG = "GG-combustor.json"


def get_path_to_restart_dir(num_iterations: int) -> str:
    current_dir_full = os.path.abspath(os.getcwd())
    return os.path.join(current_dir_full, f'solution/fluid_iter{num_iterations:010d}')


def update_json_data(config: dict, num_iterations: int) -> None:
    config['Flow']['initCase']['restartDir'] = get_path_to_restart_dir(num_iterations)


def return_slurm_file():
    # Find slurm or LSF .out files
    slurm_file = glob('*.out')

    if not slurm_file:
        raise FileNotFoundError("No slurm file found")
    elif len(slurm_file) > 1:
        raise RuntimeError("Multiple slurm files found")

    return slurm_file


def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python prepare_restart_run.py <number of iterations in restart file>")

    # First rename the sample0 directory to the new solution directory

    num_iterations = int(sys.argv[1])
    assert num_iterations >= 0, "Number of iterations must be non-negative"
    assert num_iterations <= 100000, "Number of iterations expected to be less than 100,000"

    if not os.path.exists("sample0"):
        raise FileNotFoundError("sample0 directory not found")

    slurm_file = return_slurm_file()
    bash_command = ['bash', 'organize-htr.sh', slurm_file[0], 'sample0', 'solution']
    process = subprocess.Popen(bash_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Command {' '.join(bash_command)} failed with error: {error.decode().strip()}")

    # Now update the GG-combustor.json file ready for the next run

    with open(REF_CONFIG, 'r') as f:
        config = json.load(f)

    update_json_data(config, num_iterations)

    with open(REF_CONFIG, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    main()