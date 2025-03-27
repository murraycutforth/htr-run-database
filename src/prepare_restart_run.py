import os
import shutil
import subprocess
import sys
import re
import json
from glob import glob

REF_CONFIG = "GG-combustor.json"


def solution_dir(num_iterations: int) -> str:
    return f"solution_{num_iterations:10d}"


def get_path_to_restart_dir(num_iterations: int) -> str:
    return f"solution_{num_iterations:10d}/fluid_iter{num_iterations:10d}"


def update_json_data(config: dict, num_iterations: int) -> None:
    config['Flow']['initCase']['restartDir'] = get_path_to_restart_dir(num_iterations)


def return_slurm_file():
    slurm_file = glob('slurm*')

    if len(slurm_file) > 1:
        assert (0)

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











    # === reference code below, to be deleted ===


    # Ensure there are 2 or 3 command line arguments
    if len(sys.argv) == 3:
        jobid_file = sys.argv[1]
        sampledir = sys.argv[2]
        if "pp-sample" in sampledir:  # For postprocessed hdf files
            solutiondir = "postprocess"
        else:  # For solution files
            solutiondir = "solution"
    elif len(sys.argv) == 4:
        jobid_file = sys.argv[1]
        sampledir = sys.argv[2]
        solutiondir = sys.argv[3]
    else:
        print("Two or more arguments required: (1) job ID number or file, (2) sample directory, (3) solution directory")
        return

    # Extract job ID from jobid_file
    jobid = os.path.basename(jobid_file).split('.')[0]  # Remove directory and extension

    # Check if jobid is a number. If not, abort.
    if not re.match(r'^[0-9]+$', jobid):
        print(f"**ERROR** jobid={jobid} is invalid. Aborting.")
        return

    # Make target directory if it doesn't exist already
    if not os.path.exists(solutiondir):
        os.makedirs(solutiondir)

    # Find common files between the directories
    sample_files = sorted(os.listdir(sampledir))
    solution_files = sorted(os.listdir(solutiondir))

    common_files = set(sample_files) & set(solution_files)

    # Issue warning if there are common files
    if common_files:
        print("=========== WARNING ===========")
        print("Common files exist between {} and {}:".format(sampledir, solutiondir))
        print()
        for file in common_files:
            print(file)
        print()
        print("===============================")

        # Overwrite data if requested (auto proceed set to true)
        proceed = True
        if proceed:
            for file in common_files:
                os.remove(os.path.join(solutiondir, file))
                shutil.move(os.path.join(sampledir, file), os.path.join(solutiondir, file))

    # Move non-common fluid_iter solution directories
    for file in sorted(os.listdir(sampledir)):
        if file.startswith("fluid_iter"):
            shutil.move(os.path.join(sampledir, file), solutiondir)

    # Move probe files, if any
    probe_files = [f for f in os.listdir(sampledir) if f.startswith("probe") and f.endswith(".csv")]
    for f in probe_files:
        new_file_name = f.replace(".csv", f"-{jobid}.csv")
        shutil.move(os.path.join(sampledir, f), os.path.join(solutiondir, new_file_name))

    # Move console and output files to target directory
    shutil.move(os.path.join(sampledir, "console.txt"), os.path.join(solutiondir, f"console-{jobid}.txt"))
    shutil.move(jobid_file, solutiondir)

    # Move grid files
    for file in os.listdir(sampledir):
        if file.endswith("_grid"):
            shutil.move(os.path.join(sampledir, file), solutiondir)

    # Move to autodelete
    if os.listdir(sampledir):
        print(f"**WARNING** {sampledir} is not empty:")
        print(os.listdir(sampledir))
    else:
        shutil.rmtree(sampledir)

    print("Done.")

if __name__ == "__main__":
    main()