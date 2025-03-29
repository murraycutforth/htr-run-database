# htr-run-database

This project contains scripts to automate the setup and execution of simulation runs for a laser-ignited rocket combustor model with the HTR-solver codebase. 

## Project Structure

- `src/`: Contains the main Python scripts for preparing and managing simulation runs.
  - `main_lf_run_database.py`: Script to create the run database and generate pair plots.
  - `prepare_run_batch.py`: Script to set up a batch of runs based on a reference configuration and a database of parameters.
- `scripts/`: Contains shell scripts for organizing and executing simulation runs.
  - `set_off_runs.sh`: Script to start multiple simulation runs in the background.
  - `organize-htr.sh`: Script to organize output files from simulation runs.
- `GG-combustor-default.json`: Reference configuration file for the simulations.

## Usage

### Setting Up a Batch of Runs

1. **Prepare the Run Database:**
   Run the `main_lf_run_database.py` script to create the run database and generate pair plots.
   ```bash
   python src/main_lf_run_database.py
   ```

2. **Set Up the Batch:**
   Use the `prepare_run_batch.py` script to set up a batch of runs.
   ```bash
   python src/prepare_run_batch.py <path_to_database> <base_dir>
   ```

### Starting Simulation Runs

Use the `set_off_runs.sh` script to start the simulation runs in the background.
```bash
./scripts/set_off_runs.sh <start_run_id> <end_run_id>
```

## Requirements

- Python 3.x
- Bash
- Required Python packages: `seaborn`, `matplotlib`, `pandas`, `numpy`
