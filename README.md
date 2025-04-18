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
   python src/prepare_run_batch.py <path_to_database> <base_dir> <grid>
   ```

### Starting Simulation Runs

Use the `set_off_runs.sh` script to start the simulation runs in the background.
```bash
./scripts/set_off_runs.sh <start_run_id> <end_run_id> <max iterations>
```

## Requirements

- Python 3.x
- Bash
- Required Python packages: `seaborn`, `matplotlib`, `pandas`, `numpy`


## Reference scales used in simulation

- pRef = 101325.0 [Pa]
- Tref = 300.00 [K]
- tRef = 1.1375e-5 [s]
- Lref = 0.003175 [m]
- uRef = 2.792e+02 [ms^1]
- rhoRef = 1.2998553074969466 [kgm^3]
- eRef   = 77950.983786892 [J/m^3]


## Run batches

1. 15M grid, matching experimental locations
2. N/A
3. 5M grid, matching experimental locations
4. 2M grid, matching experimental locations
5. 2M grid, sweeping over all z values (part 1)
6. 2M grid, sweeping over all z values (part 2)
7. 2M grid, 2 x aleatoric repeats of transition zone, plus fill in z=6,13,19
