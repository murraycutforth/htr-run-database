"""
We take reference GG-combustor.json file and modify it for each run in the batch, put them into a new directory
for each one, and record the mapping from run_id to directory in a csv file.
"""

import argparse
import json
import shutil
from pathlib import Path
import csv
import os
import stat

# Path to reference config file
REF_CONFIG = 'GG-combustor-default.json'

# Path to CommonCase BC and grid files (currently hardcoded for me on dane.llnl.gov)
# COMMON_CASE_DIR = '/p/lustre1/cutforth1/PSAAP/'

# Path to dir where runs are executed and output written (again, hardcoded for me on dane.llnl.gov)
# RUN_DIR = '/p/lustre1/cutforth1/PSAAP/'

# Reference length scale [m]
LREF = 0.003175


def get_path_to_common_case(xi: list, base_dir: Path) -> Path:
    common_x_locs = [6.0, 7.0, 8.0, 9.0, 10.0]
    common_z_locs = [6.0, 13.0, 19.0]

    # Convert
    xi_x = xi[0]  # Radial
    xi_z = xi[2]  # Streamwise

    # Find closest common x and z locations
    x_idx = min(range(len(common_x_locs)), key=lambda i: abs(common_x_locs[i] - xi_x))
    z_idx = min(range(len(common_z_locs)), key=lambda i: abs(common_z_locs[i] - xi_z))

    x_str = f'{int(common_x_locs[x_idx]):02d}'
    z_str = f'{int(common_z_locs[z_idx]):02d}'
    path = base_dir / f'location_{x_str}_0_{z_str}' / 'CommonCase' / xi[16]
    assert path.exists(), f"Common case directory {path} does not exist"
    return path


def update_json_data(config: dict, xi: list, base_dir: Path, wall_time: int) -> None:
    """Edit the given JSON data with the sampled parameters."""
    common_case_dir = get_path_to_common_case(xi, base_dir)

    # Update laser focal location
    # xi is given in mm, so multiply by 0.001 and divide by LREF to get dimensionless units
    # In simulation, coords are (streamwise, radial, azimuthal)
    # When sampling xi, coords are (radial, azimuthal, streamwise)
    assert -1.0 <= xi[0] <= 13.0, f"xi[0] out of bounds: {xi[0]}"
    assert -1.0 <= xi[1] <= 1.0, f"xi[1] out of bounds: {xi[1]}"
    assert 5.0 <= xi[2] <= 21.0, f"xi[2] out of bounds: {xi[2]}"
    config['Flow']['laser']['focalLocation'][0] = xi[2] * 0.001 / LREF
    config['Flow']['laser']['focalLocation'][1] = xi[0] * 0.001 / LREF
    config['Flow']['laser']['focalLocation'][2] = xi[1] * 0.001 / LREF

    # Update axial length and radii
    axial_l_i = xi[3]
    near_radius = axial_l_i / (2 * xi[4])
    config['Flow']['laser']['axialLength'] = axial_l_i
    config['Flow']['laser']['nearRadius'] = near_radius
    config['Flow']['laser']['farRadius'] = near_radius / xi[5]

    # Update energy deposited
    config['Flow']['laser']['peakEdotPerMass'] = xi[6]

    # Update FWHM
    config['Flow']['laser']['pulseFWHM'] = xi[7]

    # Update times
    a = 1000 + xi[14]
    b = a + 2000
    c = b + 2000
    d = c + 2000
    config['Integrator']['TimeStep']['zone1'] = a
    config['Integrator']['TimeStep']['zone2'] = b
    config['Integrator']['TimeStep']['zone3'] = c
    config['Integrator']['TimeStep']['zone4'] = d
    config['Flow']['laser']['pulseTime'] += xi[14] * 0.003
    config['Integrator']['TimeStep']['time5'] = 0.002

    # Update methane content
    config['Flow']['initCase']['restartDir'] = str(common_case_dir / 'solution' / f'{xi[15]}')

    # Update thickened flame parameters
    config['Flow']['TFModel']['Efficiency']['beta'] = xi[8]
    config['Flow']['TFModel']['Efficiency']['sL0'] = xi[9]
    config['Flow']['TFModel']['Efficiency']['Arr_factor'] = xi[10]

    # Update Smagorinsky constant
    config['Flow']['sgsModel']['TurbViscModel']['C_S'] = xi[11]

    # Update mass flow rates
    config['BC']['xBCLeft']['mDot1'] = xi[12]
    config['BC']['xBCLeft']['mDot2'] = xi[13]

    # Update file directories
    config['BC']['xBCLeft']['MixtureProfile']['FileDir'] = str(common_case_dir / 'bc-6sp')
    config['BC']['xBCLeft']['TemperatureProfile']['FileDir'] = str(common_case_dir / 'bc-6sp')
    config['BC']['xBCLeft']['VelocityProfile']['FileDir'] = str(common_case_dir / 'bc-6sp')
    config['Grid']['GridInput']['gridDir'] = str(common_case_dir / 'bc-6sp' / 'grid')

    # Update wall time
    config['Mapping']['wallTime'] = wall_time



def read_csv_to_list_of_lists(file_path, num_header_rows=1):
    # Utility function
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for _ in range(num_header_rows):
            next(reader)  # Skip the header rows
        data = [row for row in reader]

    print(f'Loaded {len(data)} rows from {file_path}')
    return data


def load_xi_from_database_row(row: list) -> list:
    return [
        float(row[2]),
        float(row[3]),
        float(row[4]),
        float(row[5]),
        float(row[6]),
        float(row[7]),
        float(row[8]),
        float(row[9]),
        float(row[10]),
        float(row[11]),
        float(row[12]),
        float(row[13]),
        float(row[14]),
        float(row[15]),
        int(row[16]),
        str(row[17]),
        str(row[18]),
    ]


def get_batch_ids(rows: list) -> list:
    return [int(row[1]) for row in rows]


def parse_args():
    parser = argparse.ArgumentParser(description='Set up a batch of runs')
    parser.add_argument('database', type=str, help='Path to the database file')
    parser.add_argument('base_dir', type=str, help='Base directory for the runs, e.g. /p/lustre1/cutforth1/PSAAP/ ')
    parser.add_argument('use_cuda', type=int, default=0, help='Use CUDA or not')
    parser.add_argument('wall_time', type=int, default=1430, help='Wall time for each run')
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    database_path = Path(args.database)
    config_path = Path(REF_CONFIG)
    use_cuda = int(args.use_cuda)
    wall_time = int(args.wall_time)

    assert base_dir.exists(), f"Base directory {base_dir} does not exist"
    assert database_path.exists(), f"Database file {database_path} does not exist"
    assert config_path.exists(), f"Reference config file {config_path} does not exist"
    assert use_cuda in [0, 1], f"Invalid use_cuda value: {use_cuda}"
    assert wall_time > 0, f"Invalid wall time: {wall_time}"

    run_database = read_csv_to_list_of_lists(database_path)
    batch_ids = get_batch_ids(run_database)

    assert len(set(batch_ids)) == 1, f"Multiple batch IDs found: {set(batch_ids)}"
    batch_id = batch_ids[0]

    outdir_base = base_dir / f'runs_batch_{batch_id:04d}'
    assert not outdir_base.exists(), f"Output directory {outdir_base} already exists"
    outdir_base.mkdir()

    # Load the reference config
    with open(REF_CONFIG, 'r') as f:
        ref_config = json.load(f)

    for row in run_database:
        run_id = int(row[0])
        xi = load_xi_from_database_row(row)
        assert len(xi) == 17, f"Expected 17 parameters, got {len(xi)}"

        run_dir = outdir_base / f'{run_id:04d}'
        run_dir.mkdir()

        config = ref_config.copy()
        update_json_data(config, xi, base_dir, wall_time)

        # Write the config to a new file
        with open(run_dir / 'GG-combustor.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Write the run-htr.sh script
        with open(run_dir / 'run-htr.sh', 'w') as f:
            f.write('outidr="."')
            f.write(f'USE_CUDA={use_cuda} DEBUG=0 PROFILE=0 QUEUE="pbatch" $HTR_DIR/prometeo.sh -i GG-combustor.json -o "."')

        # Add in copies of common scripts we will use
        shutil.copy('scripts/run-htr-with-restarts_slurm.sh', run_dir)
        shutil.copy('scripts/run-htr-with-restarts_lsf.sh', run_dir)
        shutil.copy('src/prepare_restart_run.py', run_dir)
        shutil.copy('scripts/organize-htr.sh', run_dir)

        # Make executable

        st = os.stat(run_dir / 'run-htr.sh')
        os.chmod(run_dir / 'run-htr.sh', st.st_mode | stat.S_IEXEC)

        st = os.stat(run_dir / 'run-htr-with-restarts_slurm.sh')
        os.chmod(run_dir / 'run-htr-with-restarts_slurm.sh', st.st_mode | stat.S_IEXEC)

        st = os.stat(run_dir / 'run-htr-with-restarts_lsf.sh')
        os.chmod(run_dir / 'run-htr-with-restarts_lsf.sh', st.st_mode | stat.S_IEXEC)

        st = os.stat(run_dir / 'organize-htr.sh')
        os.chmod(run_dir / 'organize-htr.sh', st.st_mode | stat.S_IEXEC)

    shutil.copy('scripts/set_off_runs.sh', outdir_base)
    st = os.stat(outdir_base / 'set_off_runs.sh')
    os.chmod(outdir_base / 'set_off_runs.sh', st.st_mode | stat.S_IEXEC)

if __name__ == "__main__":
    main()
