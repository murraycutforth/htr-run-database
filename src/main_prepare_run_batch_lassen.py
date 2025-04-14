"""
We take reference GG-combustor.json file and modify it for each run in the batch, put them into a new directory
for each one, and record the mapping from run_id to directory in a csv file. Expected to be run on lassen.
"""

import argparse
import json
import shutil
from distutils.command.config import config
from pathlib import Path
import csv
import os
import stat

# Path to reference config file
#REF_CONFIG = 'configs/GG-combustor-default-lassen.json'

# Path to CommonCase BC and grid files (currently hardcoded for me on dane.llnl.gov)
# COMMON_CASE_DIR = '/p/lustre1/cutforth1/PSAAP/'

# Path to dir where runs are executed and output written (again, hardcoded for me on dane.llnl.gov)
# RUN_DIR = '/p/lustre1/cutforth1/PSAAP/'

# Reference length scale [m]
LREF = 0.003175


def get_path_to_common_case(xi: list, base_dir: Path, grid_size: str) -> Path:
    if grid_size == '15M':
        common_x_locs = [6.0, 7.0, 8.0, 9.0, 10.0]
        common_z_locs = [6.0, 13.0, 19.0]
    elif grid_size == '5M':
        common_x_locs = [6.0, 7.0, 8.0, 9.0, 10.0]
        common_z_locs = [6.0, 13.0, 19.0]
    elif grid_size == '2M':
        common_x_locs = [6.0, 7.0, 8.0, 9.0, 10.0]
        common_z_locs = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]  # TODO: extend to 24.0 for batch 6 in future
    else:
        raise ValueError(f"Unknown grid size {grid_size}")


    # Convert
    xi_x = xi[0]  # Radial
    xi_z = xi[2]  # Streamwise

    # Find closest common x and z locations
    x_idx = min(range(len(common_x_locs)), key=lambda i: abs(common_x_locs[i] - xi_x))
    z_idx = min(range(len(common_z_locs)), key=lambda i: abs(common_z_locs[i] - xi_z))

    x_str = f'{int(common_x_locs[x_idx]):02d}'
    z_str = f'{int(common_z_locs[z_idx]):02d}'

    if grid_size == '15M':
        path = base_dir / f'location_{x_str}_0_{z_str}' / 'CommonCase' / xi[16]
    elif grid_size == '5M':
        path = base_dir / '04_GMRC_5M' / f'location_{x_str}_0_{z_str}' / 'CommonCase' / xi[16]
    elif grid_size == '2M':
        path = base_dir / '04_GMRC_2M' / f'location_{x_str}_0_{z_str}' / 'CommonCase' / xi[16]
    else:
        raise ValueError(f"Unknown grid size {grid_size}")

    assert path.exists(), f"Common case directory {path} does not exist"
    return path


def update_json_data(config: dict, xi: list, base_dir: Path, grid_size: str) -> None:
    """Edit the given JSON data with the sampled parameters."""
    common_case_dir = get_path_to_common_case(xi, base_dir, grid_size)

    # Update laser focal location
    # xi is given in mm, so multiply by 0.001 and divide by LREF to get dimensionless units
    # In simulation, coords are (streamwise, radial, azimuthal)
    # When sampling xi, coords are (radial, azimuthal, streamwise)
    assert -1.0 <= xi[0] <= 13.0, f"xi[0] out of bounds: {xi[0]}"
    assert -1.0 <= xi[1] <= 1.0, f"xi[1] out of bounds: {xi[1]}"
    assert 2.0 <= xi[2] <= 25.0, f"xi[2] out of bounds: {xi[2]}"
    config['Flow']['laser']['focalLocation'][0] = xi[2] * 0.001 / LREF  # streamwise
    config['Flow']['laser']['focalLocation'][1] = xi[0] * 0.001 / LREF  # radial
    config['Flow']['laser']['focalLocation'][2] = xi[1] * 0.001 / LREF  # azimuthal

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

    # Update time steps, so the zones are aligned with the laser firing time
    # This also depends on the grid
    if grid_size == '15M':
        config['Integrator']['TimeStep']['zone1'] = 1000 + xi[14]  # xi[14] takes values in [1000, 2000]
        config['Integrator']['TimeStep']['zone2'] = 3000 + xi[14]
        config['Integrator']['TimeStep']['zone3'] = 5000 + xi[14]
        config['Integrator']['TimeStep']['zone4'] = 7000 + xi[14]
        config['Integrator']['TimeStep']['zone5'] = 9000 + xi[14]
    elif grid_size == '5M':
        config['Integrator']['TimeStep']['zone1'] = 1000 + xi[14]  # use time1 until laser
        config['Integrator']['TimeStep']['zone2'] = 2000 + xi[14]  # use time2 for 1k iterations post-laser
        config['Integrator']['TimeStep']['zone3'] = 3000 + xi[14]  # use time3 for 1k further iterations
        config['Integrator']['TimeStep']['zone4'] = 7300 + xi[14]  # use time4 for 5000 further iterations, then time5
    elif grid_size == '2M':
        config['Integrator']['TimeStep']['zone1'] = 1000 + xi[14]
        config['Integrator']['TimeStep']['zone2'] = 2000 + xi[14]
        config['Integrator']['TimeStep']['zone3'] = 3000 + xi[14]
        config['Integrator']['TimeStep']['zone4'] = 7300 + xi[14]
    else:
        raise ValueError(f"Unknown grid size {grid_size}")



    # Laser firing time - set to slightly after the zone2 time steps begin
    if grid_size == '15M':
        pulseOffset = config['Flow']['laser']['pulseFWHM'] * 5
        assert config['Integrator']['TimeStep']['time1'] == 0.003
        config['Flow']['laser']['pulseTime'] = pulseOffset + (1000 + xi[14]) * 0.003
    elif grid_size == '5M':
        pulseOffset = config['Flow']['laser']['pulseFWHM'] * 5
        assert config['Integrator']['TimeStep']['time1'] == 0.003
        config['Flow']['laser']['pulseTime'] = pulseOffset + (1000 + xi[14]) * 0.003  # Keep the pre-laser time step consistent so we are firing at same time
    elif grid_size == '2M':
        pulseOffset = config['Flow']['laser']['pulseFWHM'] * 5
        assert config['Integrator']['TimeStep']['time1'] == 0.003
        config['Flow']['laser']['pulseTime'] = pulseOffset + (1000 + xi[14]) * 0.003
    else:
        raise ValueError(f"Unknown grid size {grid_size}")

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
    if grid_size == '15M':
        config['Mapping']['wallTime'] = 720
    elif grid_size == '5M':
        config['Mapping']['wallTime'] = 420
    elif grid_size == '2M':
        config['Mapping']['wallTime'] = 200



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


def load_nominal_xz_coords_from_database_row(row: list) -> list:
    return [
        float(row[2]),  # radial [mm]
        float(row[4])  # streamwise [mm]
    ]


def set_restart_frequency(config: dict, xz_coords: list, batch_id: int) -> None:
    """We are more interested in the indirect ignition cases, so adjust output accordingly.

    At 2.2GB per snapshot, this yields either 88GB or 22GB per run.
    """
    if batch_id == 1:
        radial_dist_mm = xz_coords[0]

        if radial_dist_mm > 7.5:
            restart_every = 1000
        else:
            restart_every = 4000
    elif batch_id == 2:
        restart_every = 4000
    elif batch_id == 3:
        restart_every = 5000
    elif batch_id in [4, 5, 6, 7]:
        restart_every = 5000
    else:
        raise ValueError(f"Unknown batch ID {batch_id}")

    config["IO"]["restartEveryTimeSteps"] = restart_every


def print_out_config_info(config: dict) -> None:
    print("New config info for this run:")
    print(f' Laser focal location: {config["Flow"]["laser"]["focalLocation"]} which has radial location in mm of {config["Flow"]["laser"]["focalLocation"][1] * 1000 * LREF} [mm] ')
    print(f' Output frequency: {config["IO"]["restartEveryTimeSteps"]}')
    print(f' Laser pulse time: {config["Flow"]["laser"]["pulseTime"]}')
    time1 = config["Integrator"]["TimeStep"]["time1"]
    zone1 = config["Integrator"]["TimeStep"]["zone1"]
    print(f' time1: {time1}, zone1: {zone1}, zone1 ends at time {time1*zone1} (should be just before pulse time)')
    print(f' beta (near radius / far radius): {config["Flow"]["laser"]["nearRadius"] / config["Flow"]["laser"]["farRadius"]}')


def get_batch_ids(rows: list) -> list:
    return [int(row[1]) for row in rows]


def parse_args():
    parser = argparse.ArgumentParser(description='Set up a batch of runs')
    parser.add_argument('database', type=str, help='Path to the database file, e.g. output/run_database_batch_1.csv')
    parser.add_argument('base_dir', type=str, help='Base directory for the runs, e.g. /p/lustre1/cutforth1/PSAAP/ or /p/gpfs1/cutforth1/PSAAP/')
    parser.add_argument('grid_size', type=str, help='Grid size for the runs, e.g. 5M or 15M')
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.base_dir)
    database_path = Path(args.database)
    grid_size = args.grid_size
    use_cuda = 1

    assert grid_size in ['2M', '5M', '15M'], f"Grid size {grid_size} not recognized. Use 5M or 15M."
    assert base_dir.exists(), f"Base directory {base_dir} does not exist"
    assert database_path.exists(), f"Database file {database_path} does not exist"

    if grid_size == '2M':
        config_path = Path('configs/GG-combustor-default-lassen-2M.json')
    elif grid_size == '5M':
        config_path = Path('configs/GG-combustor-default-lassen-5M.json')
    elif grid_size == '15M':
        config_path = Path('configs/GG-combustor-default-lassen-15M.json')
    else:
        raise ValueError(f"Grid size {grid_size} not recognized. Use 5M or 15M.")

    assert config_path.exists(), f"Reference config file {config_path} does not exist"

    run_database = read_csv_to_list_of_lists(database_path)
    batch_ids = get_batch_ids(run_database)

    assert len(set(batch_ids)) == 1, f"Multiple batch IDs found: {set(batch_ids)}"
    batch_id = batch_ids[0]

    outdir_base = base_dir / f'runs_batch_{batch_id:04d}'
    assert not outdir_base.exists(), f"Output directory {outdir_base} already exists"
    outdir_base.mkdir()

    # Load the reference config
    with open(config_path, 'r') as f:
        ref_config = json.load(f)

    for row in run_database:
        run_id = int(row[0])
        xi = load_xi_from_database_row(row)
        nominal_xz_coords = load_nominal_xz_coords_from_database_row(row)
        assert len(xi) == 17, f"Expected 17 parameters, got {len(xi)}"

        print(f'Preparing run {run_id} with nominal xz coords {nominal_xz_coords} and xi {xi}')

        run_dir = outdir_base / f'{run_id:04d}'
        run_dir.mkdir()

        config = ref_config.copy()
        update_json_data(config, xi, base_dir, grid_size)
        set_restart_frequency(config, nominal_xz_coords, batch_id)

        # Write the config to a new file
        with open(run_dir / 'GG-combustor.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Print out new config as sanity check
        with open(run_dir / 'GG-combustor.json', 'r') as f:
            new_config = json.load(f)
            print_out_config_info(new_config)

        # Write the run-htr.sh script
        with open(run_dir / 'run-htr.sh', 'w') as f:
            f.write('outidr="."')
            f.write(f'USE_CUDA={use_cuda} DEBUG=0 PROFILE=0 QUEUE="pbatch" $HTR_DIR/prometeo.sh -i GG-combustor.json -o "."')

        # Add in copies of common scripts we will use
        shutil.copy('scripts/jobscripts/run-htr-with-restarts_lsf.sh', run_dir)
        shutil.copy('src/prepare_restart_run.py', run_dir)
        shutil.copy('scripts/utils/organize-htr.sh', run_dir)

        # Make executable

        st = os.stat(run_dir / 'run-htr.sh')
        os.chmod(run_dir / 'run-htr.sh', st.st_mode | stat.S_IEXEC)

        st = os.stat(run_dir / 'run-htr-with-restarts_lsf.sh')
        os.chmod(run_dir / 'run-htr-with-restarts_lsf.sh', st.st_mode | stat.S_IEXEC)

        st = os.stat(run_dir / 'organize-htr.sh')
        os.chmod(run_dir / 'organize-htr.sh', st.st_mode | stat.S_IEXEC)

    shutil.copy('scripts/jobscripts/start_many_runs_lsf.sh', outdir_base)
    st = os.stat(outdir_base / 'start_many_runs_lsf.sh')
    os.chmod(outdir_base / 'start_many_runs_lsf.sh', st.st_mode | stat.S_IEXEC)


if __name__ == "__main__":
    main()
