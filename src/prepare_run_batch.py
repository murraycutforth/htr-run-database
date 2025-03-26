"""
We take reference GG-combustor.json file and modify it for each run in the batch, put them into a new directory
for each one, and record the mapping from run_id to directory in a csv file.
"""

import argparse
import json
from pathlib import Path
import csv

# Path to reference config file
REF_CONFIG = 'GG-combustor-default.json'

# Path to CommonCase BC and grid files (currently hardcoded for me on dane.llnl.gov)
COMMON_CASE_DIR = '/p/lustre1/cutforth1/PSAAP/'

# Path to dir where runs are executed and output written (again, hardcoded for me on dane.llnl.gov)
RUN_DIR = '/p/lustre1/cutforth1/PSAAP/'

# Reference length scale [m]
LREF = 0.003175


def get_path_to_common_case(xi: list) -> Path:
    common_x_locs = [6.0, 7.0, 8.0, 9.0, 10.0]
    common_z_locs = [6.0, 13.0, 19.0]

    xi_x = xi[0] / LREF
    xi_z = xi[2] / LREF

    # Find closest common x and z locations
    x_idx = min(range(len(common_x_locs)), key=lambda i: abs(common_x_locs[i] - xi_x))
    z_idx = min(range(len(common_z_locs)), key=lambda i: abs(common_z_locs[i] - xi_z))

    x_str = f'{int(common_x_locs[x_idx]):02d}'
    z_str = f'{int(common_z_locs[z_idx]):02d}'
    path = Path(COMMON_CASE_DIR) / f'location_{x_str}_0_{z_str}' / 'CommonCase' / xi[16]
    assert path.exists(), f"Common case directory {path} does not exist"
    return path


def update_json_data(config: dict, xi: list) -> None:
    """Edit the given JSON data with the sampled parameters."""
    common_case_dir = get_path_to_common_case(xi)

    # Update laser focal location
    config['Flow']['laser']['focalLocation'][0] = xi[0] / LREF
    config['Flow']['laser']['focalLocation'][1] = xi[2] / LREF  # Note switched y/z indices
    config['Flow']['laser']['focalLocation'][2] = xi[1] / LREF

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
    config['Flow']['initCase']['restartDir'] = common_case_dir / 'solution' / f'{xi[15]}'

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
    config['BC']['xBCLeft']['MixtureProfile']['FileDir'] = common_case_dir / 'bc-6sp'
    config['BC']['xBCLeft']['TemperatureProfile']['FileDir'] = common_case_dir / 'bc-6sp'
    config['BC']['xBCLeft']['VelocityProfile']['FileDir'] = common_case_dir / 'bc-6sp'
    config['Grid']['GridInput']['gridDir'] = common_case_dir / 'bc-6sp' / 'grid'


def parse_args():
    parser = argparse.ArgumentParser(description='Set up a batch of runs')
    parser.add_argument('database', type=str, help='Path to the database file')
    return parser.parse_args()


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


def main():
    args = parse_args()

    database_path = Path(args.database)
    config_path = Path(REF_CONFIG)

    assert database_path.exists(), f"Database file {database_path} does not exist"
    assert config_path.exists(), f"Reference config file {config_path} does not exist"

    run_database = read_csv_to_list_of_lists(database_path)
    batch_ids = get_batch_ids(run_database)

    assert len(set(batch_ids)) == 1, f"Multiple batch IDs found: {set(batch_ids)}"
    batch_id = batch_ids[0]

    outdir_base = Path(RUN_DIR) / f'runs_batch_{batch_id:04d}'
    assert not outdir_base.exists(), f"Output directory {outdir_base} already exists"
    outdir_base.mkdir()

    # Load the reference config
    with open(REF_CONFIG, 'r') as f:
        ref_config = json.load(f)

    for row in run_database:
        run_id = int(row[0])
        xi = load_xi_from_database_row(row)
        assert len(xi) == 16, f"Expected 16 parameters, got {len(xi)}"

        run_dir = outdir_base / f'{run_id:04d}'
        run_dir.mkdir()

        config = ref_config.copy()
        update_json_data(config, xi)

        # Write the config to a new file
        with open(run_dir / 'GG-combustor.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Write the run-htr.sh script
        with open(run_dir / 'run-htr.sh', 'w') as f:
            f.write('outidr="."')
            f.write('USE_CUDA=0 DEBUG=0 PROFILE=0 QUEUE="pbatch" $HTR_DIR/prometeo.sh -i GG-combustor.json -o "."')


if __name__ == "__main__":
    main()