import csv
from pathlib import Path
import argparse
import logging

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.output_utils.main_assemble_pressure_trace_matrix import read_pressure_file, read_all_pressure_files, get_average_pressure_trace, load_xi, run_database_path

# TODO

# These are some hardcoded non-dimensional times
MAX_LASER_DELAY=21  # Based on the zone1 time step and the laser delay iterations
MAX_TIME=30  # Drop anything shorter than this, as it has crashed early

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Process pressure files.")
    parser.add_argument('--run-dir', type=str, help='Path to directory where runs are stored')
    args = parser.parse_args()
    return args


def plot_laser_locations(run_id_to_df, xis):
    all_xi = np.array(xis)
    xs = all_xi[:, 2].astype(float)  # Radial distance [mm]
    zs = all_xi[:, 4].astype(float)  # Streamwise distance [mm]

    fig, ax = plt.subplots(dpi=200)
    ax.scatter(xs, zs, s=10, alpha=0.2)
    ax.set_xlabel('Radial Distance [mm]')
    ax.set_ylabel('Streamwise Distance [mm]')
    ax.set_title('Laser Locations')
    ax.set_aspect('equal')
    plt.show()


def plot_ignitions(xis, chis):
    all_xi = np.array(xis)
    xs = all_xi[:, 2].astype(float)
    zs = all_xi[:, 4].astype(float)
    fig, ax = plt.subplots(dpi=200)
    colors = ['red' if chi else 'blue' for chi in chis]
    ax.scatter(xs, zs, s=10, c=colors, alpha=1)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='red', label='Ignition',
                                  markersize=8),
                       plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='blue', label='No Ignition',
                                  markersize=8)]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('Radial Distance [mm]')
    ax.set_ylabel('Streamwise Distance [mm]')
    ax.set_aspect('equal')

    plt.show()


def apply_laser_time_shift(run_id_to_df, xis):
    # Align all runs so that laser fires at the same time

    run_id_to_df_updated = {}
    xis_updated = []

    for run_id, xi in zip(run_id_to_df.keys(), xis):
        assert int(xi[0]) == int(run_id), f"Run ID {run_id} does not match xi file {xi[0]}"

        laser_delay_iterations = int(xi[16])
        assert laser_delay_iterations % 1000 == 0, f"Laser delay iterations {laser_delay_iterations} is not a multiple of 1000"
        assert laser_delay_iterations >= 1000, f"Laser delay iterations {laser_delay_iterations} is less than 1000"
        assert laser_delay_iterations in [1000, 2000, 3000, 4000, 5000, 6000]

        start_ind = (laser_delay_iterations + 1000) // 10
        assert start_ind >= 0, f"Start index {start_ind} is less than 0"

        df = run_id_to_df[run_id]

        if len(df) < start_ind:
            print(f"Run {run_id} is too short to apply laser time shift, skipping")
            continue

        df = df.iloc[start_ind:]

        start_time = df['Time'].iloc[0]

        assert start_time <= MAX_LASER_DELAY, f"Start time {start_time} is greater than MAX_LASER_DELAY {MAX_LASER_DELAY}"

        # Create the new column all at once
        df = df.assign(Time_shifted=df['Time'] - start_time)

        # Only keep this run if is within 1 unit of MAX_TIME
        if df['Time_shifted'].max() < MAX_TIME - 1:
            print(f"Run {run_id} is too short ({df['Time_shifted'].max()}) after applying laser time shift, skipping")
            continue

        run_id_to_df_updated[run_id] = df
        xis_updated.append(xi)

    return run_id_to_df_updated, np.array(xis_updated)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()

    assert run_dir.exists(), f"Run directory {run_dir} does not exist."

    run_id_to_df = {}

    solution_dirs = list(run_dir.glob('*/solution'))

    print(f'Found {len(list(solution_dirs))} solution directories in {run_dir}')

    for solution_dir in tqdm(solution_dirs, desc="Loading runs"):

        i = int(solution_dir.parent.name)

        try:
            assert solution_dir.exists(), f"Solution directory {solution_dir} does not exist."

            probe_ind_to_df = read_all_pressure_files(solution_dir)
            avg_pressure_trace = get_average_pressure_trace(probe_ind_to_df)
            run_id_to_df[i] = avg_pressure_trace

        except AssertionError as e:
            print(f"Skipping run {i:04d}: {e}")
            continue

    xis = [load_xi(run_id) for run_id in run_id_to_df.keys()]

    run_id_to_df, xis = apply_laser_time_shift(run_id_to_df, xis)

    print(f'After laser time shift, we have {len(run_id_to_df)} runs.')

    pressures = []
    for run_id, df in run_id_to_df.items():
        pressures.append(df['Avg_Pressure'].values)
        print(f"Run {run_id} has shape {df.shape}")

    avg_p0 = np.mean([p[0] for p in pressures])
    pressures = [p - avg_p0 for p in pressures]

    # Ignition status
    chis = np.array([p[-1] > 0.1 for p in pressures])

    plot_ignitions(xis, chis)

    assert Path(f'ignitions_{run_dir.stem}.npz').exists() == False, f"File ignitions_{run_dir.stem}.npz already exists. Please delete it before running this script."

    np.savez_compressed(f'ignitions_{run_dir.stem}.npz', xis=xis, chis=chis)

    print(f'Saved file ignitions_{run_dir.stem}.npz')









if __name__ == "__main__":
    main()
