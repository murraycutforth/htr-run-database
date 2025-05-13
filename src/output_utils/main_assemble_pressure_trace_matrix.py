import csv
from pathlib import Path
import argparse
import logging

import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# These are some hardcoded non-dimensional times
MAX_TIME=45  # 700 us post-laser
MAX_LASER_DELAY=21  # Based on the zone1 time step and the laser delay iterations

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Process pressure files.")
    parser.add_argument('--run-dir', type=str, help='Path to directory where runs are stored')
    parser.add_argument('--dt', type=float, default=0.008, help='Time step for resampling (0.006 for 2M, 0.004 for 15M)')
    args = parser.parse_args()
    return args


def read_pressure_file(file_path: Path):
    assert file_path.exists(), f"File {file_path} does not exist."
    assert file_path.suffix == '.csv', f"File {file_path} is not a CSV file."
    df = pd.read_csv(file_path, header=0, sep=r'\s+')[['Iter', 'Time', 'Pressure']]
    return df


def read_all_pressure_files(solution_dir: Path):
    assert solution_dir.exists()

    probe_inds = [0, 1, 2, 3, 4, 5]

    probe_ind_to_df = {}

    for probe_ind in probe_inds:
        pressure_files = sorted(solution_dir.glob(f"probe{probe_ind}-*.csv"))
        sub_dfs = []
        for pressure_file in pressure_files:
            sub_df = read_pressure_file(pressure_file)
            sub_dfs.append(sub_df)

        # Concatenate all sub_dfs and group by 'Iter' to handle overlapping ranges (if there are multiple restarts)
        if sub_dfs:
            combined_df = pd.concat(sub_dfs).groupby('Iter').agg({
                'Time': 'first',
                'Pressure': 'first'  # Or any other aggregation function you prefer
            }).reset_index()
            probe_ind_to_df[probe_ind] = combined_df

    return probe_ind_to_df


def get_average_pressure_trace(probe_ind_to_df):
    # Check that 'Iter' and 'Time' are identical across all DataFrames
    iter_time_set = None
    for df in probe_ind_to_df.values():
        iter_time = df[['Iter', 'Time']].drop_duplicates()
        if iter_time_set is None:
            iter_time_set = iter_time
        else:
            if not iter_time.equals(iter_time_set):
                raise ValueError("Mismatch in 'Iter' and 'Time' columns across DataFrames")

    # Compute the average pressure trace
    avg_pressure_trace = pd.DataFrame()
    avg_pressure_trace['Iter'] = iter_time_set['Iter']
    avg_pressure_trace['Time'] = iter_time_set['Time']
    avg_pressure_trace['Avg_Pressure'] = pd.concat([df['Pressure'] for df in probe_ind_to_df.values()], axis=1).mean(axis=1)

    return avg_pressure_trace


def plot_pressure_traces(run_id_to_df):
    # Plot avg pressure vs time
    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.plot(data['Time'], data['Avg_Pressure'])
    plt.xlabel('Time')
    plt.ylabel('Avg Pressure')
    plt.title('Average Pressure Traces')
    plt.show()

    # Two axes, one with complete runs and the other with incomplete runs

    #global_max_iter = 0
    #for data in run_id_to_df.values():
    #    global_max_iter = max(global_max_iter, data['Iter'].max())

    #fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    #for run_id, data in run_id_to_df.items():
    #    if data['Iter'].max() < global_max_iter:  # Example condition for incomplete runs
    #        ax[1].plot(data['Time'], data['Avg_Pressure'], label=f'Run {run_id}')
    #    else:
    #        ax[0].plot(data['Time'], data['Avg_Pressure'], label=f'Run {run_id}')
    #ax[0].set_title('Complete Runs')
    #ax[1].set_title('Incomplete Runs')
    #ax[0].set_xlabel('Time')
    #ax[0].set_ylabel('Avg Pressure')
    #ax[1].set_xlabel('Time')
    #ax[1].set_ylabel('Avg Pressure')
    #plt.tight_layout()
    #plt.show()


def plot_max_iter_histogram(run_id_to_df):
    # Plot histogram of max iterations
    max_iters = [df['Iter'].max() for df in run_id_to_df.values()]
    plt.hist(max_iters, bins=30)
    plt.xlabel('Max Iteration')
    plt.ylabel('Frequency')
    plt.title('Histogram of Max Iterations')
    plt.show()


def run_database_path():
    # Get the path to the run database
    return Path(__file__).parent.parent.parent / 'output' / 'run_database_total.csv'


def load_xi(run_id):
    with open(run_database_path(), 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        ids = [int(row[0]) for row in rows]
    assert run_id in ids, f"Run ID {run_id} not found in run_database_total.csv"
    idx = np.nonzero(np.array(ids) == run_id)[0][0]
    return rows[idx]


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

        # Drop any rows where Time_shifted is greater than MAX_TIME
        df = df[df['Time_shifted'] <= MAX_TIME]

        # Only keep this run if is within 1 unit of MAX_TIME
        if df['Time_shifted'].max() < MAX_TIME - 1:
            print(f"Run {run_id} is too short ({df['Time_shifted'].max()}) after applying laser time shift, skipping")
            continue

        run_id_to_df_updated[run_id] = df
        xis_updated.append(xi)

    return run_id_to_df_updated, xis_updated


def resample_run_data(run_id_to_df, time_points):
    """
    Resample all runs in run_id_to_df to the specified time points using linear interpolation.

    Parameters:
        run_id_to_df (dict): Dictionary where keys are run IDs and values are DataFrames with 'Time' and 'Avg_Pressure'.
        time_points (array-like): Array of time points to interpolate at.

    Returns:
        dict: Updated run_id_to_df with resampled data.
    """
    resampled_run_id_to_df = {}

    for run_id, df in tqdm(run_id_to_df.items(), desc="Resampling pressure to common time points"):
        # Ensure the DataFrame has the required columns
        assert 'Time_shifted' in df.columns and 'Avg_Pressure' in df.columns, \
            f"DataFrame for run {run_id} must contain 'Time_shifted' and 'Avg_Pressure' columns."

        # Perform linear interpolation
        interpolated_pressure = np.interp(time_points, df['Time_shifted'], df['Avg_Pressure'])

        logger.debug(f"Resampling run {run_id} from {len(df)} to {len(time_points)} points. Original time range from {df['Time_shifted'].min()} to {df['Time_shifted'].max()}.")

        # Create a new DataFrame with the resampled data
        resampled_df = pd.DataFrame({
            'Time': time_points,
            'Avg_Pressure': interpolated_pressure
        })

        resampled_run_id_to_df[run_id] = resampled_df

    return resampled_run_id_to_df


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    dt = args.dt

    assert run_dir.exists(), f"Run directory {run_dir} does not exist."
    assert dt > 0, f"Time step {dt} must be greater than 0."

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

    plot_pressure_traces(run_id_to_df)
    plot_max_iter_histogram(run_id_to_df)
    plot_laser_locations(run_id_to_df, xis)

    run_id_to_df, xis = apply_laser_time_shift(run_id_to_df, xis)

    time_points = np.arange(0, MAX_TIME, dt)
    run_id_to_df_resampled = resample_run_data(run_id_to_df, time_points)

    plot_pressure_traces(run_id_to_df_resampled)

    # Write out to CSV after combining to one dataframe with new run_id row
    run_ids = np.array(list(run_id_to_df_resampled.keys()))

    pressures = []
    for run_id, df in run_id_to_df_resampled.items():
        pressures.append(df['Avg_Pressure'].values)
        print(f"Run {run_id} has shape {df.shape}")

    # Check that all pressures are the same length
    if len(set([len(p) for p in pressures])) != 1:
        raise ValueError(f"Not all pressure traces are the same length: {set([len(p) for p in pressures])}")
    pressures = np.array(pressures, dtype=np.float32)
    pressures = pressures.T  # Transpose to have shape (num_time_points, num_runs)

    # Subtract average initial pressure, so we have the pressure change
    p0 = np.mean(pressures[0, :])
    pressures = pressures - p0

    print(f'Resampled run time range from {time_points[0]} to {time_points[-1]}')

    assert Path(f'pressure_traces_{run_dir.stem}.npz').exists() == False, f"File pressure_traces_{run_dir.stem}.npz already exists. Please delete it before running this script."

    np.savez_compressed(f'pressure_traces_{run_dir.stem}.npz', pressure=pressures, run_id=run_ids, time_points=time_points)
    print(f"Saved pressure traces to pressure_traces_{run_dir.stem}.npz with shape {pressures.shape}")









if __name__ == "__main__":
    main()
