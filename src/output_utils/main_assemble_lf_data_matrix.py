from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BATCH=4
#MAX_ITER=38780
MAX_TIME=69

# For 15M grid, time=70 corresponds to approx 38780 iterations


def read_pressure_file(file_path: Path):
    assert file_path.exists(), f"File {file_path} does not exist."
    assert file_path.suffix == '.csv', f"File {file_path} is not a CSV file."
    df = pd.read_csv(file_path, header=0, sep='\s+')[['Iter', 'Time', 'Pressure']]
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

        # Concatenate all sub_dfs and group by 'Iter' to handle overlapping ranges
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

    # Drop all rows where Time > 70.0
    avg_pressure_trace = avg_pressure_trace[avg_pressure_trace['Time'] <= MAX_TIME + 1]

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


def plot_pressure_traces_2(run_id_to_df):
    # Two axes, one with complete runs and the other with incomplete runs
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for run_id, data in run_id_to_df.items():
        if data['Time'].max() < MAX_TIME:  # Example condition for incomplete runs
            ax[1].plot(data['Time'], data['Avg_Pressure'], label=f'Run {run_id}')
        else:
            ax[0].plot(data['Time'], data['Avg_Pressure'], label=f'Run {run_id}')
    ax[0].set_title('Complete Runs')
    ax[1].set_title('Incomplete Runs')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Avg Pressure')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Avg Pressure')
    plt.tight_layout()
    plt.show()


def plot_max_iter_histogram(run_id_to_df):
    # Plot histogram of max iterations
    max_iters = [df['Iter'].max() for df in run_id_to_df.values()]
    print(max_iters)
    max_times = [df['Time'].max() for df in run_id_to_df.values()]
    print(max_times)
    plt.hist(max_iters, bins=30)
    plt.xlabel('Max Iteration')
    plt.ylabel('Frequency')
    plt.title('Histogram of Max Iterations')
    plt.show()


def fill_missing_values(run_id_to_df):
    def calculate_similarity(df1, df2):
        # Calculate similarity based on the valid part of the run
        min_iter = min(df1['Iter'].max(), df2['Iter'].max())
        df1_valid = df1[df1['Iter'] <= min_iter]
        df2_valid = df2[df2['Iter'] <= min_iter]
        return np.linalg.norm(df1_valid['Avg_Pressure'].values - df2_valid['Avg_Pressure'].values)

    run_ids = list(run_id_to_df.keys())
    new_run_id_to_df  = {}

    run_id_to_complete_runs = {}
    for run_id in run_ids:
        df = run_id_to_df[run_id]
        if df['Time'].max() >= MAX_TIME:
            run_id_to_complete_runs[run_id] = df
            new_run_id_to_df[run_id] = df

    print(f"Found {len(run_id_to_complete_runs)} complete runs.")

    for run_id in run_ids:
        df = run_id_to_df[run_id]
        if df['Time'].max() < MAX_TIME and df['Time'].max() > 20:  # Example condition for incomplete runs, otherwise just skip entirely
            print(f"Run {run_id} is incomplete (max iter={df['Iter'].max()}, max time={df['Time'].max()}), filling missing values.")
            similarities = []
            for other_run_id in run_id_to_complete_runs.keys():
                assert other_run_id != run_id
                other_df = run_id_to_df[other_run_id]
                similarity = calculate_similarity(df, other_df)
                similarities.append((similarity, other_run_id))
            similarities.sort(key=lambda x: x[0])
            most_similar_runs = [run_id_to_df[similarities[0][1]], run_id_to_df[similarities[1][1]]]
            most_similar_df_1 = most_similar_runs[0]
            most_similar_df_2 = most_similar_runs[1]
            print(f"Most similar two runs for {run_id}: {[s[1] for s in similarities[:2]]}")

            current_time = df['Time'].max()
            current_iter = df['Iter'].max()
            rows_added = 0

            # We want to keep adding rows until time > MAX_TIME
            while current_time < MAX_TIME:
                # Get the line from similar dfs with the smallest time greater than start time
                df_1 = most_similar_df_1[most_similar_df_1['Time'] > current_time]
                df_2 = most_similar_df_2[most_similar_df_2['Time'] > current_time]

                assert len(df_1) > 0, f"Expected at least one row in df_1, got {len(df_1)}"
                assert len(df_2) > 0, f"Expected at least one row in df_2, got {len(df_2)}"

                # Get the row with smallest time from each
                row_1 = df_1.iloc[0]
                row_2 = df_2.iloc[0]
                time_1 = row_1['Time']
                time_2 = row_2['Time']
                p1 = row_1['Avg_Pressure']
                p2 = row_2['Avg_Pressure']
                avg_pressure = 0.5 * (p1 + p2)
                current_time = 0.5 * (time_1 + time_2)
                current_iter = current_iter + 1

                new_row = {'Iter': current_iter, 'Time': current_time, 'Avg_Pressure': avg_pressure}
                df = df._append(new_row, ignore_index=True)

                current_time = df['Time'].max()
                rows_added += 1

            print(f'New shape of df: {df.shape}, added {rows_added} rows')
            new_run_id_to_df[run_id] = df

    run_id_to_df = new_run_id_to_df

    # Now check max times of all dfs
    for run_id, df in run_id_to_df.items():
        print(f"Run {run_id} has max time {df['Time'].max()} and max iter {df['Iter'].max()}")

    # Now truncate all runs to 36,000 iters
    #for run_id, df in run_id_to_df.items():
    #    if df['Iter'].max() > MAX_ITER:
    #        run_id_to_df[run_id] = df[df['Iter'] <= MAX_ITER]

    # Truncate all runs to same number of rows, using minimum length
    min_length = min([len(df) for df in run_id_to_df.values()])
    for run_id, df in run_id_to_df.items():
        if len(df) > min_length:
            run_id_to_df[run_id] = df.iloc[:min_length]

    # Check all runs are the same length now
    lengths = [len(df) for df in run_id_to_df.values()]
    if len(set(lengths)) != 1:
        raise ValueError(f"Not all runs are the same length after filling missing values. Lengths: {set(lengths)}")

    print('Total of ', len(run_id_to_df), 'runs with imputed values')

    return run_id_to_df


def main():
    base_dir = Path(f'~/Downloads/run_batch_{BATCH}/runs_batch_{BATCH:04d}').expanduser()
    assert base_dir.exists(), f"Base directory {base_dir} does not exist."

    run_id_to_df = {}

    solution_dirs = list(base_dir.glob('*/solution'))

    print(f'Found {len(list(solution_dirs))} solution directories in {base_dir}')

    for solution_dir in solution_dirs:

        i = int(solution_dir.parent.name)
        print(f"Processing run {i:04d}...")

        try:
            assert solution_dir.exists(), f"Solution directory {solution_dir} does not exist."

            probe_ind_to_df = read_all_pressure_files(solution_dir)
            avg_pressure_trace = get_average_pressure_trace(probe_ind_to_df)

            run_id_to_df[i] = avg_pressure_trace

        except AssertionError as e:
            print(f"Skipping run {i:04d}: {e}")
            continue

    plot_pressure_traces(run_id_to_df)
    plot_pressure_traces_2(run_id_to_df)
    plot_max_iter_histogram(run_id_to_df)

    # Need to do something about runs which crashed early so
    # Impute missing values from other runs?
    # Let us fill missing values by taking the mean of the two most similar runs (measured over the valid part of the run)

    run_id_to_df = fill_missing_values(run_id_to_df)
    plot_pressure_traces(run_id_to_df)

    # Write out to CSV after combining to one dataframe with new run_id row
    run_ids = np.array(list(run_id_to_df.keys()))

    pressures = []
    for run_id, df in run_id_to_df.items():
        pressures.append(df['Avg_Pressure'].values)
        print(f"Run {run_id} has shape {df.shape}")

    # Check that all pressures are the same length
    if len(set([len(p) for p in pressures])) != 1:
        raise ValueError(f"Not all pressure traces are the same length: {set([len(p) for p in pressures])}")
    pressures = np.array(pressures, dtype=np.float32)
    pressures = pressures.T

    np.savez_compressed(f'pressure_traces_{BATCH}.npz', pressure=pressures, run_id=run_ids)
    print(f"Saved pressure traces to pressure_traces_{BATCH}.npz with shape {pressures.shape}")









if __name__ == "__main__":
    main()
