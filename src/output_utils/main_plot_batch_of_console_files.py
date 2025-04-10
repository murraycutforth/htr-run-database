from os.path import expanduser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_line(line):
    parts = line.split()
    return [float(x) for x in parts]

def read_console_files(base_dir: Path, run_id: int):
    console_dir = base_dir / f"{run_id:04d}/solution"
    console_files = sorted(console_dir.glob("console-*.txt"))

    df_list = []
    for f in console_files:
        with open(f, 'r') as infile:
            read_lines = infile.readlines()
            for k, line in enumerate(read_lines[1:-1]):
                if k == 0:
                    continue
                try:
                    df_list.append(read_line(line))
                except ValueError:
                    print(f"Error processing file: {f} in run: {run_id}")
                    raise

    data = np.array(df_list)
    return data if len(data.shape) == 2 else None

def plot_data(run_id_to_df):
    # Plot avg temperature vs. iteration
    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.scatter(data[:, 1], data[:, 5])
    plt.xlabel('time')
    plt.ylabel('Avg Temperature')
    plt.show()

    # Plot average pressure vs. iteation
    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.scatter(data[:, 1], data[:, 4])
    plt.xlabel('time')
    plt.ylabel('Avg Pressure')
    plt.show()

    # time vs iteration
    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.scatter(data[:, 1], data[:, 0])
    plt.xlabel('time')
    plt.ylabel('Iteration')
    plt.show()

    # Plot largest iteration for each run as 1D scatter plot
    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.scatter([run_id], np.max(data[:, 0]))
        print(f"Run ID: {run_id}, Largest Iteration: {np.max(data[:, 0])}")
    plt.xlabel('Run ID')
    plt.ylabel('Largest Iteration')
    plt.show()

    # Plot vol burned vs. iteration
    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.plot(data[:, 1], data[:, 5])
    plt.xlabel('time')
    plt.ylabel('Vol Burned')
    plt.show()

    fig, ax = plt.subplots()
    for run_id, data in run_id_to_df.items():
        ax.scatter([run_id], [data[0, 5]])
    plt.xlabel('Run ID')
    plt.ylabel('Initial Avg Pressure')
    plt.title('Run ID vs Initial Avg Pressure')
    plt.show()

def plot_focal_coordinates(csv_file):
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['focal_x'], df['focal_z'], marker='o', alpha=0.7)
    plt.xlabel('Focal X')
    plt.ylabel('Focal Z')
    plt.title('Focal X vs Focal Z Coordinates')
    plt.grid(True)
    plt.show()


def main():
    base_dir = Path('~/Downloads/run_batch_1/runs_batch_0001').expanduser()
    assert base_dir.exists(), f"Base directory {base_dir} does not exist."

    run_id_to_df = {}
    for i in range(0, 390):
        data = read_console_files(base_dir, i)
        if data is not None:
            run_id_to_df[i] = data

    plot_data(run_id_to_df)

    plot_focal_coordinates('../../output/run_database_batch_1.csv')


if __name__ == "__main__":
    main()