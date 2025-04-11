import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.linalg.interpolative import interp_decomp, reconstruct_matrix_from_id

BATCH = 1


def load_xi(run_id):
    with open('./../../output/run_database_total.csv', 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        ids = [int(row[0]) for row in rows]
    assert run_id in ids
    idx = np.nonzero(np.array(ids) == run_id)[0][0]
    return rows[idx]


def data_matrix_preprocessing(pressures, run_ids):
    # Convert from p to delta p, by taking average of first value and subtracting

    p0 = np.mean(pressures[0, :])
    pressures = pressures - p0

    # Remove one particular outlier, p > 0.75 at index 1000
    if BATCH == 1:
        outlier_idx = np.where(pressures[1000, :] > 0.75)[0]
        print(outlier_idx)
        pressures = pressures[:, [i for i in range(pressures.shape[1]) if i not in outlier_idx]]
        run_ids = run_ids[[i for i in range(len(run_ids)) if i not in outlier_idx]]

    print(len(run_ids), pressures.shape)

    return pressures, run_ids


def plot_chi_vs_focal_position(xis, chis):
    plt.figure(figsize=(3, 4), dpi=200)
    xs = [float(xi[2]) for xi in xis]
    ys = [float(xi[4]) for xi in xis]
    plt.scatter(xs, ys, c=chis, alpha=0.3)
    plt.xlabel('radial distance [mm]')
    plt.ylabel('streamwise distance [mm]')
    plt.show()


def main():
    print("Current working directory:", os.getcwd())

    data = np.load(f'./../pressure_traces_{BATCH}.npz', allow_pickle=True)
    pressures = data['pressure'].astype(np.float64)
    run_ids = data['run_id']

    pressures, run_ids = data_matrix_preprocessing(pressures, run_ids)

    ignition_mask = pressures[-1, :] > 0.1

    plt.figure()
    for i in range(pressures.shape[1]):
        if ignition_mask[i]:
            plt.plot(pressures[:, i])
    plt.show()


    print(len(run_ids))

    ignited_runs = [id for k, id in enumerate(run_ids) if ignition_mask[k]]
    non_ignited_runs = [id for k, id in enumerate(run_ids) if not ignition_mask[k]]

    print(len(ignited_runs), len(non_ignited_runs))

    xis = np.array([load_xi(run_id) for run_id in run_ids])
    chis = []
    for xi in xis:
        run_id = int(xi[0])
        if run_id in ignited_runs:
            chis.append(1)
        else:
            chis.append(0)

    plot_chi_vs_focal_position(xis, chis)


if __name__ == "__main__":
    main()