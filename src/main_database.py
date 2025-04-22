"""
Run this script to create the run database. This outputs a csv file, which can be copied to Dane/Lassen, and used as the input
to `prepare_run_batch.py`, to set up the required directories and config files for a simulation. Can be run locally, then
commit the output database files.
"""
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from uq_database import *


def create_pairplot(database_path, batch_id):
    df = pd.read_csv(database_path, header=0)
    sns.pairplot(df, diag_kind='kde')
    plt.savefig(f'./../output/database_pairplot_batch_{batch_id}.png')
    plt.close()


def main():
    # Seed for reproducibility every time we run this function
    np.random.seed(42)

    database_creators = [
        CreateDatabaseBatchV1(),
        CreateDatabaseBatchV2(),
        CreateDatabaseBatchV3(),
        CreateDatabaseBatchV4(),
        CreateDatabaseBatchV5(),
        CreateDatabaseBatchV6(),
        CreateDatabaseBatchV7(),
        CreateDatabaseBatchV8(),
        CreateDatabaseBatchV9(),
        CreateDatabaseBatchV10(),
    ]

    # Add assertions here to check database never changes
    assert float(database_creators[0].rows[0][2]) == 0.07450712295168489, f'{database_creators[0].rows[0][2]}'
    assert float(database_creators[0].rows[-1][2]) == 11.928219254936558, f'{database_creators[0].rows[-1][2]}'

    for creator in database_creators:
        create_pairplot(creator.outdir / creator.batch_database_name, creator.batch_id)



if __name__ == "__main__":
    main()


