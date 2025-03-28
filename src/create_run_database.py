import csv
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

from focal_position_offset import spark_x_offset_params, spark_y_offset_params, spark_z_offset_params


# Reference length scale [m]
LREF = 0.003175


# Nominal laser focal positions ====================================
# See: https://docs.google.com/spreadsheets/d/1OXftyrMRB5fDfLHqA_bKCJLqps3D0yuWGEzWwQqnMVc/edit?gid=0#gid=0
# We use functions in focal_position_offset to sample uncertainty in focal position given nominal x,z

NOMINAL_LASER_Xs = [  # [mm]
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
]

NOMINAL_LASER_Zs = [  # [mm]
    6.4,
    12.7,
    19.05,
]


# Nominal values of other UQ continuous UQ parameters =============================
# We use a uniform distribution centered around the nominal value to sample these parameters
# Each parameter has a different relative uncertainty used to set the parameters of the uniform distribution

U_AXIAL_LENGTH = 0.571
TF_BETA_REF = 0.55  # Thickened flame beta
SL0_REF = 0.01031913541
C_S_REF = 0.16
MDOT1_REF = 1.77125
MDOT2_REF = 0.45374
FWHM_REF = 0.0015175099041

# Parameters of uniform distros
PARAM_DISTRIBUTIONS = {
    'axial_l': (U_AXIAL_LENGTH * 0.9, U_AXIAL_LENGTH * 1.1),  # Axial length, part of alpha
    'alpha': (2.0, 2.5),  # Alpha parameter
    'beta': (1.0, 2.3),  # Beta parameter
    'energy': (467460.0, 571340.0),  # Energy parameter
    'fwhm': (FWHM_REF * 0.5, FWHM_REF * 1.5),  # FWHM gains
    'tf_beta': (TF_BETA_REF * 0.9, TF_BETA_REF * 1.1),  # Thickened flame beta
    'sL0': (SL0_REF * 0.95, SL0_REF * 1.05),  # Laminar flame speed
    'arr_factor': (0.95, 1.05),  # Reaction rate
    'C_S': (C_S_REF * 0.95, C_S_REF * 1.05),  # Smagorinsky constant
    'mDot1': (MDOT1_REF * 0.95, MDOT1_REF * 1.05),  # Mass flow rate oxygen
    'mDot2': (MDOT2_REF * 0.95, MDOT2_REF * 1.05),  # Mass flow rate methane
}

# Categorical UQ parameters ========================================

METH_RESTARTS = ['fluid_iter0000040000', 'fluid_iter0000060000',
                 'fluid_iter0000080000']  # Change amount of background methane
SQUIRCS = ['S_0p70', 'S_0p80', 'S_0p90']  # Different squircularity

# Aleatoric uncertainty parameters =================================
# This is treated differently from the other epistemic parameters
# We want to run multiple aleatoric samples for each epistemic sample
# But only at some locations

TIMES_VEC = [1000, 2000, 3000, 4000, 5000,
             6000]  # Different turbulent initialisation (start from this # of timesteps on top of methane restart)


# Functions to sample UQ params, and edit config file ==============

def get_peak_e_dot(beta):
    # This is an empirical relation to set the energy (not a UQ var) to compensate for the fact that
    # beta is set by increasing the near radius (increasing the size of the kernel and thus total energy)
    # so this relation scales energy so that total energy is independent of beta

    e_ref = 19.76569917
    e_nominal = 448285.0

    x_fit = np.array([1.0, 1.10714286, 1.21428571, 1.32142857, 1.42857143,
                      1.53571429, 1.64285714, 1.75, 1.85714286, 1.96428571,
                      2.07142857, 2.17857143, 2.28571429, 2.39285714, 2.5])
    y_fit = np.array([30.02545745, 27.94039608, 26.19985435, 24.74690647, 23.53403262,
                      22.52156479, 21.67638943, 20.9708644, 20.38191495, 19.89027901,
                      19.47987725, 19.13728715, 18.85130402, 18.61257457, 18.41329095])
    interp = interp1d(x_fit, y_fit, kind='linear')
    e_beta = interp(beta)

    # This scales e_nominal by a factor which is inversely proportional to beta
    peak_e_dot = e_nominal * e_ref / e_beta

    return peak_e_dot


def sample_uq_vector_v1(nominal_x: float, nominal_z: float):
    xi = []

    # First sample laser focal position (xi_0 to xi_2)
    xi.append(nominal_x + np.random.normal(**spark_x_offset_params(nominal_x, nominal_z)))
    xi.append(0.0 + np.random.normal(**spark_y_offset_params()))
    xi.append(nominal_z + np.random.normal(**spark_z_offset_params(nominal_x, nominal_z)))

    # Then sample continuous UQ params defined by uniform distributions above (xi_3 to xi_13)
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['axial_l']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['alpha']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['beta']))
    peak_e_dot = get_peak_e_dot(xi[-1])
    xi.append(peak_e_dot)  # xi[6]
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['fwhm']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['tf_beta']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['sL0']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['arr_factor']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['C_S']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['mDot1']))
    xi.append(np.random.uniform(*PARAM_DISTRIBUTIONS['mDot2']))

    # Finally sample discrete UQ params (xi_14 to xi_16)
    xi.append(int(np.random.choice(TIMES_VEC)))
    xi.append(str(np.random.choice(METH_RESTARTS)))
    xi.append(str(np.random.choice(SQUIRCS)))

    return xi


def resample_aleatoric_vars(xi: list) -> list:
    xi = xi.copy()
    xi[14] = np.random.choice(TIMES_VEC)
    return xi



class CreateDatabaseBatch(ABC):
    def __init__(self, batch_id: int):
        self.batch_id = batch_id
        self.batch_database_name = f'run_database_batch_{batch_id}.csv'
        self.outdir = Path(__file__).parent.parent / 'output'
        self.total_database_name = 'run_database_total.csv'

        print(f'Looking for existing database in {self.outdir}')

        # Define the database creation steps here, so when a child class is instantiated, the database is created
        existing_batches = self.load_existing_batch_ids()

        self.rows = self.create_batch()

        if (not existing_batches) or (not self.batch_id in set(existing_batches)):
            self.validate_batch(self.rows)
            self.save_batch(self.rows)
        else:
            print(f'Batch {self.batch_id} already exists in database, not overwriting')

    @abstractmethod
    def create_batch(self) -> list:
        pass

    def load_existing_ids(self):
        try:
            with open(self.outdir / self.total_database_name, 'r') as f:
                reader = csv.reader(f)
                _ = next(reader)
                rows = [row for row in reader]
                ids = [int(row[0]) for row in rows]
                return ids
        except:
            return None

    def load_existing_batch_ids(self):
        try:
            with open(self.outdir / self.batch_database_name, 'r') as f:
                reader = csv.reader(f)
                _ = next(reader)
                rows = [row for row in reader]
                ids = [int(row[1]) for row in rows]
                return ids
        except:
            return None

    def save_batch(self, rows):

        num_runs = len(rows)

        with open(self.outdir / self.batch_database_name, 'w') as f:
            f.write('run_id,batch_id,focal_x,focal_y,focal_z,axial_l,alpha,beta,energy,fwhm,tf_beta,sL0,arr_factor,C_S,mDot1,mDot2,times,meth_restart,squirc\n')
            for i in range(num_runs):
                f.write(','.join(map(str, rows[i])) + '\n')

        with open(self.outdir / self.total_database_name, 'a') as f:
            for i in range(num_runs):
                f.write(','.join(map(str, rows[i])) + '\n')

        print(f'Saved batch {self.batch_id} to {self.outdir / self.batch_database_name}')

    def validate_batch(self, rows):
        print(f'Validating batch {self.batch_id}...')

        # Check if any of the new runs have the same ID as existing runs
        ids = self.load_existing_ids()
        new_ids = [rows[i][0] for i in range(len(rows))]
        new_ids_set = set(new_ids)
        assert len(new_ids) == len(new_ids_set), f'New IDs contain duplicates: {new_ids}'
        if ids:
            ids_set = set(ids)
            assert len(ids) == len(ids_set), 'Existing IDs contain duplicates'
            assert len(ids_set.intersection(new_ids_set)) == 0, 'New IDs overlap with existing IDs'

        # Check all new runs have correct batch ID
        new_batch_ids = [rows[i][1] for i in range(len(rows))]
        new_batch_ids_set = set(new_batch_ids)
        assert len(new_batch_ids_set) == 1, 'New runs have different batch IDs'
        for id in new_batch_ids:
            assert id == self.batch_id, 'New runs have incorrect batch ID'

        batch_ids = self.load_existing_batch_ids()
        if batch_ids:
            batch_ids_set = set(batch_ids)
            assert self.batch_id not in batch_ids_set, 'Batch ID already exists in database'

        # Check types
        assert isinstance(rows, list)
        for row in rows:
            assert isinstance(row, list)
            assert len(row) == 19, f'Expected 19 columns, got {len(row)}, row: {row}'
            assert isinstance(row[0], int)
            assert isinstance(row[1], int)
            assert isinstance(row[2], float)
            assert isinstance(row[3], float)
            assert isinstance(row[4], float)
            assert isinstance(row[5], float)
            assert isinstance(row[6], float)
            assert isinstance(row[7], float)
            assert isinstance(row[8], float)
            assert isinstance(row[9], float)
            assert isinstance(row[10], float)
            assert isinstance(row[11], float)
            assert isinstance(row[12], float)
            assert isinstance(row[13], float)
            assert isinstance(row[14], float)
            assert isinstance(row[15], float)
            assert isinstance(row[16], int)
            assert isinstance(row[17], str)
            assert isinstance(row[18], str)


class CreateDatabaseBatchV1(CreateDatabaseBatch):
    """First batch, run 10 epistemic samples for each location, with just 1 aleatoric sample each
    """

    def __init__(self):
        super().__init__(batch_id=1)

    def create_batch(self) -> list:
        runs_per_loc = 10
        rows = []
        ids = self.load_existing_ids()
        run_id = max(ids) + 1 if ids else 0

        for x in NOMINAL_LASER_Xs:
            for z in NOMINAL_LASER_Zs:
                for _ in range(runs_per_loc):
                    xi = sample_uq_vector_v1(x, z)
                    rows.append([run_id, self.batch_id] + xi)
                    run_id += 1

        print(f'Created batch {self.batch_id} with {len(rows)} runs')

        return rows


