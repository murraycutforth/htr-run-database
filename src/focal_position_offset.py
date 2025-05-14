"""There is a significant difference between nominal location and measured spark location in the experimental data.
The offset can be modeled as a 2D Gaussian. We fit different parameters to each nominal location, and then
use linear interpolation in 2D to obtain parameters for the Gaussian distribution modeling the offset at
any nominal location.

This is used in our UQ space dataset generation (setup_lf_run_database.py)
"""
from collections import defaultdict
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf


# These functions are called by setup_lf_run_database.py to generate the offset parameters

"""
See this email from Ryan:

The "measured" spark locations come from the first frame of the video where laser plasma is observable. 
Since we were recording at 500 kHz, the maximum amount of time between the first frame and time of laser introduction 
is 2 microseconds (it's actually slightly less at 1800ns, because the exposure time was 200 ns). This is why there is
 a larger variation in Z location within the reactant core, as the spark is advected some distance axially before we 
 capture a frame. Unfortunately, we cannot sync the camera to the firing of the laser, as the laser does not operate 
 with a clock in this configuration. However, the newest dataset has an included time between the laser pulse and the 
 start of the first frame (accurate down to 286 ns). This means while we can't perfectly sync the camera to the laser, 
 we can at least measure the time of the first frame.

The size of the kernel is around 3mm in diameter. For the dataset you are using, the kernel appears as an entirely white
 (saturated) region. I isolated this region and calculated the centroid to provide the center of the spark location. 
 Most (if not all) of the deviation between the nominal and measured location in the X direction is due to us not being 
 perfectly centered at the time of testing. This is why there are large deviations between the nominal values, but 
 small deviations between measured values (low accuracy, high precision). To be clear, the low accuracy is corrected 
 by measuring the spark location using the image, so that "accuracy" is just low when compared to the nominal value. 
 The true uncertainty of the measured spark location is less than 120 um.

In the newest dataset (now on Dropbox, with more repeats and overall improvements to data quality), we have much 
stronger control over spark location. There will still be deviation due to the flow velocity and changing refractive 
index through the flow field. I have not yet measured the spark location of each of the newest tests, but I plan on 
using a very similar method as before. In general, it will still be difficult to get an accurate measurement of the 
spark within the central jet due to the high shear and velocity.

=====================

As a result, it is not possible to compensate for the measured offset in a consistent way, since it is a mixture of
advection and laser alignment. So I think it is best to instead just go ahead with nominal values and an isotropic
uncertainty of 0.15mm.

"""

def spark_x_offset_params(x: float, z: float) -> dict:
    """Return the mean and standard deviation [mm] of the Gaussian distribution modeling the spark x offset
    """
    #query_coord = np.array([x, z])
    #mean, std = spark_model.get_offset_params(query_coord)
    #assert std > 1e-6, f'Interpolated std is non-positive for query {query_coord}: {std}'

    mean = 0.0
    std = 0.15

    return {'loc': mean, 'scale': std}  # Marginalize out z, params do not change as we model with isotropic Gaussian


def spark_y_offset_params() -> dict:
    """We assume a consant spark y offset, as we have no observed data to suggest otherwise
    """
    return {'loc': 0.0, 'scale': 0.15}


def spark_z_offset_params(x: float, z: float) -> dict:
    """Return the mean and standard deviation [mm] of the Gaussian distribution modeling the spark z offset
    """
    #query_coord = np.array([x, z])
    #mean, std = spark_model.get_offset_params(query_coord)

    mean = 0.0
    std = 0.15

    return {'loc': mean, 'scale': std}  # Marginalize out x, params do not change


# Utility functions

def load_grouped_spark_data():
    # Load spark data from test summary spreadsheet, grouped by nominal location
    # Return a dict of (nominal position) -> (list of offsets)
    sheet_path = 'Test Summary Sheet 20211220 - Sheet1.csv'
    df = pd.read_csv(sheet_path, header=0)
    nominal_cols = ['Spark X Location [mm]', 'Spark Z Location [mm]']
    measured_cols = ['Spark X Measured [mm]', 'Spark Z Measured [mm]']

    for col in nominal_cols + measured_cols:
        assert col in df.columns, f'Column {col} not found in {sheet_path}'

    # Load all data into float32 arrays, round to nearest 0.01mm

    nominal_data = df[nominal_cols].values.astype(np.float32)
    measured_data = df[measured_cols].values.astype(np.float32)

    assert len(nominal_data) == len(measured_data), 'Mismatch in number of rows'

    nominal_data = np.round(nominal_data, 2)
    measured_data = np.round(measured_data, 2)

    # We are interested in the offset from nominal
    measured_data -= nominal_data

    # Group by nominal location
    grouped_data = defaultdict(list)

    for i in range(nominal_data.shape[0]):
        nominal = tuple(nominal_data[i])
        measured = tuple(measured_data[i])
        grouped_data[nominal].append(measured)

    return grouped_data


def fit_gaussian_to_data(grouped_data: dict) -> dict:
    # Fit gaussian to each nominal location
    # Return dict of nominal -> (mean, std)
    nominal_to_params = {}
    for nominal, data in grouped_data.items():
        mean, std = fit_isotropic_gaussian(data)
        #std = max(std, 0.1)  # Ensure std is at least 0.1mm
        nominal_to_params[nominal] = (mean, std)
    return nominal_to_params


def fit_isotropic_gaussian(coords):
    coords = np.array(coords)
    mean = np.mean(coords, axis=0)
    distances = np.linalg.norm(coords - mean, axis=1)
    std = np.std(distances, ddof=1)
    return mean, std


def create_rbf_interpolator(nominal_to_params: dict):
    # Extract nominal positions and parameters
    nominal_positions = np.array(list(nominal_to_params.keys()))
    means = np.array([params[0] for params in nominal_to_params.values()])
    stds = np.array([params[1] for params in nominal_to_params.values()])

    for std in stds:
        assert std > 1e-6

    # Separate the nominal positions into x and y components
    x = nominal_positions[:, 0]
    y = nominal_positions[:, 1]

    # Plot x, y, std data
    #fig, ax = plt.subplots()
    #ax.scatter(x, y, c=stds, cmap='viridis')
    #ax.set_xlabel('Spark X Location [mm]')
    #ax.set_ylabel('Spark Z Location [mm]')
    #ax.set_title('Spark Offset Standard Deviation [mm]')
    #ax.set_aspect('equal')
    #plt.colorbar(ax.scatter(x, y, c=stds, cmap='viridis'), label='Standard Deviation [mm]')
    #plt.show()

    # Create RBF interpolators for means and stds
    mean_interpolators = [Rbf(x, y, means[:, i], function='multiquadric', epsilon=2) for i in range(means.shape[1])]
    std_interpolator = Rbf(x, y, stds, function='multiquadric', epsilon=2)

    def rbf_mean(query_coords):
        qx, qy = query_coords[0]
        return np.array([interp(qx, qy) for interp in mean_interpolators])

    def rbf_std(query_coords):
        qx, qy = query_coords[0]
        return std_interpolator(qx, qy)

    return rbf_mean, rbf_std

class SparkOffsetModel:
    def __init__(self):
        self.grouped_data = load_grouped_spark_data()
        self.nominal_to_params = fit_gaussian_to_data(self.grouped_data)
        #self.interp_mean, self.interp_std = create_interpolator(self.nominal_to_params)
        #self.nearest_mean, self.nearest_std = create_nearest_interpolator(self.nominal_to_params)
        self.interp_mean, self.interp_std = create_rbf_interpolator(self.nominal_to_params)


    def get_offset_params(self, query_coord: np.ndarray):
        mean = self.interp_mean([query_coord])
        std = self.interp_std([query_coord])

        #if np.isnan(mean).any() or np.isnan(std):
        #    mean = self.nearest_mean([query_coord])
        #    std = self.nearest_std([query_coord])

        assert not np.isnan(mean).any(), f'Interpolated mean is NaN for query {query_coord}'
        assert not np.isnan(std), f'Interpolated std is NaN for query {query_coord}'
        assert std > 1e-6, f'Interpolated std is non-positive for query {query_coord}'

        return mean, std

spark_model = SparkOffsetModel()  # Construct once at file scope and use for all queries


# Unit tests for the SparkOffsetModel class

class TestSparkOffsetModel(unittest.TestCase):

    def setUp(self):
        self.spark_model = SparkOffsetModel()

    def test_spark_x_offset_params(self):
        result = spark_x_offset_params(3, 15)
        self.assertIn('loc', result)
        self.assertIn('scale', result)
        self.assertTrue(np.abs(result['loc']) < 1)
        self.assertTrue(np.abs(result['scale']) < 1)

    def test_spark_y_offset_params(self):
        result = spark_y_offset_params()
        self.assertIn('loc', result)
        self.assertIn('scale', result)
        self.assertEqual(result['loc'], 0.0)
        self.assertEqual(result['scale'], 0.1)

    def test_spark_z_offset_params(self):
        result = spark_z_offset_params(3, 15)
        self.assertIn('loc', result)
        self.assertIn('scale', result)
        self.assertTrue(np.abs(result['loc']) < 1)
        self.assertTrue(np.abs(result['scale']) < 1)

    def test_load_grouped_spark_data(self):
        grouped_data = load_grouped_spark_data()
        self.assertIsInstance(grouped_data, dict)
        for key, value in grouped_data.items():
            self.assertIsInstance(key, tuple)
            self.assertIsInstance(value, list)

    def test_fit_gaussian_to_data(self):
        grouped_data = load_grouped_spark_data()
        nominal_to_params = fit_gaussian_to_data(grouped_data)
        self.assertIsInstance(nominal_to_params, dict)
        for key, value in nominal_to_params.items():
            self.assertIsInstance(key, tuple)
            self.assertIsInstance(value, tuple)
            self.assertEqual(len(value), 2)
            self.assertIsInstance(value[0], np.ndarray)

    def test_fit_isotropic_gaussian(self):
        coords = [(1, 2), (2, 3), (3, 4), (4, 5)]
        mean, std = fit_isotropic_gaussian(coords)
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(std, float)

    def test_create_interpolator(self):
        grouped_data = load_grouped_spark_data()
        nominal_to_params = fit_gaussian_to_data(grouped_data)
        mean_interpolator, std_interpolator = create_interpolator(nominal_to_params)
        self.assertIsNotNone(mean_interpolator)
        self.assertIsNotNone(std_interpolator)

    def test_plot_fitted_gaussians(self):
        grouped_data = load_grouped_spark_data()
        nominal_to_params = fit_gaussian_to_data(grouped_data)

        fig, ax = plt.subplots()
        for nominal, (mean, std) in nominal_to_params.items():
            ax.plot(nominal[0], nominal[1], 'bo')  # Plot nominal points
            ax.plot(nominal[0] + mean[0], nominal[1] + mean[1], 'ro')
            # Line connecting nominal to mean
            ax.plot([nominal[0], nominal[0] + mean[0]], [nominal[1], nominal[1] + mean[1]], 'r-')
            circle = plt.Circle((nominal[0] + mean[0], nominal[1] + mean[1]), std, color='black', fill=False)  # Plot circle with radius std

            # Add label with std
            ax.text(nominal[0] + mean[0], nominal[1] + mean[1], f'{std:.2f}', fontsize=8)

            ax.add_patch(circle)
            circle = plt.Circle((nominal[0] + mean[0], nominal[1] + mean[1]), 2*std, color='black', fill=False)  # Plot circle with radius std
            ax.add_patch(circle)
            circle = plt.Circle((nominal[0] + mean[0], nominal[1] + mean[1]), 3*std, color='black', fill=False)  # Plot circle with radius std
            ax.add_patch(circle)

        ax.set_xlabel('Spark X Location [mm]')
        ax.set_ylabel('Spark Z Location [mm]')
        ax.set_title('Fitted Gaussians for Nominal Data')
        ax.set_aspect('equal')
        ax.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()