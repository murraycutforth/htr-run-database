import unittest
import os

from src.xi_utils import extract_xi_from_json, LREF

class TestXiUtils(unittest.TestCase):

    def setUp(self):
        # Get path to the test config JSON file
        self.config_file_path = os.path.join(os.path.dirname(__file__), "test_config.json")

    def test_extract_xi_from_json(self):
        # Extract xi from the test JSON file
        xi = extract_xi_from_json(self.config_file_path)

        # Assert xi has expected length
        self.assertEqual(len(xi), 17)

        # Check individual xi values
        # Laser focal location (converted from dimensionless to mm)
        self.assertAlmostEqual(xi[0], 2.5700948317054855 * LREF * 1000)  # radial
        self.assertAlmostEqual(xi[1], 0.0 * LREF * 1000)  # azimuthal
        self.assertAlmostEqual(xi[2], 4.152980309805737 * LREF * 1000)  # streamwise

        # Axial length and beam parameters
        self.assertAlmostEqual(xi[3], 0.6141869883022284)  # axial length
        self.assertAlmostEqual(xi[4], 0.6141869883022284 / (2 * 0.15222237954805365))  # axial_l_i / (2 * near_radius)
        self.assertAlmostEqual(xi[5], 0.15222237954805365 / 0.07977924032314787)  # near_radius / far_radius

        # Energy and pulse width
        self.assertAlmostEqual(xi[6], 439771.4208326538)  # peakEdotPerMass
        self.assertAlmostEqual(xi[7], 0.0010998646745878683)  # pulseFWHM

        # Thickened flame parameters
        self.assertAlmostEqual(xi[8], 0.5740661387564826)  # beta
        self.assertAlmostEqual(xi[9], 0.0100700916646296)  # sL0
        self.assertAlmostEqual(xi[10], 0.9568432682716905)  # Arr_factor

        # Smagorinsky constant
        self.assertAlmostEqual(xi[11], 0.16350268505459764)  # C_S

        # Mass flow rates
        self.assertAlmostEqual(xi[12], 1.8507681952878152)  # mDot1
        self.assertAlmostEqual(xi[13], 0.47141988228106296)  # mDot2

        # Time offset
        self.assertEqual(xi[14], 7000 - 1000)  # zone1 - 1000

        # Restart directory
        self.assertEqual(xi[15], "fluid_iter0000040000")  # Last part of restart path

        # Squirc parameter - should extract S_0p90 from the path
        self.assertEqual(xi[16], "S_0p90")  # Extracted from MixtureProfile FileDir


if __name__ == "__main__":
    unittest.main()