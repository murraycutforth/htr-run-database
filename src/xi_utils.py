import json
from pathlib import Path

LREF = 0.003175


def extract_xi_from_json(json_file_path: str or Path) -> list:
    """
    Extract xi vector from a HTR combustor configuration JSON file.

    Args:
        json_file_path: Path to the JSON configuration file

    Returns:
        List containing the extracted xi vector with 17 elements
    """
    # Load the JSON configuration
    with open(json_file_path, 'r') as f:
        config = json.load(f)

    # Extract xi values in the correct order
    xi = [0] * 17  # Initialize with zeros

    # Extract laser focal location (convert from dimensionless to mm)
    xi[0] = config['Flow']['laser']['focalLocation'][1] * LREF * 1000  # radial (mm)
    xi[1] = config['Flow']['laser']['focalLocation'][2] * LREF * 1000  # azimuthal (mm)
    xi[2] = config['Flow']['laser']['focalLocation'][0] * LREF * 1000  # streamwise (mm)

    # Extract axial length
    xi[3] = config['Flow']['laser']['axialLength']

    # Calculate beam width parameters
    near_radius = config['Flow']['laser']['nearRadius']
    far_radius = config['Flow']['laser']['farRadius']
    xi[4] = xi[3] / (2 * near_radius)  # axial_l_i / (2 * near_radius)
    xi[5] = near_radius / far_radius  # beta

    # Extract energy deposited
    xi[6] = config['Flow']['laser']['peakEdotPerMass']

    # Extract FWHM
    xi[7] = config['Flow']['laser']['pulseFWHM']

    # Extract thickened flame parameters
    xi[8] = config['Flow']['TFModel']['Efficiency']['beta']
    xi[9] = config['Flow']['TFModel']['Efficiency']['sL0']
    xi[10] = config['Flow']['TFModel']['Efficiency']['Arr_factor']

    # Extract Smagorinsky constant
    xi[11] = config['Flow']['sgsModel']['TurbViscModel']['C_S']

    # Extract mass flow rates
    xi[12] = config['BC']['xBCLeft']['mDot1']
    xi[13] = config['BC']['xBCLeft']['mDot2']

    # Extract time offset from zone1
    xi[14] = config['Integrator']['TimeStep']['zone1'] - 1000
    assert xi[14] in [1000, 2000, 3000, 4000, 5000, 6000]

    # Extract restart directory (methane content)
    restart_dir = config['Flow']['initCase']['restartDir']
    xi[15] = restart_dir.split('/')[-1]  # Extract just the restart folder name

    METH_RESTARTS = ['fluid_iter0000040000', 'fluid_iter0000060000', 'fluid_iter0000080000']
    assert xi[15] in METH_RESTARTS, f"Restart directory {xi[15]} not in expected values: {METH_RESTARTS}"

    # Extract squirc parameter from common case directory
    # This is trickier as we don't directly store it, need to infer from path
    bc_dir = Path(config['BC']['xBCLeft']['MixtureProfile']['FileDir'])
    if isinstance(bc_dir, str):
        bc_dir = Path(bc_dir)

    # Assuming the structure follows the pattern in get_path_to_common_case
    xi[16] = bc_dir.parent.name  # This should be the squirc value
    assert xi[16] in ["S_0p90", "S_0p80", "S_0p70"]

    return xi