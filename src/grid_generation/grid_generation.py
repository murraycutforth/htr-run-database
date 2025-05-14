##############################################################################
# Grid Generation Module
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path


def sigmoid(xi, c, Lb, L):
    """Sigmoid function transitioning from combustor (xi=0) to nozzle (xi=1)"""
    sig = 0.5 * (1 + np.tanh(c * (xi - Lb / L)))
    return sig

def gaussian(xi, c, Lb, L):
    """Gaussian function centered on transition region"""
    F = np.exp(-c**2 / 4 * (xi - 0.9 * Lb / L)**2)
    return F

def squircle_y(y, z, d):
    """Squircle transform for y coordinate"""
    R = y[None, :]**2 + z[:, None]**2
    return y[None, :] * np.sqrt(R - d * (y[None, :] * z[:, None])**2) / np.maximum(np.sqrt(R), 1e-60)

def squircle_z(y, z, d):
    """Squircle transform for z coordinate"""
    R = y[None, :]**2 + z[:, None]**2
    return z[:, None] * np.sqrt(R - d * (y[None, :] * z[:, None])**2) / np.maximum(np.sqrt(R), 1e-60)

def stretch_coordinate_x(xi, a):
    """Stretch x coordinate using tanh function"""
    xihat = 1 - np.tanh(a * (1 - xi)) / np.tanh(a)
    # Normalize to [0,1]
    xihat = (xihat - xihat[0]) / (xihat[-1] - xihat[0])
    return xihat

def stretch_coordinate_yz(eta, zet, b, radial_stretch_type, e1=0, e2=1):
    """Stretch y and z coordinates using specified function"""
    if radial_stretch_type == 'cosh':
        etahat = np.sinh(b * eta) / np.sinh(b)
        zethat = np.sinh(b * zet) / np.sinh(b)
    elif radial_stretch_type == 'tanh':
        etahat = ((e2 - 1) * np.log(np.cosh(b * (e1 - eta))) - 
                  (e2 - 1) * np.log(np.cosh(b * (e1 + eta))) + 
                  2 * b * e2 * eta) / (2.0 * b)
        etahat /= ((e2 - 1) * np.log(np.cosh(b * (e1 - 1.0))) - 
                   (e2 - 1) * np.log(np.cosh(b * (e1 + 1.0))) + 
                   2 * b * e2 * 1.0) / (2.0 * b)
        
        zethat = ((e2 - 1) * np.log(np.cosh(b * (e1 - zet))) - 
                  (e2 - 1) * np.log(np.cosh(b * (e1 + zet))) + 
                  2 * b * e2 * zet) / (2.0 * b)
        zethat /= ((e2 - 1) * np.log(np.cosh(b * (e1 - 1.0))) - 
                   (e2 - 1) * np.log(np.cosh(b * (e1 + 1.0))) + 
                   2 * b * e2 * 1.0) / (2.0 * b)
    else:
        raise ValueError(f'Unrecognized radial_stretch_type: {radial_stretch_type}')
    
    # Normalize to [-1,1]
    etahat = -1.0 + 2 * (etahat - etahat[0]) / (etahat[-1] - etahat[0])
    zethat = -1.0 + 2 * (zethat - zethat[0]) / (zethat[-1] - zethat[0])
    
    return etahat, zethat

def calculate_cell_sizes(x, y, z):
    """Calculate cell sizes and report statistics"""
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dz = np.zeros_like(z)
    
    dx[:, :, 0] = x[:, :, 1] - x[:, :, 0]
    dx[:, :, 1:-1] = 0.5 * (x[:, :, 2:] - x[:, :, :-2])
    dx[:, :, -1] = x[:, :, -1] - x[:, :, -2]
    
    dy[:, 0, :] = y[:, 1, :] - y[:, 0, :]
    dy[:, 1:-1, :] = 0.5 * (y[:, 2:, :] - y[:, :-2, :])
    dy[:, -1, :] = y[:, -1, :] - y[:, -2, :]
    
    dz[0, :, :] = z[1, :, :] - z[0, :, :]
    dz[1:-1, :, :] = 0.5 * (z[2:, :, :] - z[:-2, :, :])
    dz[-1, :, :] = z[-1, :, :] - z[-2, :, :]
    
    dcell = pow(dx * dy * dz, 1.0 / 3.0)
    
    cell_stats = {
        'min_dx': np.min(dx) * 1e6,
        'max_dx': np.max(dx) * 1e6,
        'min_dy': np.min(dy) * 1e6,
        'max_dy': np.max(dy) * 1e6,
        'min_dz': np.min(dz) * 1e6,
        'max_dz': np.max(dz) * 1e6,
        'min_dcell': np.min(dcell) * 1e6,
        'max_dcell': np.max(dcell) * 1e6
    }
    
    return dx, dy, dz, dcell, cell_stats

def calculate_stretch(xihat, etahat):
    """Calculate stretching factors for verification"""
    d_xihat = np.diff(xihat)
    stretch_xihat = np.diff(d_xihat) / d_xihat[0:-1]
    
    d_etahat = np.diff(etahat)
    stretch_etahat = np.diff(d_etahat) / d_etahat[0:-1]
    
    return np.max(np.abs(stretch_xihat)) * 100, np.max(np.abs(stretch_etahat)) * 100

def generate_node_coordinates(xi, eta, zet, params):
    """Generate node coordinates based on given parameters and stretching functions"""
    # Extract parameters
    Ln = params['Ln']
    Lb = params['Lb']
    L = Ln + Lb
    Rb = params['Rb']
    Rn = params['Rn']
    A = params['A']
    a = params['a']
    b = params['b']
    c = params['c']
    d = params['d']
    e1 = params['e1']
    e2 = params['e2']
    f1 = params['f1']
    f2 = params['f2']
    radial_stretch_type = params['radial_stretch_type']
    
    # Apply stretching
    xihat = stretch_coordinate_x(xi, a)
    etahat, zethat = stretch_coordinate_yz(eta, zet, b, radial_stretch_type, e1, e2)
    
    # Compute x coordinate
    x = L * xihat[None, None, :] + A * gaussian(xihat[None, None, :], c, Lb, L) * (etahat[None, :, None]**2 + zethat[:, None, None]**2)
    
    # Transition functions
    sx = sigmoid(x / L, c, Lb, L)
    sx2 = sigmoid(f1 * x / L + f2, c, Lb, L)
    
    # Compute y and z coordinates
    y = (Rb * (1 - sx) + Rn * sx) * (squircle_y(etahat, zethat, d)[:, :, None] * (1 - sx2) + eta[None, :, None] * sx2)
    z = (Rb * (1 - sx) + Rn * sx) * (squircle_z(etahat, zethat, d)[:, :, None] * (1 - sx2) + zet[:, None, None] * sx2)
    
    # Calculate stretch statistics
    x_stretch, yz_stretch = calculate_stretch(xihat, etahat)
    print(f'Max x stretch is {x_stretch:.2f} %')
    print(f'Max y/z stretch is {yz_stretch:.2f} %')
    
    # Calculate cell sizes
    _, _, _, _, cell_stats = calculate_cell_sizes(x, y, z)
    for key, value in cell_stats.items():
        print(f'{key.replace("_", " ")} = {value:.2f} microns')
    
    return x, y, z

def plot_grid_spacing(x, y, z, figwidth=10, fac_fig=1.0, cm=0.393701):
    """Plot grid spacing in different directions"""
    # Plot the stretched coordinates (X)
    current_width = figwidth * cm * fac_fig
    fig = plt.figure(1, figsize=[1.0 * current_width, 0.82 * current_width])
    plt.clf()
    Nplt_x = 5
    
    # Create a colormap for x
    cmap_x = plt.cm.viridis
    norm_x = plt.Normalize(0, int((x.shape[1] - 1)))
    
    for i in range(Nplt_x):
        iy = i * int((x.shape[1] - 1) / (Nplt_x - 1))
        x_plot = x[0, iy, :]
        delta_x = np.diff(x_plot)
        plt.plot(x_plot[0:-1] * 1e3, delta_x * 1e6, color=cmap_x(norm_x(iy)))
    
    plt.xlabel('x [mm]')
    plt.ylabel('$\Delta x$ [$\mu$m]')
    #plt.xlim([0, 130])
    #plt.ylim([0, 800])
    
    # Add colorbar
    sm_x = plt.cm.ScalarMappable(cmap=cmap_x, norm=norm_x)
    sm_x.set_array([])
    cbar = plt.colorbar(sm_x, ax=plt.gca())
    cbar.set_label('y index')
    
    ax = plt.gca()
    chartBox = ax.get_position()
    ax.set_position([0.16, 0.13, chartBox.width * 0.85, chartBox.height * 1.02])

    plt.show()
    
    # Plot the stretched coordinates (Y)
    fig = plt.figure(2, figsize=[1.0 * current_width, 0.82 * current_width])
    plt.clf()
    Nplt_y = 50
    
    # Create a colormap for y
    cmap_y = plt.cm.viridis
    norm_y = plt.Normalize(0, y.shape[2]-1)
    
    for i in range(Nplt_y):
        iz = 0
        ix = i * int((y.shape[2] - 1) / (Nplt_y - 1))
        y_plot = y[iz, :, ix]
        delta_y = np.diff(y_plot)
        plt.plot(y_plot[0:-1] * 1e3, delta_y * 1e6, color=cmap_y(norm_y(ix)))
    
    plt.xlabel('y [mm]')
    plt.ylabel('$\Delta y$ [$\mu$m]')
    plt.xlim([-30, 30])
    plt.ylim([0, 500])
    
    # Add colorbar
    sm_y = plt.cm.ScalarMappable(cmap=cmap_y, norm=norm_y)
    sm_y.set_array([])
    cbar = plt.colorbar(sm_y, ax=plt.gca())
    cbar.set_label('x index')
    
    ax = plt.gca()
    chartBox = ax.get_position()
    ax.set_position([0.16, 0.13, chartBox.width * 0.85, chartBox.height * 1.02])

    plt.show()
    

def plot_grid_2d(x, y, z, xNum, yNum, zNum, figwidth=10, fac_fig=1.0, cm=0.393701):
    """Plot 2D views of the grid"""
    # Show the grid in center XY plane
    current_width = 2 * figwidth * cm * fac_fig
    fig = plt.figure(3, figsize=[1.0 * current_width, 0.90 * current_width])
    plt.clf()
    iz = int(zNum / 2.0)
    grid_skip = 1
    xiter = np.arange(0, xNum, grid_skip)
    yiter = np.arange(0, yNum, grid_skip)
    grid_skip_zoom = 1
    xiter_zoom = np.arange(0, xNum, grid_skip_zoom)
    yiter_zoom = np.arange(0, yNum, grid_skip_zoom)
    
    # Global
    ax1 = plt.subplot(211)
    for j in yiter:
        ax1.plot(x[iz, j, :], y[iz, j, :], 'b-', linewidth=0.3)
    for i in xiter:
        ax1.plot(x[iz, :, i], y[iz, :, i], 'b-', linewidth=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Zoom 1 and 2
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)
    for j in yiter_zoom:
        ax2.plot(x[iz, j, :], y[iz, j, :], 'b-', linewidth=0.3)
        ax3.plot(x[iz, j, :], y[iz, j, :], 'b-', linewidth=0.3)
    for i in xiter_zoom:
        ax2.plot(x[iz, :, i], y[iz, :, i], 'b-', linewidth=0.3)
        ax3.plot(x[iz, :, i], y[iz, :, i], 'b-', linewidth=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    
    ax2.set_xlim((0, 0.04))
    ax2.set_ylim((-0.02, 0.02))
    ax3.set_xlim((0.09, 0.13))
    ax3.set_ylim((-0.01, 0.03))
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    plt.tight_layout()
    plt.show()
    
    # Show the grid in center YZ plane
    current_width = 3 * figwidth * cm * fac_fig
    fig2 = plt.figure(4, figsize=[1.0 * current_width, 0.40 * current_width])
    plt.clf()
    ix = int(0.65 * xNum)
    grid_skip = 1
    ziter = np.arange(0, zNum, grid_skip)
    yiter = np.arange(0, yNum, grid_skip)
    
    # Inlet
    ax1 = plt.subplot(131)
    for j in yiter:
        ax1.plot(z[:, j, ix], y[:, j, ix], 'b-', linewidth=0.3)
    for k in ziter:
        ax1.plot(z[k, :, ix], y[k, :, ix], 'b-', linewidth=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Close to nozzle outlet
    ix = int(0.85 * xNum)
    ax2 = plt.subplot(132)
    for j in yiter:
        ax2.plot(z[:, j, ix], y[:, j, ix], 'b-', linewidth=0.3)
    for k in ziter:
        ax2.plot(z[k, :, ix], y[k, :, ix], 'b-', linewidth=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # Outlet
    ix = int(0.99 * xNum)
    ax3 = plt.subplot(133)
    for j in yiter:
        ax3.plot(z[:, j, ix], y[:, j, ix], 'b-', linewidth=0.3)
    for k in ziter:
        ax3.plot(z[k, :, ix], y[k, :, ix], 'b-', linewidth=0.3)
    ax3.set_aspect('equal', adjustable='box')

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

    plt.tight_layout()
    plt.show()


def generate_grid(config, show_plots=False):
    """Main function to generate the grid with the given configuration"""
    tic = time.time()
    
    # Extract parameters from config
    xNum = config["Grid"]["xNum"]
    yNum = config["Grid"]["yNum"]
    zNum = config["Grid"]["zNum"]
    Ln = config["Case"]["LengthNozzle"]
    Lb = config["Case"]["LengthBurner"]
    Rb = config["Case"]["RadiusBurner"]
    Rn = config["Case"]["RadiusNozzle"]
    R_F_out = config["Case"]["Radius_F_out"] # Outer radius of fuel co-flow
    LRef = R_F_out
    
    # Stretch parameters
    params = {
        'Ln': Ln,
        'Lb': Lb,
        'Rb': Rb,
        'Rn': Rn,
        'A': config["Case"]["nozzleStretch"] * Rn,
        'a': config["Case"]["xStretch"],
        'b': config["Case"]["yzStretch"],
        'c': config["Case"]["nozzleTransitionRate"],
        'd': config["Case"]["squircleStretch"],
        'e1': config["Case"]["yzStretch_position"],
        'e2': config["Case"]["yzStretch_ratio"],
        'f1': config["Case"]["xStretch_sigmoid1"],
        'f2': config["Case"]["xStretch_sigmoid2"],
        'radial_stretch_type': config["Case"]["radial_stretch_type"]
    }
    
    # Construct grid nodes
    xi = np.linspace(0, 1, xNum + 1)  # xi nodes
    eta = np.linspace(-1, 1, yNum + 1)  # eta nodes
    zet = np.linspace(-1, 1, zNum + 1)  # zet nodes
    
    # Generate coordinates
    x, y, z = generate_node_coordinates(xi, eta, zet, params)
    
    # Scale coordinates by reference length
    x_scaled = x / LRef
    y_scaled = y / LRef
    z_scaled = z / LRef
    
    # Combine scaled coordinates into nodes array
    shape = [zNum + 1, yNum + 1, xNum + 1]
    nodes = np.ndarray(shape, dtype=np.dtype('(3,)f8'))
    nodes[:, :, :, 0] = x_scaled
    nodes[:, :, :, 1] = y_scaled
    nodes[:, :, :, 2] = z_scaled
    
    # Display grid information
    xmin = np.min(nodes[:, :, :, 0])
    xmax = np.max(nodes[:, :, :, 0])
    ymin = np.min(nodes[:, :, :, 1])
    ymax = np.max(nodes[:, :, :, 1])
    zmin = np.min(nodes[:, :, :, 2])
    zmax = np.max(nodes[:, :, :, 2])
    
    print('shape(nodes) = {}'.format(np.shape(nodes)))
    print('x=[{:10.3e},{:10.3e}], y=[{:10.3e},{:10.3e}], z=[{:10.3e},{:10.3e}]'
         .format(xmin, xmax, ymin, ymax, zmin, zmax))
    print('Nodes generated. ({:.1f} s)'.format(time.time() - tic))
    
    # Plot grid if requested
    if show_plots:
        plot_grid_spacing(x, y, z)
        plot_grid_2d(x, y, z, xNum, yNum, zNum)
    
    return nodes

def save_grid(config, nodes):
    """Save the generated grid to disk"""

    import gridGen_new # Import here, so we can run other functions locally for debugging

    tic = time.time()
    
    # Define node access function for gridGen_new
    def nodes_f(lo, hi):
        return nodes[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0], :]
    
    # Save full node positions
    np.savez('00positions.npz', nodes=nodes)
    
    # Extract bounds
    xmin = np.min(nodes[:, :, :, 0])
    xmax = np.max(nodes[:, :, :, 0])
    ymin = np.min(nodes[:, :, :, 1])
    ymax = np.max(nodes[:, :, :, 1])
    zmin = np.min(nodes[:, :, :, 2])
    zmax = np.max(nodes[:, :, :, 2])
    
    # Generate cell centers (assuming gridGen_new is available)
    
    xGrid, yGrid, zGrid, dx, dy, dz = gridGen_new.getCellCenters(
        config, 
        nodes=nodes_f,
        xmin=xmin, xmax=xmax,
        ymin=ymin, ymax=ymax,
        zmin=zmin, zmax=zmax,
        physical_nodes=nodes
    )  # This writes to grid directory
    
    # Save cell centers
    cc = np.stack((xGrid, yGrid, zGrid), axis=-1)
    np.savez('00cellCentres.npz', arr_0=cc)
    
    print('Xgrid.shape:', xGrid.shape)
    print('nodes.shape:', nodes.shape)
    print('Grid written to {}  ({:.1f} s)\n'.format(
        config["Grid"]["GridInput"]["gridDir"], time.time() - tic))
    

def main(config_file):
    """Main function to run the grid generation process"""
    assert Path(config_file).exists(), f"Configuration file {config_file} does not exist."

    print(f"Loading configuration from {config_file}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    print("Generating grid, with show_plots=True")
    
    _ = generate_grid(config, show_plots=True)



if __name__ == "__main__":
    config_path = Path(__file__).parent / "base_coarse_Tony.json"
    main(config_path)