"""
Scattering & structure-factor utilities with MC-DFM.

This module converts HOOMD GSD trajectories into SAXS-like outputs:
- Extract lattice coordinates (positions + Euler angles) from GSD
- Build building-block grids (sphere points)
- Run pairwise scattering simulator to compute I(q)
- Convert I(q) to S(q) using a sphere form factor
- Save per-frame and averaged curves, with the same plots as the original script
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gsd.hoomd
from Scattering_Simulator import pairwise_method
from scipy.spatial.transform import Rotation as R

# ======== Extraction utilities ========

def extract_positions_orientations(filename):
    """
    Extract positions and orientations from each frame of a GSD file.

    Parameters
    ----------
    filename : str
        Path to the GSD trajectory file.

    Returns
    -------
    positions : list of np.ndarray
        List of arrays of shape (N, 3) with particle positions per frame.
    orientations : list of np.ndarray or None
        List of arrays of shape (N, 4) with quaternions per frame,
        or None if not present in the frame.
    """
    traj = gsd.hoomd.open(name=filename, mode='r')

    positions = []
    orientations = []
    for frame in traj:
        positions.append(frame.particles.position.copy())
        if hasattr(frame.particles, 'orientation'):
            orientations.append(frame.particles.orientation.copy())
        else:
            orientations.append(None)

    orientations_euler = []
    for i in range(len(orientations)):
        angles = quaternion_to_euler(orientations[i][:, :])
        orientations_euler.append(angles)

    lattice_coordinates = []
    for i in range(len(orientations_euler)):
        lattice_coordinates.append(np.hstack((positions[i][:, :], orientations_euler[i][:, :])))
    return lattice_coordinates

def quaternion_to_euler(quat, degrees=True, order='xyz'):
    """
    Convert a quaternion (HOOMD format: [qw, qx, qy, qz]) to Euler angles.

    Parameters:
    - quat: array-like, quaternion [qw, qx, qy, qz]
    - degrees: bool, return angles in degrees if True (default), radians if False
    - order: str, axes sequence for Euler angles ('xyz', 'zyx', etc.)

    Returns:
    - tuple of 3 Euler angles (angle_x, angle_y, angle_z)
    """
    scipy_quat = quat
    r = R.from_quat(scipy_quat)
    angles = r.as_euler(order, degrees=degrees)
    return angles

def grid_points_in_sphere(D, spacing):
    """
    Generate a regular 3D grid of points spaced by 'spacing' that fit inside a sphere.

    Parameters
    ----------
    D : float
        Diameter of the sphere.
    spacing : float
        Distance between adjacent grid points.

    Returns
    -------
    points : np.ndarray of shape (M, 3)
        Grid points inside the sphere.
    """
    radius = D / 2.0
    r2 = radius ** 2
    coords = np.arange(-radius, radius + spacing, spacing)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    mask = np.sum(grid**2, axis=1) <= r2
    points = grid[mask]
    return points

def fill_cube_with_points(N, edge_length):
    """
    Fill a cube with N points approximately uniformly distributed.

    Parameters
    ----------
    N : int
        Number of points to place inside the cube.
    edge_length : float
        Length of each edge of the cube.

    Returns
    -------
    points : np.ndarray of shape (N, 3)
        Coordinates of the points inside the cube.
    """
    n_side = int(np.ceil(N ** (1/3)))
    spacing = edge_length / n_side
    coords = np.linspace(-edge_length/2 + spacing/2, edge_length/2 - spacing/2, n_side)
    X, Y, Z = np.meshgrid(coords, coords, coords)
    grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    points = grid[:N]
    return points

# ======== SAXS helpers ========

def sphere(q, r):
    """Monodisperse sphere form factor used in your pipeline."""
    return 3*(np.sin(q*r) - q*r*np.cos(q*r))**2/(q*r)**6  # :contentReference[oaicite:4]{index=4}

def convert_data(data, model):
    """
    Resample `model` onto q-grid of `data` by nearest match (as in original).
    Both inputs are (N,2): [q, I].
    """
    model_x = model[:,0]
    model_y = model[:,1]
    index = np.linspace(0, len(model_x)-1, len(model_x))
    model_q_new, model_I_new = [], []
    for i in range(len(data)):
        data_q = data[i,0]
        array = np.abs(model_x - data_q)
        array = np.hstack((array.reshape(-1,1), index.reshape(-1,1)))
        array = array[np.argsort(array[:, 0])]
        loc = int(array[0,1])
        model_q_new.append(model_x[loc])
        model_I_new.append(model_y[loc])
    q = np.array(model_q_new).reshape(-1,1)
    I = np.array(model_I_new).reshape(-1,1)
    new_model_data = np.hstack((q, I))
    return new_model_data

def normalize_scattering_curves(q, I1, I2, q_min, q_max):
    """
    Normalizes the second scattering curve to match the first within the q-range [q_min, q_max].

    Parameters:
        data (np.ndarray): 2D array with shape (N, 3), where
                           - column 0: q values,
                           - column 1: intensity of curve 1,
                           - column 2: intensity of curve 2
        q_min (float): lower bound of q-range for normalization
        q_max (float): upper bound of q-range for normalization

    Returns:
        np.ndarray: New array with same shape as input, with column 2 normalized
    """
    mask = (q >= q_min) & (q <= q_max)
    if not np.any(mask):
        raise ValueError("No data points found within the specified q-range.")
    scale_factor = np.mean(I1[mask]) / np.mean(I2[mask])
    return I2 * scale_factor  # :contentReference[oaicite:6]{index=6}

def calculate_structure_factor(data0, data2, q_min, q_max, plot):
    """
    Compute S(q) = I_sim(q) / I_sphere(q) by:
      1) Resampling sim curve onto sphere q-grid (convert_data)
      2) Normalizing sphere intensities to match mean in [q_min, q_max]
      3) Dividing resampled sim by normalized sphere curve
    Returns (N,2) array [q, S(q)].
    """
    new_data_2 = convert_data(data0, data2)
    fig, ax = plt.subplots(figsize=(7,7))
    normalized_data = normalize_scattering_curves(new_data_2[:,0], new_data_2[:,1], data0[:,1], q_min, q_max)
    plt.scatter(data0[:,0], normalized_data, color='blue', label='data0')
    plt.scatter(new_data_2[:,0], new_data_2[:,1], color='red',  label='data2')
    plt.xscale('log'); plt.yscale('log')
    plt.ylabel('Intensity (arb. unit)'); plt.xlabel('q ($\\AA^{-1}$)')
    if plot is False:
        plt.close()
    structure_factor = new_data_2[:,1] / normalized_data
    structure_factor = np.hstack((new_data_2[:,0].reshape(-1,1), structure_factor.reshape(-1,1)))
    return structure_factor  # :contentReference[oaicite:7]{index=7}

# ======== End-to-end: GSD → I(q) → S(q) ========

def convert_to_SAXS(save_dir):
    """
    From a GSD in `save_dir`, compute scattering curves and structure factors
    for the last few frames, save per-frame results and an average curve.
    Produces the same .npy and .png artifacts as the original.
    """
    filenames = sorted(os.listdir(save_dir))
    file_path = save_dir + '/' + filenames[0]  # first file is the .gsd
    lattice_coordinates = extract_positions_orientations(file_path)

    # Sphere building block
    D = 1
    spacing = 0.03
    points = grid_points_in_sphere(D, spacing)
    ones = np.array([1]*len(points))
    points = np.hstack((points, ones.reshape(-1,1)))

    # -------- inputs (preserved) --------
    plot_every_n_frame = 50
    histogram_bins = 10000
    q = np.geomspace(0.8, 20, 2000)
    path = save_dir + '/scattering_data/'
    path_intensity = save_dir + '/scattering_data_intensity/'
    # ------------------------------------

    os.makedirs(path, exist_ok=True)
    os.makedirs(path_intensity, exist_ok=True)

    frames = len(lattice_coordinates)
    cmap = cm.jet
    norm = plt.Normalize(0, frames - 1)
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(10,7))

    for i in range(1, 6):
        n_samples = 10_000_000
        simulator = pairwise_method.scattering_simulator(n_samples)
        simulator.sample_building_block(points)
        simulator.sample_lattice_coordinates(lattice_coordinates[-i])
        simulator.calculate_structure_coordinates()

        I_q = simulator.simulate_multiple_scattering_curves_lattice_coords(
            points, lattice_coordinates[i], histogram_bins, q, save=False
        ).cpu().numpy()
        I_q = np.mean(I_q, axis=1)

        plt.rcParams.update({'font.size': 18})
        q_rescaled = q / 260
        ax.plot(q_rescaled, I_q, color='k', linewidth=3, label='scattering curve')
        ax.set_yscale('log'); ax.set_xscale('log')
        ax.set_ylabel('S(q)'); ax.set_xlabel('q ($\\AA^{-1}$)')

        data = np.hstack((q_rescaled.reshape(-1,1), I_q.reshape(-1,1)))
        np.save(path_intensity + f'Intensity_plot{-i}.npy', data)
        plt.savefig(path_intensity + f'Intensity_plot{-i}.png', dpi=600, bbox_inches="tight")

        # Convert to structure factor with a monodisperse sphere model
        q_sphere = np.geomspace(0.003, 0.08, 500)
        I_sphere = sphere(q_sphere, 130)
        monodisperse_sphere = np.hstack((q_sphere.reshape(-1,1), I_sphere.reshape(-1,1)))

        S_q = calculate_structure_factor(
            monodisperse_sphere,
            np.hstack((q_rescaled.reshape(-1,1), I_q.reshape(-1,1))),
            0.02, 0.03, False
        )

        fig2, ax2 = plt.subplots(figsize=(10,7))
        ax2.plot(S_q[:,0], S_q[:,1], color='k', linewidth=3, label='Structure Factor')
        ax2.set_yscale('log'); ax2.set_xscale('log')
        ax2.set_ylabel('S(q)'); ax2.set_xlabel('q ($\\AA^{-1}$)')
        np.save(path + f'structure_factor{-i}.npy', S_q)
        plt.savefig(path + f'structure_factor_plot{-i}.png', dpi=600, bbox_inches="tight")
        plt.close(fig2)

        if i == 1:
            all_Sq = S_q[:,1].reshape(-1,1)
        else:
            all_Sq = np.hstack((all_Sq, S_q[:,1].reshape(-1,1)))

    plt.close(fig)
    fig3, ax3 = plt.subplots(figsize=(10,7))
    ax3.plot(S_q[:,0], np.mean(all_Sq, axis=1), color='k', linewidth=3, label='average S(q)')
    ax3.set_yscale('log'); ax3.set_xscale('log')
    ax3.set_ylabel('Intensity (arb. unit)'); ax3.set_xlabel('q ($\\AA^{-1}$)')
    plt.savefig(path + 'scattering_curve_plot_average.png', dpi=600, bbox_inches="tight")
    plt.close(fig3)

    data = np.hstack((S_q[:,0].reshape(-1,1), np.mean(all_Sq, axis=1).reshape(-1,1)))
    np.save(path + 'average_structure_factor.npy', data)
