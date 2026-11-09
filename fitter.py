import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import logging

from sts_ladder_metrology.sensor_shape import SensorShape


logger = logging.getLogger(__name__)

# ---------------------------
# Residual function (cost function)
# ---------------------------
def residual_sensor_shape(params: np.ndarray, points: np.ndarray, size: tuple[float,float]) -> np.ndarray:
    """
    Compute distances from all points to closest rectangle edge for optimization.
    """
    cx, cy, cz, roll, pitch, yaw = params
    shape = SensorShape(size=size, center=(cx, cy, cz), angles=(roll, pitch, yaw))
    distances = np.array([shape.distance_to_edge(p) for p in points])

    return distances

# ---------------------------
# Wrapper cost function (scalar)
# ---------------------------
def cost_fn(params: np.ndarray, points: np.ndarray, size: tuple[float,float]) -> float:
    """
    Scalar cost: 0.5 * sum(distances^2)
    This is what optimizers that only accept a cost function expect.
    """
    resid = residual_sensor_shape(params, points, size)  # returns array of distances
    return 0.5 * np.sum(resid * resid)


def estimate_initial_angles(points: np.ndarray):
    """
    Estimate initial Euler angles for SensorShape from 3D points using PCA.
    Returns angles in radians in 'xyz' order.
    """
    # Center points
    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA: get principal axes
    U, S, Vt = np.linalg.svd(centered)
    axes = Vt.T  # columns are principal directions

    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    # Convert rotation matrix to Trait-Bryan
    rot = R.from_matrix(axes)
    angles = rot.as_euler('zyx', degrees=False)
    return angles

def make_callback(pts: np.ndarray, rect_size: tuple[float, float]):
    last = {'iter': 0}
    def cb(xk):
        last['iter'] += 1
        if last['iter'] % 50 == 0:
            logger.info("[minimize cb] iter %d cost=%.6e", last['iter'], cost_fn(xk, pts, rect_size))
    return cb

def fit_sensor_measurements(pts:np.ndarray, sensor_size:tuple[float,float], d_marker:float = 0.030) -> SensorShape:
    rect_size = sensor_size[0] - d_marker, sensor_size[1] - d_marker

    # Initial guess: center at mean of points, no rotation
    init_center = pts.mean(axis=0)
    init_angles = np.zeros(3)
    init_params = np.concatenate([init_center, init_angles])

    # Bounds determined by nominal distance from markers to edge
    margin = 0.1
    x_min, x_max = pts[:,0].min() - margin, pts[:,0].max() + margin
    y_min, y_max = pts[:,1].min() - margin, pts[:,1].max() + margin
    z_min, z_max = pts[:,2].min() - margin, pts[:,2].max() + margin
    bounds_lower = [x_min, y_min, z_min, -np.pi, -np.pi, -np.pi]
    bounds_upper = [x_max, y_max, z_max, np.pi, np.pi, np.pi]
    bounds = tuple(zip(bounds_lower, bounds_upper))

    # Optimize
    res_min = minimize(
        fun=lambda q: cost_fn(q, pts, rect_size),
        x0=init_params,
        method='L-BFGS-B',
        # bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-10},
        callback=make_callback(pts, rect_size),
    )
    fit_result = type("obj", (), {})()
    fit_result.x = res_min.x
    fit_result.optimizer = res_min  # raw result if you want more details


    # Construct optimized shape
    cx, cy, cz, roll, pitch, yaw = fit_result.x
    return SensorShape(size=rect_size, center=(cx, cy, cz), angles=(roll, pitch, yaw))

# ---------------------------
# Visualization
# ---------------------------
def visualize_optimization(shape: SensorShape, points: np.ndarray, fname: str):
    """
    Visualize the YX projection of a 3D rectangle and points.
    Y axis vertical, X axis horizontal.
    """
    # Rectangle corners in local frame
    hx, hy = shape.size[0]/2, shape.size[1]/2
    corners_local = np.array([
        [-hx, -hy, 0],
        [ hx, -hy, 0],
        [ hx,  hy, 0],
        [-hx,  hy, 0],
        [-hx, -hy, 0]  # close loop
    ])
    corners_global = corners_local @ shape.R.T + shape.center

    fig =  plt.figure(figsize=(3*2, 3*4), layout='constrained')
    axes = fig.subplot_mosaic([["YX", "YZ"],
                                ["res_vs_x", "res_vs_x"],
                                ["res_vs_y", "res_vs_y"],
                                ["res_vs_z", "res_vs_z"],
                                ])

    # YX plane
    axes["YX"].scatter(points[:,0], points[:,1], c=points[:,2], label='Points', s=30)
    axes["YX"].plot(corners_global[:,0], corners_global[:,1], color='red', lw=2, label='Fitted Rectangle')
    axes["YX"].set_xlabel('X')
    axes["YX"].set_ylabel('Y')
    axes["YX"].set_title('YX Projection of Rectangle Fit')
    axes["YX"].set_aspect('equal', adjustable='box')
    axes["YX"].legend()
    axes["YX"].grid(True)

    # Right: YZ projection
    axes["YZ"].scatter(points[:,2], points[:,1], c=points[:,0], label='Points', s=30)  # Z horiz, Y vertical
    axes["YZ"].plot(corners_global[:,2], corners_global[:,1], color='red', lw=2, label='Rectangle')
    axes["YZ"].set_title('YZ Projection')
    axes["YZ"].legend()
    axes["YZ"].grid(True)

    # Residuals
    residuals = np.array([(*p,shape.distance_to_edge(p)) for p in points])
    axes["res_vs_x"].scatter(residuals[:,0], residuals[:,3], alpha=0.3, color='tab:blue', s=30)
    axes["res_vs_x"].set_ylabel('dr')
    axes["res_vs_x"].set_xlabel('X (mm)')

    axes["res_vs_y"].scatter(residuals[:,1], residuals[:,3], alpha=0.3, color='tab:green', s=30)
    axes["res_vs_y"].set_ylabel('dr')
    axes["res_vs_y"].set_xlabel('Y (mm)')

    axes["res_vs_z"].scatter(residuals[:,2], residuals[:,3], alpha=0.3, color='tab:red', s=30)
    axes["res_vs_z"].set_ylabel('dr')
    axes["res_vs_z"].set_xlabel('Z (mm)')

    # Save figure
    plt.savefig(fname)
    plt.close(fig)

def draw_residual(self, A, X, sigmas=None):
    """
    Draw residual scatter plots for X, Y, Z axes in a single vertically stacked figure.

    Parameters
    ----------
    A : np.ndarray
        Observation matrix used in fit
    X : np.ndarray
        Estimated 6-DOF parameters
    sigmas : np.ndarray or None
        Standard deviations per axis [sigma_x, sigma_y, sigma_z].
        If None or any sigma=0, plot raw residuals.
    """

    data = self.markers
    N = data.shape[0]

    # Nominal coordinates
    P = data[:, 0:3]

    # Observation vector
    b = data[:, 3:6].reshape(-1)

    # Residuals
    r = b - A @ X
    residuals = r.reshape(-1, 3)

    # Standardize if sigmas provided and >0
    if sigmas is not None and np.all(np.array(sigmas) > 0):
        std_res = residuals / np.array(sigmas)
        ylabel = ["Std Residual X", "Std Residual Y", "Std Residual Z"]
        draw_lines = True
    else:
        std_res = residuals
        ylabel = ["Residual X [m]", "Residual Y [m]", "Residual Z [m]"]
        draw_lines = False

    axes_labels = ['X', 'Y', 'Z']

    # Create figure with 3 vertical subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    for i in range(3):
        axs[i].scatter(P[:, i], std_res[:, i], color='blue', alpha=0.7)
        axs[i].set_ylabel(ylabel[i])
        axs[i].set_title(f"{ylabel[i]} vs Nominal {axes_labels[i]}")
        axs[i].grid(True)
        if draw_lines:
            axs[i].axhline(0, color='k')
            axs[i].axhline(3, color='r', linestyle='--')
            axs[i].axhline(-3, color='r', linestyle='--')

    axs[2].set_xlabel("Nominal coordinate")

    plt.tight_layout()
    fig.savefig
    
def draw_ladder_residuals(ladder_residual:np.ndarray, fname:str | None = "ladder_residual") -> None:
    fig =  plt.figure(figsize=(3*2, 3*4), layout='constrained')
    axes = fig.subplot_mosaic([
        ["res_vs_x", "res_vs_x"],
        ["res_vs_y", "res_vs_y"],
        ["res_vs_z", "res_vs_z"],
                            ])

    axes["res_vs_x"].scatter(ladder_residual[:,0], ladder_residual[:,3], alpha=0.3, color='tab:blue', s=30)
    axes["res_vs_x"].set_ylabel('dr')
    axes["res_vs_x"].set_xlabel('X (mm)')

    axes["res_vs_y"].scatter(ladder_residual[:,1], ladder_residual[:,3], alpha=0.3, color='tab:green', s=30)
    axes["res_vs_y"].set_ylabel('dr')
    axes["res_vs_y"].set_xlabel('Y (mm)')

    axes["res_vs_z"].scatter(ladder_residual[:,2], ladder_residual[:,3], alpha=0.3, color='tab:red', s=30)
    axes["res_vs_z"].set_ylabel('dr')
    axes["res_vs_z"].set_xlabel('Z (mm)')

    # Save figure
    plt.savefig(fname)
    plt.close(fig)
