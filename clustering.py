import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
    

def constrained_agglomerative(points: np.ndarray,
                         n_of_cluster: int,
                         x_max: float | None = None,
                         y_max: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Cluster points by Z with optional hard X- and Y-distance constraints.

    Parameters
    ----------
    points : (N,3) array
        Each row is [x, y, z]
    n_of_cluster : int
        Number of clusters to form
    x_max : float or None
        Maximum allowed X-distance for points to be in the same cluster. None disables X constraint.
    y_max : float or None
        Maximum allowed Y-distance for points to be in the same cluster. None disables Y constraint.

    Returns
    -------
    labels : (N,) array
        Cluster labels
    centers_z : (n_of_cluster,) array
        Z-centroid of each cluster
    inertia : float
        Sum of squared Z distances to cluster centroids
    """
    N = len(points)

    # Build connectivity matrix
    connectivity = lil_matrix((N, N), dtype=int)
    for i in range(N):
        for j in range(i + 1, N):
            ok = True
            if x_max is not None:
                ok &= abs(points[i, 0] - points[j, 0]) <= x_max
            if y_max is not None:
                ok &= abs(points[i, 1] - points[j, 1]) <= y_max
            if ok:
                connectivity[i, j] = 1
                connectivity[j, i] = 1

    # Agglomerative clustering along Z
    clustering = AgglomerativeClustering(
        n_clusters=n_of_cluster,
        linkage='ward',  # good for 1D Z distance
        connectivity=connectivity
    )
    labels = clustering.fit_predict(points[:, 2].reshape(-1, 1))

    # Compute Z-centroids
    centers = np.array([(points[labels == i, 0].mean(),points[labels == i, 1].mean(),points[labels == i, 2].mean()) for i in range(n_of_cluster)])

    # Compute inertia (sum of squared distances to centroid in Z)
    inertia = sum(np.sum((points[labels == i, 2] - centers[i][2])**2) for i in range(n_of_cluster))

    return labels, centers, inertia


def visualize_clusters_2d(points: np.ndarray,
                           labels: np.ndarray,
                           centers_z: np.ndarray = None,
                           title: str = None,
                           marker_size: int = 20,
                           sensor_bboxes:np.ndarray | None = None):
    """
        Creates two side-by-side 2D panels:
        - Left:  X (horizontal) vs Y (vertical)  (XY)
        - Right: Z (horizontal) vs Y (vertical)  (YZ)
        Points are colored by integer cluster labels.

        Function:
            fig = visualize_clusters_2d(points, labels, centers_z=None, title=None, figsize=(12,5))

        Arguments:
            points: (N,3) numpy array
            labels: (N,) integer array of cluster labels (0..k-1)
            centers_z: optional (k,) array of center z-values (will be shown on YZ as vertical dashed lines)
            title: optional overall title string
            figsize: tuple for figure size
        Returns:
            matplotlib.figure.Figure object (so you can save/show further)
    """
    # Basic validation
    points = np.asarray(points)
    labels = np.asarray(labels)

    if points.shape[0] != labels.shape[0]:
        raise ValueError("points and labels must have same length")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    def get_plot_range(points, padding:float = 0.05) ->tuple[float,float]:
        rng = float(max(points) - min(points))
        return (float(min(points) - padding*rng), float(max(points) + padding*rng))

    x_range = get_plot_range(x)
    y_range = get_plot_range(y)
    z_range = get_plot_range(z)

    # Extract span values from tuples (ensure they are float)
    x_span = float(x_range[1]) - float(x_range[0])
    y_span = float(y_range[1]) - float(y_range[0])
    z_span = float(z_range[1]) - float(z_range[0])

    # Guard against degenerate ranges
    if x_span == 0:
        x_span = 1.0
    if y_span == 0:
        y_span = 1.0
    if z_span == 0:
        z_span = 1.0

    aspect_ratio = y_span / x_span
    PLOT_DPI = 300

    fig_width_pxl = 2 * 800
    fig_height_pxl = int(fig_width_pxl * aspect_ratio)

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(int(fig_width_pxl / PLOT_DPI), int(fig_height_pxl / PLOT_DPI)), dpi=PLOT_DPI)

    ax_xy.set_xlim(*x_range)
    ax_yz.set_xlim(*z_range)

    ax_xy.set_aspect('equal', 'box')
    ax_yz.set_box_aspect(aspect_ratio)

    for ax in (ax_xy, ax_yz):
        ax.set_ylim(*y_range)
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(3)  # change 2 to whatever thickness you want
        ax.tick_params(
            direction='in',      # ticks point inward
            length=6, width=1,   # optional tick size
            top=True,         # ticks on top
            right=True,       # ticks on right
            labelbottom=True,    # x-axis labels on bottom
            labeltop=True,      # x-axis labels on top
            labelleft=True,      # y-axis labels on left
            labelright=True,    # y-axis labels on right
            pad=-30              # negative padding moves labels inward
        )

    # XY panel: X horizontal, Y vertical
    if sensor_bboxes is not None:
        for box in sensor_bboxes:
            size_x = box[1] - box[0]
            size_y = box[3] - box[2]
            x_0 = 0.5*(box[0] + box[1])
            y_0 = 0.5*(box[2] + box[3])
            rect = Rectangle(
                (x_0 - 0.5 * size_x, y_0 - 0.5 * size_y),
                size_x,
                size_y,
                fill=True,
                edgecolor='black',
                facecolor='gray',
                alpha=0.3
            )
            ax_xy.add_patch(rect)
            
    ax_xy.scatter(x, y, c=labels, s=marker_size, cmap='tab10', edgecolors='none')
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("XY projection")
    ax_xy.grid(True, linestyle=':', linewidth=0.5)

    

    # YZ panel: Z horizontal, Y vertical (user requested Y always vertical)
    if sensor_bboxes is not None:
        for box in sensor_bboxes:
            size_x = max(z) - min(z)
            size_y = box[3] - box[2]
            x_0 = 0.5*(max(z) + min(z))
            y_0 = 0.5*(box[2] + box[3])
            rect = Rectangle(
                (x_0 - 0.5 * size_x, y_0 - 0.5 * size_y),
                size_x,
                size_y,
                fill=True,
                edgecolor='black',
                facecolor='gray',
                alpha=0.3
            )
            ax_yz.add_patch(rect)
    ax_yz.scatter(z, y, c=labels, s=marker_size, cmap='tab10', edgecolors='none')
    ax_yz.set_xlabel("Z")             # Z on horizontal axis
    ax_yz.set_ylabel("Y")             # Y always vertical
    ax_yz.set_title("YZ projection")
    ax_yz.grid(True, linestyle=':', linewidth=0.5)

    

    # If centers_z provided, draw vertical dashed lines at each center z
    if centers_z is not None:
        centers_z = np.asarray(centers_z)
        # compute visible y-range to draw lines across that range
        y_min, y_max = np.min(y), np.max(y)
        for i, cz in enumerate(centers_z):
            ax_yz.vlines(cz, y_min, y_max, linestyles='dotted', linewidth=1.0, label=f"center {i}")

    fig.savefig(f"clustering_sensors.png", bbox_inches='tight', pad_inches=0.01)


def generate_synth_data(N:int = 300) -> np.array:
    np.random.seed(0)
    # create synthetic 3D points: three clusters in z, with varying x used as weight
    xs = np.random.uniform(0.1, 2.0, size=N)         # use x as positive weights (avoid zeros)
    ys = np.concatenate([
        np.random.normal(loc=-5.0, scale=0.5, size=100),
        np.random.normal(loc=+0.0, scale=0.5, size=100),
        np.random.normal(loc=+5.0, scale=0.5, size=100),
    ])
    zs = np.concatenate([
        np.random.normal(loc=-5.0, scale=0.5, size=100),
        np.random.normal(loc=0.5, scale=0.4, size=100),
        np.random.normal(loc=6.0, scale=0.6, size=100),
    ])
    points = np.vstack([xs, ys, zs]).T
    return points

# -------------------------
# Example usage (single example)
# -------------------------
if __name__ == "__main__":
    points = generate_synth_data(300)
    k = 3

    # labels, centers_z, inertia = kmeans_z_weighted(points, k=k, weight_axis='x')
    labels, centers_z, inertia = constrained_agglomerative(points, n_of_cluster=3, x_max=2)
    visualize_clusters_2d(points, labels, centers_z=centers_z, title="KMeans (z-weighted) clusters")


    print("Cluster centers (z):", centers_z)
    print("Inertia:", inertia)
    # count members per cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("Counts per cluster:", dict(zip(unique.tolist(), counts.tolist())))
