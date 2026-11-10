from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from sts_geo_layout import SensorType
from sts_ladder_metrology.fitter import cost_fn, make_callback
from sts_ladder_metrology.point3d import Point3D
from sts_ladder_metrology.sensor_fit import SensorFit
from sts_ladder_metrology.sensor_shape import SensorShape


@dataclass(frozen=True)
class SensorMeasurements:
    """
    Represents a set of metrology measurement for a sensor onto a ladder.

    It holds:
        - surface measurements
        - markers measurements
    """

    markers: list[Point3D]
    surface: list[Point3D]
    errors: tuple[float, float, float]

    # ============ Array Conversion ============
    def markers_np_array(self) -> np.ndarray:
        """Convert markers to numpy array using vectorized stack"""
        if not self.markers:
            return np.empty((0, 6), dtype=np.float64)
        return np.column_stack(
            [
                [p.x for p in self.markers],
                [p.y for p in self.markers],
                [p.z for p in self.markers],
                [p.x_err for p in self.markers],
                [p.y_err for p in self.markers],
                [p.z_err for p in self.markers],
            ]
        )

    def surface_np_array(self) -> np.ndarray:
        """Convert surface points to numpy array using vectorized stack"""
        if not self.surface:
            return np.empty((0, 6), dtype=np.float64)
        return np.column_stack(
            [
                [p.x for p in self.surface],
                [p.y for p in self.surface],
                [p.z for p in self.surface],
                [p.x_err for p in self.surface],
                [p.y_err for p in self.surface],
                [p.z_err for p in self.surface],
            ]
        )

    def all_points_np_array(self) -> np.ndarray:
        """Combine all points (markers + surface) into single array"""
        markers_arr = self.markers_np_array()
        surface_arr = self.surface_np_array()
        return (
            np.vstack([markers_arr, surface_arr])
            if len(surface_arr) > 0
            else markers_arr
        )

    # ============ Drawing helpers ============
    def draw_markers(self):
        """
        Draw markers points
        """
        points_xy = (
            self.markers_np_array()[:, [0, 1]] + self.markers_np_array()[:, [3, 4]]
        )

        # XY projection
        plt.scatter(points_xy[:, 0], points_xy[:, 1], marker="s", s=10)
        plt.scatter(
            points_xy[:, 0],
            points_xy[:, 1],
            marker="+",
            s=8,
            color="black",
            linewidth=0.2,
        )

    def draw_surface(self):
        """
        Draw surface points
        """
        points = self.surface_np_array()[:, 0:2]

        # XY projection
        points[:, [0, 1]] = points[:, [1, 0]]
        plt.scatter(points[:, 0], points[:, 1], marker="s", s=10)
        plt.scatter(
            points[:, 0], points[:, 1], marker="+", s=8, color="black", linewidth=0.2
        )

    def draw_residual(self, residuals: np.ndarray, f_name: str = "fig.png"):
        """
        Draw residuals vs nominal coordinates for X, Y, Z axes
        in a single figure with 3 vertically stacked subplots.
        Residuals are standardized if errors > 0, otherwise raw.
        """
        P = self.markers_np_array()
        N = P.shape[0]

        # Check if we can standardize
        if all(e > 1e-3 for e in self.errors):
            std_res = residuals / np.array(self.errors)
            ylabel = ["Std Residual X", "Std Residual Y", "Std Residual Z"]
            draw_lines = True
        else:
            std_res = residuals
            ylabel = ["Residual X [mm]", "Residual Y [mm]", "Residual Z [mm]"]
            draw_lines = False

        axes_labels = ["X", "Y", "Z"]

        # Create 3x2 subplot grid
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        axs = axs.reshape(3, 2)

        for i, axis in enumerate(["x", "y", "z"]):
            # Scatter plot (left)
            axs[i, 0].scatter(P[:, 0], std_res[:, i], color="blue", alpha=0.7)
            axs[i, 0].scatter(P[:, 0], std_res[:, i], color="blue", alpha=0.7)
            axs[i, 0].scatter(P[:, 0], std_res[:, i], color="blue", alpha=0.7)
            axs[i, 0].set_ylabel(ylabel[i])
            axs[i, 0].grid(True)
            if draw_lines:
                axs[i, 0].axhline(0, color="k")
                axs[i, 0].axhline(3, color="r", linestyle="--")
                axs[i, 0].axhline(-3, color="r", linestyle="--")
            axs[i, 0].set_xlabel(f"{axis} nominal coordinate")

            # Histogram (right)
            axs[i, 1].hist(
                std_res[:, i], bins=10, color="green", alpha=0.7, edgecolor="black"
            )
            axs[i, 1].grid(True)
            if draw_lines:
                axs[i, 1].axvline(0, color="k")
                axs[i, 1].axvline(3, color="r", linestyle="--")
                axs[i, 1].axvline(-3, color="r", linestyle="--")
            axs[i, 1].set_xlabel(f"{axis} residual value")

        plt.tight_layout()
        print(f"Saving residual plot: {f_name}")
        fig.savefig(f_name)

    # ============ Fit ============
    def fit(self, sensor_type: SensorType) -> SensorFit:
        sensor_size = sensor_type.size_xy()

        pts = self.markers_np_array()[:, :3]

        if len(pts) < 4:
            cx = np.mean(pts[:, 0])
            cy = np.mean(pts[:, 1])
            cz = np.mean(pts[:, 2])
            return SensorFit(
                SensorShape(
                    size=sensor_size, center=(cx, cy, cz), angles=(roll, pitch, yaw)
                ),
                (cx, cy, cz, roll, pitch, yaw),
                (0, 0, 0, 0, 0, 0),
            )

        # Initial guess: center at mean of points, no rotation
        init_center = pts.mean(axis=0)
        init_angles = np.zeros(3)
        init_params = np.concatenate([init_center, init_angles])

        # Bounds determined by nominal distance from markers to edge
        margin = 0.1
        x_min, x_max = pts[:, 0].min() - margin, pts[:, 0].max() + margin
        y_min, y_max = pts[:, 1].min() - margin, pts[:, 1].max() + margin
        z_min, z_max = pts[:, 2].min() - margin, pts[:, 2].max() + margin
        bounds_lower = [x_min, y_min, z_min, -np.pi, -np.pi, -np.pi]
        bounds_upper = [x_max, y_max, z_max, np.pi, np.pi, np.pi]
        bounds = tuple(zip(bounds_lower, bounds_upper))

        # Optimize
        res_min = minimize(
            fun=lambda q: cost_fn(q, pts, sensor_size),
            x0=init_params,
            method="L-BFGS-B",
            # bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-10},
            callback=make_callback(pts, sensor_size),
        )

        fit_result = type("obj", (), {})()
        fit_result.x = res_min.x
        fit_result.optimizer = res_min  # raw result if you want more details

        # Construct optimized shape
        cx, cy, cz, roll, pitch, yaw = fit_result.x
        fitted_shape = SensorShape(
            size=sensor_size, center=(cx, cy, cz), angles=(roll, pitch, yaw)
        )

        #  Uncertainties estimation
        Hinv = res_min.hess_inv.todense()
        residuals = np.array([fitted_shape.distance_to_edge(p) for p in pts])
        dof = len(pts) - len(res_min.x)
        sigma2 = np.sum(residuals**2) / dof

        cov = sigma2 * Hinv
        param_errors = np.sqrt(np.diag(cov))

        return SensorFit(
            fitted_shape, (cx, cy, cz, roll, pitch, yaw), (param_errors.tolist())
        )

    def fit_rectangle(self, size: tuple[float, float]):
        """
        Returns:
            center  : (3,)
            angles  : (3,)  -> roll, pitch, yaw (ZYX)
            cov     : (6,6) covariance of [cx,cy,cz,roll,pitch,yaw]
        """
        markers_pts = self.markers_np_array()
        points = markers_pts[:, :3] + markers_pts[:, 3:]

        Lx, Ly = size
        N = len(points)

        # -------------------------------------------------
        # 1) Plane fit (PCA)
        # -------------------------------------------------
        centroid = points.mean(axis=0)
        X = points - centroid

        C = X.T @ X / N
        eigvals, eigvecs = np.linalg.eigh(C)

        l1, l2, l3 = eigvals
        v1, v2, v3 = eigvecs.T

        normal = v1 / np.linalg.norm(v1)

        # noise variance (orthogonal residuals)
        residuals = X @ normal
        sigma2 = np.sum(residuals**2) / (N - 3)

        # normal covariance
        Cov_normal = (
            sigma2
            / N
            * (np.outer(v2, v2) / (l2 - l1) ** 2 + np.outer(v3, v3) / (l3 - l1) ** 2)
        )

        # -------------------------------------------------
        # 2) In-plane basis
        # -------------------------------------------------
        u0 = v3
        v0 = v2
        if np.dot(np.cross(u0, v0), normal) < 0:
            v0 = -v0

        x = X @ u0
        y = X @ v0
        XY = np.column_stack((x, y))

        # -------------------------------------------------
        # 3) In-plane PCA (rectangle orientation)
        # -------------------------------------------------
        C2 = XY.T @ XY / N
        eigvals2, eigvecs2 = np.linalg.eigh(C2)
        mu1, mu2 = eigvals2
        R2 = eigvecs2[:, ::-1]

        alpha = np.arctan2(R2[1, 0], R2[0, 0])
        var_alpha = sigma2 / (N * (mu2 - mu1) ** 2)

        # -------------------------------------------------
        # 4) Rectangle center
        # -------------------------------------------------
        XYr = XY @ R2
        xr, yr = XYr[:, 0], XYr[:, 1]

        cx2d = 0.5 * (xr.max() + xr.min())
        cy2d = 0.5 * (yr.max() + yr.min())

        u = u0 * R2[0, 0] + v0 * R2[1, 0]
        v = u0 * R2[0, 1] + v0 * R2[1, 1]

        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)

        center = centroid + cx2d * u + cy2d * v

        # center covariance
        Cov_centroid = sigma2 / N * np.eye(3)
        d_center_d_alpha = cx2d * (-v) + cy2d * u
        Cov_center = Cov_centroid + var_alpha * np.outer(
            d_center_d_alpha, d_center_d_alpha
        )

        # -------------------------------------------------
        # 5) Build rotation matrix
        # -------------------------------------------------
        R = np.column_stack((u, v, normal))

        # ZYX Euler extraction
        sy = -R[2, 0]
        pitch = np.arcsin(sy)

        if abs(np.cos(pitch)) > 1e-8:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], R[1, 1])

        angles = np.array([roll, pitch, yaw])

        # -------------------------------------------------
        # 6) Propagate covariance to Euler angles
        # -------------------------------------------------
        Cov_p = np.zeros((4, 4))
        Cov_p[:3, :3] = Cov_normal
        Cov_p[3, 3] = var_alpha

        def f(p):
            n = p[:3] / np.linalg.norm(p[:3])
            a = p[3]

            # rebuild basis
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(ref, n)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])

            uu = np.cross(n, ref)
            uu /= np.linalg.norm(uu)
            vv = np.cross(n, uu)

            u_rot = np.cos(a) * uu + np.sin(a) * vv
            v_rot = -np.sin(a) * uu + np.cos(a) * vv

            RR = np.column_stack((u_rot, v_rot, n))

            sy = -RR[2, 0]
            pitch = np.arcsin(sy)

            if abs(np.cos(pitch)) > 1e-8:
                roll = np.arctan2(RR[2, 1], RR[2, 2])
                yaw = np.arctan2(RR[1, 0], RR[0, 0])
            else:
                roll = 0.0
                yaw = np.arctan2(-RR[0, 1], RR[1, 1])

            return np.array([roll, pitch, yaw])

        # numerical Jacobian
        eps = 1e-8
        p0 = np.hstack((normal, alpha))
        J = np.zeros((3, 4))
        f0 = f(p0)

        for i in range(4):
            dp = np.zeros(4)
            dp[i] = eps
            J[:, i] = (f(p0 + dp) - f0) / eps

        Cov_angles = J @ Cov_p @ J.T

        # -------------------------------------------------
        # 7) Final 6x6 covariance
        # -------------------------------------------------
        cov = np.zeros((6, 6))
        cov[:3, :3] = Cov_center
        cov[3:, 3:] = Cov_angles

        return center, angles, cov

    def fit_deviations(self):
        """
        Estimate rigid-body alignment correction using small-angle model.

        Returns
        -------
        params : np.ndarray (6,)
            [dx, dy, dz, d_roll, d_pitch, d_yaw]  (angles in radians)

        covariance : np.ndarray (6,6)
            Covariance matrix of the estimated parameters
        """

        data = self.markers_np_array()
        N = data.shape[0]

        if N < 4:
            print("Insuficient number of points to perform fit")
            return None

        # Split nominal coordinates and measured deviations
        P = data[:, 0:3].copy()
        D = data[:, 3:6].copy()

        # ---- Important: center coordinates to improve conditioning ----
        centroid = np.mean(P, axis=0)
        P -= centroid

        # Build linear system A X = b
        A = np.zeros((3 * N, 6))
        b = D.reshape(-1)

        for i in range(N):
            x, y, z = P[i]

            A[3 * i + 0] = [1, 0, 0, 0, z, -y]
            A[3 * i + 1] = [0, 1, 0, -z, 0, x]
            A[3 * i + 2] = [0, 0, 1, y, -x, 0]

        # ---- Least squares solution ----
        if max(self.errors) != 0:
            # Build weight matrix diagonal
            W = np.zeros((3 * N, 3 * N))

            for i in range(N):
                sx, sy, sz = self.errors
                W[3 * i + 0, 3 * i + 0] = 1.0 / sx**2
                W[3 * i + 1, 3 * i + 1] = 1.0 / sy**2
                W[3 * i + 2, 3 * i + 2] = 1.0 / sz**2

            # Weighted normal equations
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b

            X = np.linalg.solve(ATA, ATb)

        else:
            ATA = A.T @ A
            ATb = A.T @ b
            X = np.linalg.solve(ATA, ATb)

        # ---- Residuals ----
        residuals = b - A @ X
        dof = 3 * N - 6
        sigma2 = (residuals @ residuals) / dof

        residuals = residuals.reshape(-1, 3)
        rms = np.sqrt(np.mean(residuals**2, axis=0))
        total_rms = np.sqrt(np.mean(residuals**2))

        # ---- Covariance of parameters ----
        if max(self.errors) != 0:
            covariance = np.linalg.inv(ATA)  # Know weights (errors provided)
        else:
            covariance = sigma2 * np.linalg.inv(
                ATA
            )  # Unknown weights (no errors provided)

        # Extract parameters
        dx, dy, dz, wx, wy, wz = X
        params = np.array([dx, dy, dz, wx, wy, wz])

        # Chi2 test
        if max(self.errors) != 0:
            chi2 = residuals.flatten().T @ W @ residuals.flatten()
        else:
            chi2 = residuals.flatten().T @ residuals.flatten()

        chi2_reduced = chi2 / (3 * N - len(params))

        # Parameter significance
        std = np.sqrt(np.diag(covariance))
        # Exact fits have zero residual variance; significance is then undefined.
        Z = np.divide(X, std, out=np.full_like(X, np.nan), where=std > 0)

        return params, covariance, residuals, total_rms, chi2_reduced, Z

    import numpy as np

    def generate_dummy(
        self,
        shape: SensorShape,
        N: int | tuple[int, int, int, int],
        d_edge: float,
        sigma: tuple[float, float, float] = (0.0, 0.0, 0.0),
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
        trans: tuple[float, float, float] = (0.0, 0.0, 0.0),
        randomize: bool = True,
    ):
        """
        Generate synthetic marker points near sensor edges.

        Parameters
        ----------
        N : int or tuple[int,int,int,int]
            Number of points. If int, equally split across 4 edges.
            If tuple, number of points per edge: (left, right, bottom, top)

        d_edge : float
            Distance of markers from the closest edge

        sigma : tuple[float,float,float]
            Gaussian noise std for (x,y,z)

        angles : tuple[float,float,float]
            (roll, pitch, yaw)

        trans : tuple[float,float,float]
            translation (dx, dy, dz)

        Returns
        -------
        points : np.ndarray shape (N,3)
        """

        Lx, Ly = shape.size

        # ---- number of points per edge ----
        if isinstance(N, int):
            base = N // 4
            remainder = N % 4
            n_each = [base] * 4
            for i in range(remainder):
                n_each[i] += 1
        else:
            n_each = list(N)

        # order: (left, right, bottom, top)
        pts = []

        def sample_line(a, b, n):
            if n == 0:
                return np.array([])
            if randomize:
                return np.random.uniform(a, b, n)
            else:
                return np.linspace(a, b, n)

        # LEFT
        x = -Lx / 2 + d_edge
        y_vals = sample_line(-Ly / 2 + d_edge, Ly / 2 - d_edge, n_each[0])
        for y in y_vals:
            pts.append([x, y, 0.0])

        # RIGHT
        x = Lx / 2 - d_edge
        y_vals = sample_line(-Ly / 2 + d_edge, Ly / 2 - d_edge, n_each[1])
        for y in y_vals:
            pts.append([x, y, 0.0])

        # BOTTOM
        y = -Ly / 2 + d_edge
        x_vals = sample_line(-Lx / 2 + d_edge, Lx / 2 - d_edge, n_each[2])
        for x in x_vals:
            pts.append([x, y, 0.0])

        # TOP
        y = Ly / 2 - d_edge
        x_vals = sample_line(-Lx / 2 + d_edge, Lx / 2 - d_edge, n_each[3])
        for x in x_vals:
            pts.append([x, y, 0.0])

        nominal = np.array(pts)

        # ---- rotation ----
        roll, pitch, yaw = angles

        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)],
            ]
        )

        Ry = np.array(
            [
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)],
            ]
        )

        Rz = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )

        R = Rz @ Ry @ Rx

        # ---- transform ----
        pts = (R @ nominal.T).T + np.array(trans)

        # ---- noise ----
        noise = np.random.normal(0, sigma, size=pts.shape)
        pts_noisy = pts + noise

        # store deviations relative to nominal
        markers = [
            Point3D(
                x=nom[0],
                y=nom[1],
                z=nom[2],
                x_err=(pt[0] - nom[0]),
                y_err=(pt[1] - nom[1]),
                z_err=(pt[2] - nom[2]),
            )
            for nom, pt in zip(nominal, pts_noisy)
        ]

        return SensorMeasurements(markers=markers, surface=[], errors=sigma)


if __name__ == "__main__":
    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    # --- define true sensor ---
    true_shape = SensorShape(
        size=(60.0, 60.0),
        center=(0.0, 0.0, 0.0),
        angles=(0.0, 0.0, 0.0),
    )

    N_iter = 10000  # or whatever
    n_comp = 6
    errors_arr = np.empty((N_iter, n_comp))  # pre-allocate
    rel_un_arr = np.empty((N_iter, n_comp))  # pre-allocate

    # Define ranges for translation and angles
    trans_range = np.array(
        [
            [-0.5, 0.5],  # dx
            [-0.5, 0.5],  # dy
            [-0.5, 0.5],
        ]
    )  # dz

    angle_range = np.array(
        [
            [-0.02, 0.02],  # roll
            [-0.02, 0.02],  # pitch
            [-0.02, 0.02],
        ]
    )  # yaw

    # Generate all random translations (shape: N_iter x 3)
    all_trans = np.random.uniform(
        low=trans_range[:, 0], high=trans_range[:, 1], size=(N_iter, 3)
    )

    # Generate all random angles (shape: N_iter x 3)
    all_angles = np.random.uniform(
        low=angle_range[:, 0], high=angle_range[:, 1], size=(N_iter, 3)
    )

    from tqdm import tqdm

    for i in tqdm(range(N_iter)):
        # --- create measurement object ---
        meas = SensorMeasurements(markers=[], surface=[], errors=(0, 0, 0))

        # Random values within ranges
        true_trans = tuple(all_trans[i])
        true_angle = tuple(all_angles[i])

        # --- generate synthetic data ---
        meas = meas.generate_dummy(
            shape=true_shape,
            N=(20, 20, 0, 5),
            d_edge=1.0,
            sigma=(0.00, 0.00, 0.00),
            angles=true_angle,
            trans=true_trans,
        )

        # --- Predict alignment parameters by fitting deviation  ---
        params_dev, cov_dev, residuals, rms, chi2, Z = meas.fit_deviations()

        # --- simple error comparison ---
        true = np.array([*true_trans, *true_angle])
        errors_arr[i, :] = params_dev - true

        # --- extract uncertainties ---
        sigma_dev = np.sqrt(np.diag(cov_dev))
        rel_unc_dev = sigma_dev / (np.abs(params_dev) + 1e-12)
        rel_un_arr[i, :] = rel_unc_dev

    # --- define QA histograms ---
    components = ["x", "y", "z", "phi", "theta", "psi"]
    units = [
        "mm",
        "mm",
        "mm",
        "rad",
        "rad",
        "rad",
    ]

    fig, axs = plt.subplots(3, 2, figsize=(10, 15), constrained_layout=True)
    from scipy.stats import norm

    for i in range(3):
        mu, sigma = norm.fit(errors_arr[:, i])
        rms = np.sqrt(np.mean(errors_arr[:, i] ** 2))
        x = np.linspace(errors_arr[:, i].min(), errors_arr[:, i].max(), 100)
        p = norm.pdf(x, mu, sigma)

        axs[i, 0].plot(x, p, "r-", lw=2)
        axs[i, 0].hist(
            errors_arr[:, i],
            bins=2000,
            range=(-0.5, +0.5),
            color="skyblue",
            edgecolor="black",
            alpha=0.8,
            density=True,
        )
        axs[i, 0].set_xlabel(
            f"(c{components[i]}_Fit - c{components[i]}_True) [{units[i]}]"
        )
        axs[i, 0].set_ylabel("Count")
        axs[i, 0].grid(True)

        # Mean text box
        axs[i, 0].text(
            0.05,
            0.95,
            rf"$\mu={mu:.3f} [{units[i]}]$",
            transform=axs[i, 0].transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )
        # Sigma text box
        axs[i, 0].text(
            0.05,
            0.85,
            rf"$\sigma={sigma:.3f} [{units[i]}]$",
            transform=axs[i, 0].transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )
        # RMS text box
        axs[i, 0].text(
            0.05,
            0.75,
            rf"$RMS={rms:.3f} [{units[i]}]$",
            transform=axs[i, 0].transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        axs[i, 1].hist(
            rel_un_arr[:, i],
            bins=50,
            range=(1e-6, 0.5),
            color="skyblue",
            edgecolor="black",
            alpha=0.8,
        )
        axs[i, 1].set_xlabel(f"Relative Error c{components[i]}_Fit")
        axs[i, 1].set_ylabel("Count")
        axs[i, 1].grid(True)

    # plt.tight_layout()
    plt.savefig("qa_sensor_measurement_fit.png", dpi=300)
