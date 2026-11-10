import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger()


from sts_geo_layout.assembly import load_layout
from sts_geo_layout.plotting import plot_layout

from sts_geo_layout import LadderLayout
from sts_ladder_metrology.point3d import Point3D
from sts_ladder_metrology.sensor_measurements import SensorMeasurements


@dataclass(frozen=True)
class LadderMeasurement:
    """
    It hold raw measurement for a single ladder.
    Measurement consist on two data sets:
        - markers measurements
        - surface measurements
    Sensors's measurement are stored as SensorMeasurement
    """

    ladder_name: str
    markers: np.ndarray
    surface: np.ndarray
    _layout: LadderLayout | None = None

    @property
    def markers_err(self) -> np.ndarray | None:
        if self.markers.shape[1] <= 6:
            return None
        return self.markers[:, -3:-1]

    @property
    def surface_err(self) -> np.ndarray | None:
        if self.surface.shape[1] < 6:
            return None
        return self.surface[:, -3:-1]

    @cached_property
    def layout(self) -> LadderLayout:
        return (
            self._layout if self._layout is not None else load_layout(self.ladder_name)
        )

    @staticmethod
    def validate_marker_file(path: Path | str) -> bool:
        """Validate the contents of a marker file."""
        path = Path(path)
        if not path.is_file():
            return False
        if path.stat().st_size == 0:
            return False

        # TODO: Additional validation logic

        return True

    @staticmethod
    def validate_surface_file(path: Path | str) -> bool:
        """Validate the contents of a surface file."""
        path = Path(path)

        if not path.is_file():
            return False
        if path.stat().st_size == 0:
            return False

        # TODO: Additional validation logic

        return True

    @staticmethod
    def load_markers_data(path: Path | str, skiprows: int = 1) -> np.ndarray:
        """Load marker data from a marker file.
        Assumptions:
        - File contains N rows
        - Each row contains 6 columns: X, Y, Z, DX, DY, DZ
        Use skiprows for header prescence
        - First row is a header
        Returns:
        Numpy array of shape (N-1, 6) containing marker points data
        """
        path = Path(path)

        if not LadderMeasurement.validate_marker_file(path):
            raise ValueError(f"Invalid marker file: {path}")

        data = np.loadtxt(path, skiprows=skiprows)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 6:
            raise ValueError(f"Marker file {path} does not have 6 columns")

        return data

    @staticmethod
    def load_surface_data(path: Path | str, skiprows: int = 1) -> np.ndarray:
        """ "
        Load surface data from surface file.
        Assumptions:
        - File contains N rows
        - Each row contains 4 columns: X,Y,Z,DZ
        Use skiprows for header prescence (default 1)
        - First row is a header: X/D:Y:Z:Zd
        Returns:
        Numpy array of shape (N-1, 4) containing surface points data
        """
        path = Path(path)

        if not LadderMeasurement.validate_surface_file(path):
            raise ValueError(f"Invalid surface file: {path}")

        data = np.loadtxt(path, skiprows=skiprows)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != 4:
            raise ValueError(f"Marker file {path} does not have 4 columns")

        return data

    @classmethod
    def load(
        cls,
        ladder_name: str,
        markers_file: Path | str,
        surface_file: Path | str,
        *,
        layout: LadderLayout | None = None,
    ) -> "LadderMeasurement":
        logger.info(
            f"{ladder_name} - Loading ladder measurements from: {markers_file}, {surface_file}"
        )
        markers_data = cls.load_markers_data(markers_file)
        surface_data = cls.load_surface_data(surface_file)

        # Go from metrology table frame to CBM frame (swap X-Y axis)
        markers_data[:, [1, 0, 2, 4, 3, 5]] = markers_data[:, [0, 1, 2, 3, 4, 5]]

        return cls(
            ladder_name=ladder_name,
            markers=markers_data,
            surface=surface_data,
            _layout=layout,
        )

    def cluster_markers(
        self, tolerance_z: float = 0.9, tolerance_y: float = 0.5
    ) -> np.ndarray:
        layout = self.layout
        bboxes = layout.sensor_bbox()

        y = self.markers[:, 1]
        z = self.markers[:, 2]

        n = len(self.markers)

        # ---- Z clustering ----
        order_z = np.argsort(z)
        z_sorted = z[order_z]

        dz = np.diff(z_sorted)
        z_split = np.where(dz > tolerance_z)[0] + 1
        z_groups = np.split(order_z, z_split)

        labels = np.full(n, -1, dtype=int)

        # ---- process each Z cluster independently ----
        labels[z_groups[0]] = 0

        y_center = np.mean(y[z_groups[0]])
        for gi, g in enumerate(z_groups[1:]):
            for i in g:
                labels[i] = 1 + gi * 2 + int(y[i] < y_center)

        if (m := np.max(labels)) == layout.n_sensors:
            labels[labels == m] = layout.n_sensors - 1

        return labels

    def grouped_sensor_measurements(self) -> list[SensorMeasurements]:
        """Return fiducial groups in the layout's bottom-to-top module order."""
        labels = self.cluster_markers()
        if np.unique(labels).size != self.layout.n_sensors:
            raise ValueError(
                f"{self.ladder_name}: marker clusters do not match assembly sensors"
            )
        groups = [self.markers[labels == label] for label in np.unique(labels)]
        groups.sort(key=lambda points: float(np.mean(points[:, 1])))
        return [
            SensorMeasurements([Point3D(*row) for row in points], [], (0.0, 0.0, 0.0))
            for points in groups
        ]

    def sensor_measurements(
        self, out_path: str | Path | None = None
    ) -> tuple[list[SensorMeasurements], list[tuple | None]]:
        measurements = self.grouped_sensor_measurements()

        plot_layout(self.layout)
        for measurement in measurements:
            measurement.draw_markers()

        for i, fig_num in enumerate(plt.get_fignums(), start=1):
            fig = plt.figure(fig_num)
            fig.tight_layout(pad=0.5)
            fig.savefig(f"{out_path}/{self.ladder_name}_layout.svg")
            fig.savefig(f"{out_path}/{self.ladder_name}_layout.png", dpi=300)
            plt.close()

        valid_markers = []
        valid_residuals = []
        fit_results: list[tuple | None] = []

        for s in measurements:
            try:
                fit_result = s.fit_deviations()

                fit_results.append(fit_result)

                if fit_result is not None:
                    valid_markers.append(s.markers_np_array())
                    valid_residuals.append(fit_result[2])
            except Exception as e:
                print(f"FIT FAILED: {e}")
                fit_results.append(None)

        if valid_markers and valid_residuals:
            markers_used = np.vstack(valid_markers)
            residuals_used = np.vstack(valid_residuals)

            self.draw_residual(
                markers_used, residuals_used, f"{out_path}/Ladder_res.png"
            )

        return measurements, fit_results

    def draw_residual(
        self, markers: np.ndarray, residuals: np.ndarray, f_name: str = "fig.png"
    ):
        """
        Draw residuals vs nominal coordinates for X, Y, Z axes
        in a single figure with 3 vertically stacked subplots.
        Residuals are standardized if errors > 0, otherwise raw.
        """
        P = markers
        N = P.shape[0]

        # Check if we can standardize
        if self.markers_err is None or self.markers_err.all() < 1e-3:
            std_res = residuals
            ylabel = ["Residual X [mm]", "Residual Y [mm]", "Residual Z [mm]"]
            draw_lines = False
        else:
            print(residuals.shape, self.markers_err.shape, residuals, self.markers_err)
            std_res = residuals / self.markers_err
            ylabel = ["Std Residual X", "Std Residual Y", "Std Residual Z"]
            draw_lines = True

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
        plt.close()


if __name__ == "__main__":
    lm = LadderMeasurement.load(
        "L4DL000161", "sources/L4DL000161_marker.txt", "sources/L4DL000161_surface.txt"
    )
    lm.sensor_measurements()
