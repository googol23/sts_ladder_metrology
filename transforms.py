"""Sensor-local nominal-to-surveyed transforms; optical fitting stays in this package."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .point3d import Point3D
from .sensor_measurements import SensorMeasurements

SENSOR_FRAME = "nominal_sensor_local_v1"


def rigid_matrix(value, label="transform"):
    matrix = np.asarray(value, dtype=float)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
    ):
        raise ValueError(f"{label} must be a finite homogeneous 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0
    ) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0):
        raise ValueError(f"{label} must have a proper orthonormal rotation")
    return matrix.copy()


@dataclass(frozen=True)
class SensorDeviation:
    nominal_to_measured: np.ndarray
    covariance: np.ndarray
    fiducial_centroid_local_mm: np.ndarray
    centroid_parameters: np.ndarray
    residuals_mm: np.ndarray
    linear_residual_rms_mm: float

    def to_dict(self):
        residuals = self.residuals_mm
        return {
            "status": "ok",
            "nominal_to_measured": self.nominal_to_measured.tolist(),
            "covariance": self.covariance.tolist(),
            "covariance_parameters": [
                "tx",
                "ty",
                "tz",
                "rotvec_x",
                "rotvec_y",
                "rotvec_z",
            ],
            "covariance_units": ["mm", "mm", "mm", "rad", "rad", "rad"],
            "fiducial_centroid_local_mm": self.fiducial_centroid_local_mm.tolist(),
            "centroid_parameters": self.centroid_parameters.tolist(),
            "marker_count": len(residuals),
            "residual_rms_mm": float(np.sqrt(np.mean(residuals**2))),
            "max_residual_norm_mm": float(np.linalg.norm(residuals, axis=1).max()),
            "linear_residual_rms_mm": self.linear_residual_rms_mm,
        }


def fit_sensor_deviation(
    measurement: SensorMeasurements, nominal_sensor_to_survey
) -> SensorDeviation:
    """Fit in sensor axes, then move the fitted pivot to the nominal sensor origin.

    The legacy small-angle estimator is retained. exp(rotvec) supplies the finite
    proper rotation; residuals are re-evaluated for that exported finite transform.
    Covariance is the residual-scaled, unweighted fit covariance, first-order
    propagated to [origin translation, rotation vector], not a 4x4 covariance.
    """
    pose = rigid_matrix(nominal_sensor_to_survey, "nominal_sensor_to_survey")
    data = measurement.markers_np_array()
    if len(data) < 4 or not np.isfinite(data).all():
        raise ValueError("Sensor fit requires at least four finite fiducials")
    if any(measurement.errors):
        raise ValueError(
            "Sensor-local export currently supports unweighted fiducial fits only"
        )
    rotation = pose[:3, :3]
    points = (data[:, :3] - pose[:3, 3]) @ rotation
    deviations = data[:, 3:6] @ rotation
    centre = points.mean(axis=0)
    if np.linalg.matrix_rank(points - centre) < 2:
        raise ValueError(
            "Sensor fiducials do not constrain all six rigid-body parameters"
        )
    local = SensorMeasurements(
        [Point3D(*row) for row in np.column_stack((points, deviations))],
        [],
        (0.0, 0.0, 0.0),
    )
    fit = local.fit_deviations()
    if fit is None:
        raise ValueError("Sensor fiducial fit failed")
    parameters, covariance, _, linear_rms, _, _ = fit
    finite_rotation = Rotation.from_rotvec(parameters[3:]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = finite_rotation
    transform[:3, 3] = centre + parameters[:3] - finite_rotation @ centre
    jacobian = np.eye(6)
    step = 1e-7
    for axis in range(3):
        plus, minus = parameters[3:].copy(), parameters[3:].copy()
        plus[axis] += step
        minus[axis] -= step
        derivative = (
            Rotation.from_rotvec(plus).as_matrix()
            - Rotation.from_rotvec(minus).as_matrix()
        ) / (2 * step)
        jacobian[:3, 3 + axis] = -derivative @ centre
    origin_covariance = jacobian @ covariance @ jacobian.T
    residuals = points + deviations - (points @ finite_rotation.T + transform[:3, 3])
    if not np.isfinite(origin_covariance).all() or not np.isfinite(residuals).all():
        raise ValueError("Non-finite sensor fit result")
    return SensorDeviation(
        rigid_matrix(transform),
        origin_covariance,
        centre,
        parameters,
        residuals,
        float(linear_rms),
    )
