"""Half-unit optical analysis and its versioned external result contract."""

import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from sts_geo_layout.assembly import load_layout
from sts_utils.sts_naming import LADDER_NAME_PATTERN, is_valid_label

from .ladder_measurement import LadderMeasurement
from .transforms import SENSOR_FRAME, fit_sensor_deviation, rigid_matrix

SCHEMA_VERSION = 1
PRODUCER_VERSION = "1.0.0"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".json", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


class LadderMetrologyAnalysis:
    """Analyze selected STS8 ladders without plotting or point-cloud dependencies."""

    def __init__(self, config, config_dir, *, connection=None):
        self.config_dir = Path(config_dir).resolve()
        self.connection: Any = connection
        half = config.get("half_unit")
        if not isinstance(half, dict) or half.get("assembly_naming") != "STS8":
            raise ValueError("half_unit must declare assembly_naming: STS8")
        half_name = half.get("name")
        if not isinstance(half_name, str) or not re.fullmatch(
            r"U[0-7][UD][LR]", half_name
        ):
            raise ValueError("half_unit.name must be an STS8 identifier such as U3DL")
        self.half_unit = half_name
        self.options = config.get("ladder_metrology")
        if not isinstance(self.options, dict):
            raise ValueError(
                "ladder_metrology must be a mapping, not the old result filename"
            )
        if self.options.get("sensor_frame") != SENSOR_FRAME:
            raise ValueError(
                f"ladder_metrology.sensor_frame must explicitly be {SENSOR_FRAME}"
            )
        self.overrides = self.options.get("ladders", {})
        if not isinstance(self.overrides, dict):
            raise ValueError(
                "ladder_metrology.ladders must be a mapping of STS8 ladder IDs"
            )
        if (
            not isinstance(self.options.get("source_dir"), str)
            or not self.options["source_dir"].strip()
        ):
            raise ValueError("ladder_metrology.source_dir must be a non-empty path")
        result_file = self.options.get("result_file")
        output_dir = self.options.get("output_dir")
        if result_file is not None and output_dir is not None:
            raise ValueError(
                "ladder_metrology must use result_file or legacy output_dir, not both"
            )
        if result_file is not None:
            if not isinstance(result_file, str) or not result_file.strip():
                raise ValueError("ladder_metrology.result_file must be a non-empty path")
            self.result_path = self.resolve(result_file)
        elif isinstance(output_dir, str) and output_dir.strip():
            self.result_path = self.resolve(output_dir) / "sensor_transforms.json"
        else:
            raise ValueError(
                "ladder_metrology.result_file must name the output JSON artifact"
            )
        self.source_dir = self.resolve(self.options["source_dir"])

    def resolve(self, path):
        return (self.config_dir / Path(path)).resolve()

    def _prepare(self, selected):
        connection: Any
        owned = self.connection is None
        if owned:
            from sts_db_api.db_api import get_conn

            connection = get_conn()
        else:
            connection = self.connection
        try:
            if owned:
                connection.set_session(readonly=True)
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT name FROM public.sts_ladder WHERE name LIKE %s ORDER BY name",
                    ("L" + self.half_unit[1:] + "%",),
                )
                inventory = [row["name"] for row in cursor.fetchall()]
            if not inventory or len(inventory) != len(set(inventory)):
                raise ValueError(
                    f"Missing or duplicate assembly inventory for {self.half_unit}"
                )
            for name in inventory:
                if not is_valid_label(name, LADDER_NAME_PATTERN):
                    raise ValueError(f"Invalid assembly ladder ID: {name}")
            if set(self.overrides) - set(inventory):
                raise ValueError(
                    "ladder_metrology.ladders contains IDs outside the half-unit inventory"
                )
            if selected is None:
                selected = self.options.get("selection")
            if selected is not None and (
                not isinstance(selected, list)
                or not selected
                or any(not isinstance(name, str) for name in selected)
                or len(set(selected)) != len(selected)
            ):
                raise ValueError(
                    "Optical ladder selection must be a non-empty list of unique IDs"
                )
            names = sorted(selected) if selected is not None else inventory
            if not names or set(names) - set(inventory):
                raise ValueError(
                    "Requested ladders are not in the STS8 half-unit inventory"
                )
            prepared = []
            for name in names:
                override = self.overrides.get(name, {})
                if not isinstance(override, dict) or set(override) - {
                    "marker_file",
                    "surface_file",
                    "sensor_file",
                    "nominal_sensor_frames",
                }:
                    raise ValueError(f"Invalid optical input overrides for {name}")
                paths = {
                    kind: self.resolve(override[kind + "_file"])
                    if kind + "_file" in override
                    else self.source_dir / name / f"{name}_{kind}.txt"
                    for kind in ("marker", "surface", "sensor")
                }
                records = {kind: file_record(path) for kind, path in paths.items()}
                layout = load_layout(name, connection)
                frames = override.get("nominal_sensor_frames", {})
                if not isinstance(frames, dict) or (
                    frames and set(frames) != set(layout.module_names)
                ):
                    raise ValueError(
                        f"{name}: explicit nominal_sensor_frames must cover every assembly module"
                    )
                descriptor = {
                    "name": name,
                    "files": records,
                    "modules": list(layout.module_names),
                    "sensor_dimensions_xy_mm": layout.sensors_size_xy().tolist(),
                    "sensor_positions_y_mm": layout.sensor_positions().tolist(),
                    "nominal_sensor_frames": frames,
                }
                prepared.append((descriptor, layout, paths))
        finally:
            if owned:
                connection.close()
        implementation = {
            path.name: file_record(path)["sha256"]
            for path in Path(__file__).parent.glob("*.py")
        }
        provenance = {
            "producer_version": PRODUCER_VERSION,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "implementation": implementation,
            "half_unit": self.half_unit,
            "assembly_naming": "STS8",
            "inventory": inventory,
            "sensor_frame": SENSOR_FRAME,
            "ladders": [item[0] for item in prepared],
        }
        return prepared, provenance

    @staticmethod
    def _nominal_sensor_frames(descriptor, layout, paths):
        explicit = descriptor["nominal_sensor_frames"]
        if explicit:
            return [
                rigid_matrix(explicit[module], module) for module in layout.module_names
            ]
        with paths["sensor"].open() as stream:
            header = stream.readline().strip()
        if not header.startswith("Size/D:X:Y:Z:"):
            raise ValueError(
                f"{paths['sensor']}: expected nominal sensor-summary columns Size/D:X:Y:Z"
            )
        summary = np.loadtxt(paths["sensor"], skiprows=1, ndmin=2)
        if (
            summary.shape[0] != layout.n_sensors
            or summary.shape[1] < 4
            or not np.isfinite(summary).all()
        ):
            raise ValueError(
                "Sensor summary does not contain one finite nominal pose per assembly sensor"
            )
        summary = summary[np.argsort(summary[:, 1])]
        centres = summary[:, [2, 1, 3]]
        expected_xy = np.column_stack(
            (np.zeros(layout.n_sensors), layout.sensor_positions())
        )
        if not np.allclose(
            centres[:, :2], expected_xy, atol=5e-4, rtol=0
        ) or not np.allclose(
            summary[:, 0], layout.sensors_size_xy()[:, 1], atol=5e-4, rtol=0
        ):
            raise ValueError(
                "Nominal sensor summary positions/dimensions do not match the assembly layout; supply explicit nominal_sensor_frames"
            )
        poses = []
        for centre in centres:
            pose = np.eye(4)
            pose[:3, 3] = centre
            poses.append(pose)
        return poses

    def _ladder_result(self, name, result, provenance, fingerprint):
        descriptor = next(
            ladder for ladder in provenance["ladders"] if ladder["name"] == name
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "producer": "sts_ladder_metrology",
            "producer_version": PRODUCER_VERSION,
            "half_unit": self.half_unit,
            "assembly_naming": "STS8",
            "ladder": name,
            "units": "mm",
            "direction": "nominal_to_measured",
            "sensor_frame": SENSOR_FRAME,
            "input_fingerprint": fingerprint,
            "result_digest": digest(result),
            "provenance": {
                "environment": provenance["environment"],
                "implementation": provenance["implementation"],
                "ladder": descriptor,
            },
            "analysis": result,
        }

    def run(self, selected=None, *, check=False):
        prepared, provenance = self._prepare(selected)
        fingerprint = digest(provenance)
        if check:
            if not self.result_path.is_file():
                raise FileNotFoundError(
                    f"Prepare optical metrology first: {self.result_path}"
                )
            payload = json.loads(self.result_path.read_text())
            if (
                payload.get("schema_version") != SCHEMA_VERSION
                or payload.get("input_fingerprint") != fingerprint
                or payload.get("provenance") != provenance
                or payload.get("result_digest") != digest(payload.get("ladders"))
            ):
                raise ValueError(
                    "Optical result is stale or modified; rerun sts_ladder_metrology with the current configuration"
                )
            for name, result in payload["ladders"].items():
                path = self.result_path.parent / name / "analysis.json"
                expected = self._ladder_result(name, result, provenance, fingerprint)
                if not path.is_file() or json.loads(path.read_text()) != expected:
                    raise ValueError(
                        f"Per-ladder optical result is stale or modified: {path}"
                    )
            return payload
        ladders = {}
        for descriptor, layout, paths in prepared:
            name = descriptor["name"]
            measurement = LadderMeasurement.load(
                name, paths["marker"], paths["surface"], layout=layout
            )
            if not np.isfinite(measurement.surface).all():
                raise ValueError(f"{name}: non-finite surface measurements")
            groups = measurement.grouped_sensor_measurements()
            poses = self._nominal_sensor_frames(descriptor, layout, paths)
            sensors = {}
            for module, group, pose, dimensions in zip(
                layout.module_names,
                groups,
                poses,
                layout.sensors_size_xy(),
                strict=True,
            ):
                data = group.markers_np_array()
                local = (data[:, :3] - pose[:3, 3]) @ pose[:3, :3]
                if np.any(np.abs(local[:, :2]) > np.asarray(dimensions) / 2 + 0.5):
                    raise ValueError(
                        f"{module}: nominal fiducials lie outside the associated sensor"
                    )
                fit = fit_sensor_deviation(group, pose)
                sensors[module] = {
                    **fit.to_dict(),
                    "sensor_frame": SENSOR_FRAME,
                    "nominal_sensor_to_survey": pose.tolist(),
                    "dimensions_xy_mm": dimensions.tolist(),
                }
            ladders[name] = {
                "module_order": list(layout.module_names),
                "sensors": sensors,
            }
        # Do not publish a result if sources changed while the fit was running.
        for descriptor, _, paths in prepared:
            if descriptor["files"] != {
                kind: file_record(path) for kind, path in paths.items()
            }:
                raise RuntimeError(
                    "Optical inputs changed during analysis; retry from a stable dataset"
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "producer": "sts_ladder_metrology",
            "producer_version": PRODUCER_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "half_unit": self.half_unit,
            "assembly_naming": "STS8",
            "units": "mm",
            "direction": "nominal_to_measured",
            "sensor_frame": SENSOR_FRAME,
            "method": "small_angle_fiducial_fit_with_exp_rotvec_export",
            "revision_policy": "Latest assembly module revisions; measurement-time revision correspondence must be verified.",
            "uncertainty_scope": "unweighted residual-scaled parameter covariance; excludes survey datum uncertainty",
            "provenance": provenance,
            "input_fingerprint": fingerprint,
            "ladders": ladders,
            "result_digest": digest(ladders),
        }
        for name, result in ladders.items():
            write_json_atomic(
                self.result_path.parent / name / "analysis.json",
                self._ladder_result(name, result, provenance, fingerprint),
            )
        # Publish the aggregate consumer contract last. A failed auxiliary write
        # therefore never advertises a complete new result set.
        write_json_atomic(self.result_path, payload)
        return payload
