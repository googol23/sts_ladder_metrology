# STS optical ladder metrology

`sts_ladder_metrology` owns optical fiducial loading, sensor grouping and fitting.
It returns **sensor-local nominal-to-measured transformations** for the ladders
in a half-unit. `PointCloudMetrology` executes this module externally and consumes
its result; it does not duplicate the optical analysis.

## Run from the workspace

Use either the shared **`sts_tools/config.yaml`** or the complete U3DL dataset
configuration at **`PointCloudMetrology/input/u3dl/config.yaml`**. All relative
paths resolve from the selected YAML's directory, not the calling directory.

```sh
python -m pip install -e ./sts_utils -e './sts_geo_layout[assembly,plot]'
python -m sts_ladder_metrology.main --config config.yaml
python -m sts_ladder_metrology.main --config config.yaml --check
```

The optical interpreter also needs NumPy, SciPy, Matplotlib and PyYAML. PostgreSQL
access uses the existing `sts_db_api` private credentials/environment mechanism.
Run the module from the workspace root so the workspace modules are importable;
shared packages must be installed in the external interpreter's environment.
No credentials are written to results.

`--check` validates the existing artifact against current measurement hashes,
assembly inventory/module revisions/layout, configuration, producer source and
numerical environment. It does **not** fit or modify the result. It still requires
access to current source files and the read-only assembly database.

By default every ladder in the half-unit is processed, with literal **STS8** names
and no STS3_5 renumbering. An explicit subset can be selected with repeated
`--ladder` options, or with `ladder_metrology.selection` in the same YAML. A subset
replaces the result artifact's coverage; prepare/check it with the same selection.
There is no first-20 limit and no silently skipped missing/failed sensor.
Module revisions come from the **current** assembly database. Verify that the
selected historical survey belongs to those revisions; this correspondence is
not inferred from filenames. The result records that revision-policy limitation.

## Configuration

```yaml
half_unit:
  name: U3DL
  assembly_naming: STS8

ladder_metrology:
  enabled: true
  source_dir: /u/dlmeas/STS
  result_file: output/U3DL/ladder_metrology/sensor_transforms.json
  sensor_frame: nominal_sensor_local_v1
  mode: run
  execution:
    python: /u/dramirez/micromamba/envs/cbm/bin/python
    working_dir: .
    timeout_seconds: 120
  ladders: {}
```

`result_file` is the explicit producer/consumer boundary. The producer atomically
writes schema-version-1 JSON there; the CLI reports the same absolute path and its
input/result digests on stdout. The point-cloud driver verifies that receipt,
reads that exact file, validates its schema and integrity, and writes
`execution.log` beside it for a normal run. Legacy configurations using
`output_dir` remain supported and resolve to `output_dir/sensor_transforms.json`,
but `result_file` and `output_dir` cannot be specified together.

`enabled`, `mode` and `execution` control the point-cloud driver; an explicit
standalone CLI invocation performs the requested run/check. `execution.working_dir`
sets the child process directory, allowing a dataset-local YAML to launch the
workspace module without changing artifact path resolution. `run` regenerates
optical results during normal pipeline processing; `reuse` validates existing
results without fitting. Registration preview always checks/reuses and never
runs an optical fit. Missing/stale results are errors, not identity fallbacks.

The default inputs per ladder are:

```text
source_dir/<STS8 ladder>/<STS8 ladder>_marker.txt
source_dir/<STS8 ladder>/<STS8 ladder>_surface.txt
source_dir/<STS8 ladder>/<STS8 ladder>_sensor.txt
```

Per-ladder `marker_file`, `surface_file` and `sensor_file` overrides live under
`ladder_metrology.ladders.<STS8 ladder>`. They use the root YAML's path base.
Surface data are loaded/validated and hashed but are **not fitted**. The sensor
summary supplies nominal reference coordinates only; its precomputed deviations
and errors are **not** inputs to the new fiducial fit.

## Sensor-frame and transformation contract

`nominal_sensor_local_v1` is an explicitly defined **nominal** sensor frame:

- Default origin: nominal `X:Y:Z` from that sensor's `_sensor.txt` summary.
- Default axes: converted survey axes, `x = table Y`, `y = table X`, `z = table Z`.
- Translation lengths: mm. Rotation vectors/covariance angular coordinates: rad.
- Active transform direction: nominal geometry **toward** the measured geometry.

The summary is sorted by nominal table X and checked against assembly-derived
bottom-to-top sensor positions and dimensions before associating module IDs.
The nominal summary XYZ is a survey reference; its Z must not be assumed to be
the ROOT box's thickness midpoint. Association to a ROOT box frame is an explicit
consumer-side assertion, including origin offsets, axis orientation and parity.
The consumer bridge may explicitly reflect an orthogonal coordinate basis; the
exported nominal-to-measured deviation itself is always a proper rigid transform.

If nominal sensor poses cannot be obtained with this default, supply a complete
`nominal_sensor_frames` mapping for that ladder. Each assembly module maps to a
proper 4x4 **sensor-local to converted-survey** pose in mm. No sensor origin or
orientation is inferred from the centroid of the measured fiducials.

The existing X/Y coordinate conversion is performed **before** fitting. Since a
bare axis swap has negative determinant, rotation-vector components must never
be converted by simply swapping their entries. Both nominal coordinates and
measured deviations are converted, and fitting is then done in the declared
sensor-local axes.

For a nominal point `q` in the sensor frame:

```text
q_measured = Delta_sensor @ q_nominal
T_halfunit_sensor_corrected = T_halfunit_sensor_nominal @ Delta_sensor
```

Internally the existing small-angle estimator fits translation `d` and rotation
vector `w` about nominal fiducial centroid `c`. The exported finite transform is:

```text
R = exp([w]x)
t = c + d - R @ c
Delta_sensor = [[R, t], [0, 1]]
```

This is not an Euler-angle interpretation and not an inverse alignment
correction. Finite-transform residuals are re-evaluated and reported alongside
linear-fit RMS. Finite export is a small-angle approximation, not a newly fitted
nonlinear solution.

Covariance is unweighted, residual-scaled fit covariance, first-order propagated
to `[tx, ty, tz, rotvec_x, rotvec_y, rotvec_z]` at the sensor origin. It is **not** a
covariance of matrix entries or an instrument-calibrated absolute uncertainty.
Survey datum/systematic uncertainty is excluded.

## API and artifact

The external boundary has two JSON objects:

1. The CLI writes one receipt to stdout with `result_file`, `input_fingerprint`,
   `result_digest`, `ladders`, `sensor_count`, and `checked_only`.
2. `result_file` contains the durable schema-version-1 artifact. Required top-level
   identity fields are `producer: sts_ladder_metrology`, `half_unit`,
   `assembly_naming: STS8`, `units: mm`, `direction: nominal_to_measured`, and
   `sensor_frame: nominal_sensor_local_v1`. `input_fingerprint` covers current
   measurements, assembly snapshot, implementation, and numerical environment;
   `result_digest` protects the complete `ladders` mapping.

Each `ladders.<STS8 ladder>` object contains an exact `module_order` and a `sensors`
mapping keyed by STS8 module ID. Every sensor contains `status: ok`, the 4x4 proper
rigid `nominal_to_measured` matrix, its 4x4 `nominal_sensor_to_survey` reference
pose, `dimensions_xy_mm`, a finite symmetric 6x6 `covariance`, and the fixed
parameter/unit order `[tx, ty, tz, rotvec_x, rotvec_y, rotvec_z]` /
`[mm, mm, mm, rad, rad, rad]`, plus fit diagnostics. PointCloudMetrology validates
these identities, matrix/covariance properties, module coverage, receipt path,
fingerprint, and digest before it returns any result to geometry preparation.

- `LadderMetrologyAnalysis(config, config_dir, connection=None).run(selected=None,
check=False)` owns the half-unit workflow. Supplied connections remain owned by
  the caller; owned database connections are explicitly read-only and closed.
- `LadderMeasurement.grouped_sensor_measurements()` is shared by legacy diagnostic
  plotting and the new analysis, avoiding duplicate grouping logic.
- `fit_sensor_deviation()` reuses `SensorMeasurements.fit_deviations()` after
  converting nominal points and deviations to the sensor frame.
- `SensorDeviation` holds the finite transformation, covariance and diagnostics.

The atomically published aggregate result is the configured `result_file` (or the
legacy `output_dir/sensor_transforms.json` fallback), schema version 1. It includes
half-unit identity, STS8 ladder/module IDs, transform/frame conventions,
full-precision matrices and covariance, nominal reference poses, residuals,
assembly snapshot, source hashes, implementation/environment provenance, an input
fingerprint and a result-data digest. The producer also writes one independently
checkable `analysis.json` per selected STS8 ladder:

```text
output/U3DL/ladder_metrology/
├── sensor_transforms.json
├── execution.log                 # Added by a normal consumer-driven run
├── L3DL100112/analysis.json
├── L3DL300112/analysis.json
└── L3DL500114/analysis.json
```

The aggregate consumer contract is published last, so an auxiliary per-ladder
write failure cannot advertise a complete new result set. `--check` validates both
the aggregate and every per-ladder file. The CLI prints a JSON receipt; logs go to
stderr. Failed analysis does not publish a partial-success aggregate artifact.

Legacy plotting/fitting methods and legacy YAML dumper types remain available,
but rounded 12-number YAML is **not** the external integration contract.

## Production U3DL boundary

The workspace root configuration preserves the existing laser settings and
provisional `L3UL*` aliases. These are not assumed to identify the STS8 `L3DL*`
assembly ladders. Before corrected preview/registration, supply each verified
`assembly_ladder`, complete `registration.optical_sensor_map`, and matching ROOT
geometry/frame. No automatic X-order/sensor-count pairing is performed. See
`PointCloudMetrology/REGISTRATION.md` for the consumer contract.
