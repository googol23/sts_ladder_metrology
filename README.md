# STS ladder metrology

`sts_ladder_metrology` determines how sensor fiducials deviate from their nominal
positions on an STS ladder.

The intended analysis supports two representations of the optical-metrology input:

- per-fiducial deviations from nominal positions; or
- measured fiducial coordinates expressed in the optical metrology table frame.

The package converts table-frame measurements where required, groups measurements by
sensor, fits the measured sensor geometry, and produces ladder-alignment parameters
relative to the nominal layout supplied by `sts_geo_layout`.

## Workspace usage

Install the shared `sts_utils` and `sts_geo_layout` packages in the active environment,
then run the current analysis entry point from the `sts_tools` workspace root:

```sh
python -m sts_ladder_metrology.main
```

Input measurement files are read from the workspace `sources/` directory. The current
implementation recognizes sensor, marker, and surface measurement files by filename.
