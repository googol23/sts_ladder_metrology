"""External half-unit optical-metrology CLI, configured by the workspace YAML."""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml


# Kept for reading/reproducing legacy run artifacts, not the new result contract.
class NoWrapDumper(yaml.SafeDumper):
    pass


class FlowList(list):
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def float_representer(dumper, value):
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.6f}")


yaml.add_representer(float, float_representer, Dumper=NoWrapDumper)
yaml.add_representer(FlowList, flow_list_representer, Dumper=NoWrapDumper)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--ladder",
        action="append",
        help="Explicit STS8 subset; repeat per ladder. Default: full half-unit.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate current sources, assembly snapshot and existing artifact without fitting.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        from .analysis import LadderMetrologyAnalysis

        path = args.config.resolve()
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict):
            raise ValueError("Configuration root must be a mapping")
        analysis = LadderMetrologyAnalysis(config, path.parent)
        result = analysis.run(args.ladder, check=args.check)
        print(
            json.dumps(
                {
                    "result_file": str(analysis.result_path),
                    "input_fingerprint": result["input_fingerprint"],
                    "result_digest": result["result_digest"],
                    "ladders": list(result["ladders"]),
                    "sensor_count": sum(
                        len(ladder["sensors"]) for ladder in result["ladders"].values()
                    ),
                    "checked_only": args.check,
                }
            )
        )
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"Optical ladder metrology failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
