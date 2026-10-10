# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check one plate against an independently transcribed input reference."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    """Run one plate check and save the reference/input/model identities.

    Args:
        argv: Optional CLI arguments; None reads sys.argv.

    Returns:
        int: Zero for exact reference agreement, one for mismatch or model error.

    Raises:
        OSError: Input or output files cannot be read or written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--test-bin", type=Path, required=True)
    parser.add_argument("--reference-plate", required=True)
    parser.add_argument("--reference-source", required=True,
                        help="Visual input-image transcription provenance, established before inference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from samples.vision.lprnet.runtime.python.cli import resolve_selection
    from samples.vision.lprnet.runtime.python.lprnet import LPRNetRecognizer
    report = {"schema": "rdk-model-zoo/lpr-reference/v1", "target": args.target,
              "asset_id": args.asset_id, "reference_plate": args.reference_plate,
              "reference_source": args.reference_source, "scope": "one input plate; not dataset accuracy",
              "input_sha256": hashlib.sha256(args.test_bin.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(args.model_path.read_bytes()).hexdigest()}
    try:
        selection = resolve_selection(args.target, asset_id=args.asset_id, model_path=args.model_path)
        model = LPRNetRecognizer(selection)
        report["plate"] = model.predict(args.test_bin)
        report["passed"] = report["plate"] == args.reference_plate
    except Exception as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
