# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""S100 native launcher with explicit build and captured run provenance."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
CPP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import sha256_file, verify_asset_file
from samples._shared.platforms import require_execution_target
from samples.vision.lanenet.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_available_assets,
    resolve_selection,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="s100"
    )
    parser.add_argument("--asset-id")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--test-img", type=Path, default=SAMPLE_DIR / "test_data/lane.jpg"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/lanenet_cpp"))
    parser.add_argument("--instance-save-path", type=Path)
    parser.add_argument("--binary-save-path", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build explicitly after identity and input checks",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def validate_output_paths(output, extras):
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    reserved = {
        "embedding.npy",
        "binary.npy",
        "instance_pred.png",
        "binary_pred.png",
        "report.json",
        "launch-report.json",
        "native.stdout.log",
        "native.stderr.log",
    }
    if len(set(extras)) != len(extras):
        raise ValueError("Additional image paths must be distinct")
    for path in extras:
        if (
            path.exists()
            or path == output
            or (
                path.parent == output
                and (path.name in reserved or path.name.startswith("raw_output_"))
            )
        ):
            raise ValueError(
                f"Additional image path exists or conflicts with output records: {path}"
            )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        if args.list_models:
            print(
                json.dumps(
                    [
                        {"asset_id": a.reference, "url": a.url, "sha256": a.sha256}
                        for a in list_available_assets(selection.target)
                    ],
                    indent=2,
                )
            )
            return 0
        if args.build and args.binary:
            raise ValueError("--build and --binary cannot be combined")
        build_dir = CPP / "build/s100"
        binary = (
            args.binary.expanduser().resolve() if args.binary else build_dir / "lanenet"
        )
        model = selection.model_path.resolve()
        image = args.test_img.expanduser().resolve()
        output = args.output.expanduser().resolve()
        extras = {
            name: path.expanduser().resolve()
            for name, path in (
                ("--instance-save-path", args.instance_save_path),
                ("--binary-save-path", args.binary_save_path),
            )
            if path is not None
        }
        command = [
            str(binary),
            "--target",
            "s100",
            "--model-path",
            str(model),
            "--test-img",
            str(image),
            "--output",
            str(output),
        ]
        for name, path in extras.items():
            command.extend([name, str(path)])
        evidence = {
            "target": "s100",
            "asset_id": selection.asset.reference,
            "publisher_sha256": selection.asset.sha256,
            "identity_note": "Observed file digest does not authenticate a publisher asset without a published checksum",
            "command": command,
            "cwd": str(ROOT),
            "build_directory": str(build_dir),
            "executed": False,
            "downloaded": False,
        }
        if args.dry_run:
            print(json.dumps(evidence, indent=2))
            return 0
        require_execution_target("s100")
        model_digest = verify_asset_file(selection.asset, model)
        if not image.is_file():
            raise ValueError(f"Input image not found: {image}")
        validate_output_paths(output, list(extras.values()))
        if args.build:
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(CPP),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DLANENET_TARGET=s100",
                ],
                check=True,
                cwd=ROOT,
            )
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--parallel", "2"],
                check=True,
                cwd=ROOT,
            )
        if not binary.is_file():
            raise ValueError(
                "Native binary missing; use --build in the S100 SDK environment or supply --binary"
            )
        evidence.update(
            model_sha256=model_digest,
            input_sha256=sha256_file(image),
            binary_sha256=sha256_file(binary),
            started_utc=datetime.now(timezone.utc).isoformat(),
        )
        result = subprocess.run(
            command, check=False, text=True, capture_output=True, cwd=ROOT
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        evidence.update(
            executed=True,
            returncode=result.returncode,
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        if output.is_dir():
            (output / "native.stdout.log").write_text(result.stdout)
            (output / "native.stderr.log").write_text(result.stderr)
            if (output / "report.json").is_file():
                evidence["native_report_sha256"] = sha256_file(output / "report.json")
            (output / "launch-report.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )
        if result.returncode == 0 and not (output / "report.json").is_file():
            raise ValueError("Native command returned success without its report")
        return result.returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
