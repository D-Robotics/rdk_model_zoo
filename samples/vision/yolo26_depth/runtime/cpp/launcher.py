# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""X5-only native build/run with explicit identity, artifact and provenance gates."""

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
from samples.vision.yolo26_depth.runtime.python.model_binding import (
    SAMPLE_DIR,
    TARGETS,
    VARIANTS,
    list_available_assets,
    resolve_selection,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("auto",) + TARGETS, default="x5")
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--converted-model", action="store_true")
    p.add_argument("--test-img", type=Path, default=SAMPLE_DIR / "test_data/bus.jpg")
    p.add_argument("--output", type=Path, default=Path("outputs/yolo26_depth_cpp"))
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Native source default: no warmup; Python defaults to 3",
    )
    p.add_argument("--binary", type=Path)
    p.add_argument(
        "--build",
        action="store_true",
        help="Explicit CMake build after identity/artifact checks",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.target not in ("auto", "x5"):
            raise ValueError(
                "YOLO26 Depth C++ supports x5 only; use Python for S targets"
            )
        if args.list_models:
            print(
                json.dumps(
                    [
                        {"asset_id": a.reference, "sha256": a.sha256, "url": a.url}
                        for a in list_available_assets("x5")
                    ],
                    indent=2,
                )
            )
            return 0
        selection = resolve_selection(
            args.target,
            variant=args.variant,
            asset_id=args.asset_id,
            model_path=args.model_path,
            converted_model=args.converted_model,
        )
        if selection.target != "x5":
            raise ValueError("Native asset contract must target x5")
        if args.warmup < 0:
            raise ValueError("warmup must be nonnegative")
        if args.build and args.binary:
            raise ValueError("--build uses the sample build directory; omit --binary")
        build_dir = CPP / "build/x5"
        binary = (
            args.binary.expanduser().resolve()
            if args.binary
            else build_dir / "yolo26_depth"
        )
        model = selection.model_path.resolve()
        image, output = (
            args.test_img.expanduser().resolve(),
            args.output.expanduser().resolve(),
        )
        command = [
            str(binary),
            "--target",
            "x5",
            "--model-path",
            str(model),
            "--test-img",
            str(image),
            "--output",
            str(output),
            "--warmup",
            str(args.warmup),
        ]
        evidence = {
            "target": "x5",
            "variant": selection.variant,
            "profile": "nv12",
            "asset_id": (
                None if selection.converted_model else selection.asset.reference
            ),
            "contract_reference": selection.asset.reference,
            "artifact_origin": (
                "user-converted" if selection.converted_model else "published-manifest"
            ),
            "publisher_sha256": (
                None if selection.converted_model else selection.asset.sha256
            ),
            "command": command,
            "build_directory": str(build_dir),
            "executed": False,
            "downloaded": False,
        }
        if args.dry_run:
            print(json.dumps(evidence, indent=2))
            return 0
        require_execution_target("x5")
        if selection.converted_model:
            if not model.is_file() or not model.stat().st_size:
                raise ValueError("Converted model must be a nonempty file")
            model_digest = sha256_file(model)
        else:
            model_digest = verify_asset_file(selection.asset, model)
        if not image.is_file():
            raise ValueError(f"Input image not found: {image}")
        if output.exists():
            raise FileExistsError(f"Output directory must be new: {output}")
        if args.build:
            subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(CPP),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", str(build_dir), "--parallel", "2"], check=True
            )
        if not binary.is_file():
            raise ValueError(
                "Native binary missing; build explicitly with --build on the X5 SDK environment"
            )
        evidence.update(
            model_sha256=model_digest,
            input_sha256=sha256_file(image),
            binary_sha256=sha256_file(binary),
            cwd=str(Path.cwd()),
            started_utc=datetime.now(timezone.utc).isoformat(),
        )
        result = subprocess.run(command, check=False, text=True, capture_output=True)
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
            native_report = output / "report.json"
            if native_report.is_file():
                evidence["native_report_sha256"] = sha256_file(native_report)
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
