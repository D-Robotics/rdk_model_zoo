# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Select exact YOLOE assets, verify local execution, and retain native run records."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
CPP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples._shared.assets import sha256_file, verify_asset_file
from samples._shared.platforms import require_execution_target
from samples.vision.yoloe.model.vocabulary import LABELS_SHA256
from samples.vision.yoloe.runtime.python.config import Config, validate_config
from samples.vision.yoloe.runtime.python.model_binding import (
    SAMPLE_DIR,
    list_models,
    resolve_selection,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    parser.add_argument("--variant", help="Defaults: x5/s100=11s, s100p=26n.")
    parser.add_argument(
        "--asset-id", help="Exact reference from the active publication manifest."
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--local-float-sha256",
        help="Expected digest of a custom float model; requires --model-path.",
    )
    parser.add_argument(
        "--test-img", type=Path, default=SAMPLE_DIR / "test_data/office_desk.jpg"
    )
    parser.add_argument(
        "--label-file", type=Path, default=SAMPLE_DIR / "test_data/classes.names"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/yoloe_cpp"),
        help="New directory; never overwrite an earlier run.",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="Use an existing native executable instead of the default build path.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Explicitly build before running, after identity and input checks.",
    )
    parser.add_argument("--score-thres", type=float, default=0.25)
    parser.add_argument("--nms-thres", type=float, help="E11 only; default 0.7.")
    parser.add_argument("--resize-type", type=int, choices=(0, 1), default=1)
    parser.add_argument("--max-det", type=int, default=300, help="E26 only, 1..8400.")
    parser.add_argument("--multi-label", action="store_true", help="E26 only.")
    parser.add_argument(
        "--no-morph",
        action="store_true",
        help="Disable the S E11 CLI default 5x5 opening.",
    )
    parser.add_argument("--no-contour", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def native_command(args, selection, binary, result_dir, digest):
    command = [
        str(binary),
        "--target",
        selection.target,
        "--variant",
        selection.variant,
        "--model-path",
        str(selection.model_path.resolve()),
        "--model-sha256",
        digest,
        "--test-img",
        str(args.test_img.expanduser().resolve()),
        "--label-file",
        str(args.label_file.expanduser().resolve()),
        "--output",
        str(result_dir),
        "--score-thres",
        str(args.score_thres),
        "--resize-type",
        str(args.resize_type),
        "--max-det",
        str(args.max_det),
    ]
    if args.nms_thres is not None:
        command += ["--nms-thres", str(args.nms_thres)]
    for name in ("no_morph", "no_contour", "multi_label"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def write_record(output, record):
    temporary = output / "launch-report.json.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output / "launch-report.json")


def run_logged(command, name, output, record):
    run = {
        "argv": command,
        "cwd": str(ROOT),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_log": name + ".stdout.log",
        "stderr_log": name + ".stderr.log",
    }
    record["processes"].append(run)
    write_record(output, record)
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    except OSError as exc:
        (output / (name + ".stdout.log")).write_text("")
        (output / (name + ".stderr.log")).write_text(str(exc) + "\n")
        run.update(
            returncode=None,
            error=str(exc),
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        raise
    stdout = (
        result.stdout.encode("utf-8")
        if isinstance(result.stdout, str)
        else result.stdout
    )
    stderr = (
        result.stderr.encode("utf-8")
        if isinstance(result.stderr, str)
        else result.stderr
    )
    (output / (name + ".stdout.log")).write_bytes(stdout)
    (output / (name + ".stderr.log")).write_bytes(stderr)
    sys.stdout.write(stdout.decode("utf-8", errors="replace"))
    sys.stderr.write(stderr.decode("utf-8", errors="replace"))
    run.update(
        returncode=result.returncode,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        stdout_log=name + ".stdout.log",
        stderr_log=name + ".stderr.log",
    )
    return result


def verify_result(result_dir, record):
    path = result_dir / "report.json"
    if result_dir.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Native command returned success without report.json")
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise ValueError("Native report must be a JSON object")
    expected = {
        "schema": "rdk-model-zoo/yoloe-native-run/v1",
        "execution_backend": "native-sdk",
        "target": record["target"],
        "variant": record["variant"],
        "model_sha256": record["model_sha256"],
        "image_sha256": record["input_sha256"],
        "vocabulary_sha256": LABELS_SHA256,
        "mask_layout": "roi",
        "image_saved": "annotated.png",
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "Native report identity/protocol differs from the requested run"
        )
    instances = report.get("instances")
    if (
        not isinstance(instances, list)
        or type(report.get("count")) is not int
        or report["count"] != len(instances)
    ):
        raise ValueError("Invalid native instance count")
    names = ["report.json", "annotated.png"]
    for item in instances:
        if not isinstance(item, dict):
            raise ValueError("Invalid native instance record")
        if item.get("mask") is not None:
            names.append(item["mask"])
    hashes = {}
    for name in names:
        if (
            not isinstance(name, str)
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
        ):
            raise ValueError("Unsafe native result path")
        member = result_dir / name
        if (
            member.is_symlink()
            or not member.resolve().is_relative_to(result_dir.resolve())
            or not member.is_file()
        ):
            raise ValueError(f"Missing or external native result file: {name}")
        hashes[name] = sha256_file(member)
    return hashes


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    record = None
    try:
        if args.list_models:
            print(
                json.dumps(
                    [
                        {
                            "target": target,
                            "variant": variant,
                            "asset_id": asset.reference,
                            "published_float": target == "x5",
                        }
                        for target, variant, asset in list_models(args.target)
                    ],
                    indent=2,
                )
            )
            return 0
        if args.build and args.binary:
            raise ValueError("--build and --binary cannot be combined")
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit --target")
        selection = resolve_selection(
            args.target,
            variant=args.variant,
            asset_id=args.asset_id,
            model_path=args.model_path,
            local_float_sha256=args.local_float_sha256,
        )
        config = Config(
            args.score_thres,
            args.nms_thres,
            args.resize_type,
            selection.target != "x5"
            and selection.variant.startswith("11")
            and not args.no_morph,
            args.max_det,
            not args.multi_label,
        )
        validate_config(selection, config)
        build_dir = CPP / "build" / selection.target
        binary = (
            args.binary.expanduser().resolve()
            if args.binary
            else build_dir / "yoloe_demo"
        )
        destination = args.output.expanduser().resolve()
        digest = (
            selection.local_float_sha256
            or selection.asset.sha256
            or "<verified-model-sha256>"
        )
        record = {
            "schema": "rdk-model-zoo/yoloe-native-launch/v1",
            "target": selection.target,
            "variant": selection.variant,
            "source_asset_id": selection.asset.reference,
            "model_kind": "local-float" if selection.local_float else "published",
            "publisher_sha256": selection.asset.sha256,
            "publisher_checksum_verified": False,
            "local_float_sha256": selection.local_float_sha256,
            "config": asdict(config),
            "cwd": str(ROOT),
            "build_directory": str(build_dir),
            "command": native_command(
                args, selection, binary, destination / "result", digest
            ),
            "executed": False,
            "downloaded": False,
            "runtime_metadata_verified": False,
            "processes": [],
            "status": "prepared",
        }
        if args.dry_run:
            record["status"] = (
                "requires local floating-output conversion"
                if not (selection.published_float or selection.local_float)
                else "metadata validation required"
            )
            print(json.dumps(record, indent=2))
            return 0
        if not selection.published_float and not selection.local_float:
            raise ValueError(
                "Published S YOLOE outputs are quantized; prepare and identify a local floating-output model first"
            )
        record["detected_target"] = require_execution_target(selection.target)
        if selection.local_float:
            digest = sha256_file(selection.model_path)
            if digest != selection.local_float_sha256:
                raise ValueError("Local float SHA-256 mismatch")
        else:
            digest = verify_asset_file(selection.asset, selection.model_path)
            record["publisher_checksum_verified"] = selection.asset.sha256 is not None
        if (
            not selection.model_path.is_file()
            or selection.model_path.stat().st_size == 0
        ):
            raise ValueError("Missing or empty model file")
        image, labels = (
            args.test_img.expanduser().resolve(),
            args.label_file.expanduser().resolve(),
        )
        if not image.is_file() or image.stat().st_size == 0:
            raise ValueError(f"Missing or empty image: {image}")
        if sha256_file(labels) != LABELS_SHA256:
            raise ValueError("Vocabulary SHA-256 mismatch")
        if (
            args.output.expanduser().is_symlink()
            or destination.exists()
            or destination.is_symlink()
        ):
            raise ValueError("Output directory must be new")
        if not args.build and (not binary.is_file() or not os.access(binary, os.X_OK)):
            raise ValueError(
                "Native executable missing; use --build on the target SDK or supply --binary"
            )
        destination.mkdir(parents=True)
        output = destination
        record.update(
            model_sha256=digest,
            input_sha256=sha256_file(image),
            vocabulary_sha256=LABELS_SHA256,
            command=native_command(
                args, selection, binary, destination / "result", digest
            ),
            status="running",
        )
        if not selection.local_float and selection.asset.sha256 is None:
            record["identity_note"] = (
                "Observed model digest binds bytes; no publisher checksum is recorded to authenticate origin"
            )
        if args.build:
            commands = [
                (
                    "configure",
                    [
                        "cmake",
                        "-S",
                        str(CPP),
                        "-B",
                        str(build_dir),
                        "-DCMAKE_BUILD_TYPE=Release",
                        "-DYOLOE_BUILD_CLI=ON",
                        "-DYOLOE_BUILD_TESTS=OFF",
                        "-DYOLOE_SANITIZERS=OFF",
                    ],
                ),
                ("build", ["cmake", "--build", str(build_dir), "--parallel", "2"]),
            ]
            for name, command in commands:
                if run_logged(command, name, output, record).returncode:
                    raise RuntimeError(f"Native {name} failed; see retained logs")
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("Native executable missing after build")
        record["binary_sha256"] = sha256_file(binary)
        result = run_logged(record["command"], "native", output, record)
        record.update(executed=True, native_returncode=result.returncode)
        if result.returncode:
            record["status"] = "failed"
            write_record(output, record)
            return result.returncode if result.returncode > 0 else 2
        record["result_sha256"] = verify_result(output / "result", record)
        record.update(status="completed", runtime_metadata_verified=True)
        write_record(output, record)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        if output is not None and record is not None:
            record.update(status="failed", error=str(exc))
            try:
                write_record(output, record)
            except OSError as log_error:
                print(f"error writing launch record: {log_error}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
