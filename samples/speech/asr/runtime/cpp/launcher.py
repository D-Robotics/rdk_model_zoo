# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Resolve published ASR assets and retain native build/run evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
CPP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.assets import sha256_file, verify_asset_file
from utils.py_utils.platforms import require_execution_target
from samples.speech.asr.runtime.python.cli import (
    SAMPLE_DIR,
    list_available_assets,
    resolve_selection,
)
from samples.speech.asr.runtime.python.cli import load_vocabulary, SHA256


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", type=Path)
    p.add_argument(
        "--audio-file", type=Path, default=SAMPLE_DIR / "test_data/chi_sound.wav"
    )
    p.add_argument(
        "--vocab-file", type=Path, default=SAMPLE_DIR / "test_data/vocab.json"
    )
    p.add_argument("--decode-mode", choices=("ctc", "legacy"), default="legacy")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/asr_cpp"))
    build = p.add_mutually_exclusive_group()
    build.add_argument("--build", action="store_true")
    build.add_argument("--binary", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def validate_report(report, record):
    expected = dict(
        schema="rdk-model-zoo/asr-native-run/v1",
        status="completed",
        execution_backend="native-sdk",
        vocabulary_sha256=SHA256,
    )
    expected.update(
        {
            key: record[key]
            for key in (
                "target",
                "asset_id",
                "model_sha256",
                "audio_sha256",
                "decode_mode",
            )
        }
    )
    if not isinstance(report, dict) or any(
        report.get(k) != v for k, v in expected.items()
    ):
        raise ValueError("Native report identity/backend/status mismatch")
    if report.get("config") != {"audio_maxlen": 30000, "new_rate": 16000}:
        raise ValueError("Native frontend configuration mismatch")
    metadata = report.get("metadata", {})
    shape = metadata.get("output_shape") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("input_shape") != [1, 30000]
        or any(type(v) is not int for v in metadata.get("input_shape", []))
        or metadata.get("dtype") != "float32"
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(type(v) is not int for v in shape)
        or shape[0] != 1
        or shape[1] <= 0
        or shape[2] != 3503
        or not isinstance(metadata.get("model_name"), str)
        or not metadata["model_name"]
    ):
        raise ValueError("Invalid native tensor metadata")
    for prefix, dimensions in [("input", [1, 30000]), ("output", shape)]:
        strides = metadata.get(prefix + "_strides")
        size = metadata.get(prefix + "_bytes")
        if (
            not isinstance(strides, list)
            or len(strides) != len(dimensions)
            or type(size) is not int
            or size <= 0
        ):
            raise ValueError("Invalid native allocation metadata")
        span = 4
        for dimension, stride in reversed(list(zip(dimensions, strides))):
            if (
                type(stride) is not int
                or stride < span
                or stride % 4
                or stride * dimension > size
            ):
                raise ValueError("Invalid native byte strides")
            span = stride * dimension
    chunks = report.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Expected native audio chunks")
    offset = 0
    rate = None
    texts = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or any(
            type(chunk.get(k)) is not int
            for k in (
                "index",
                "source_start",
                "source_frames",
                "source_rate",
                "valid_target_samples",
            )
        ):
            raise ValueError("Invalid native chunk fields")
        if (
            chunk["index"] != index
            or chunk["source_start"] != offset
            or chunk["source_frames"] <= 0
            or chunk["source_rate"] <= 0
            or not 0 < chunk["valid_target_samples"] <= 30000
            or not isinstance(chunk.get("text"), str)
        ):
            raise ValueError("Invalid native chunk geometry/text")
        if rate is not None and rate != chunk["source_rate"]:
            raise ValueError("Source sample rate changed across chunks")
        rate = chunk["source_rate"]
        if chunk["source_frames"] > (30000 * rate + 15999) // 16000:
            raise ValueError("Native chunk exceeds fixed window")
        capacity = min(30000, (chunk["source_frames"] * 16000 * 2 + rate) // (2 * rate))
        if chunk["valid_target_samples"] > capacity:
            raise ValueError("Native valid length exceeds resampling capacity")
        offset += chunk["source_frames"]
        texts.append(chunk["text"])
    if report.get("text") != "".join(texts):
        raise ValueError("Native whole-file transcript differs from its chunks")


def write_record(output, record):
    temporary = output / "launch-report.json.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output / "launch-report.json")


def run_logged(argv, name, output, record):
    run = {
        "argv": argv,
        "cwd": str(ROOT),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_log": name + ".stdout.log",
        "stderr_log": name + ".stderr.log",
    }
    record["processes"].append(run)
    write_record(output, record)
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    except OSError as error:
        run.update(
            returncode=None,
            error=str(error),
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        (output / run["stdout_log"]).write_bytes(b"")
        (output / run["stderr_log"]).write_text(str(error) + "\n")
        raise
    if name == "native":
        record["executed"] = True
    (output / run["stdout_log"]).write_bytes(result.stdout)
    (output / run["stderr_log"]).write_bytes(result.stderr)
    run.update(
        returncode=result.returncode,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    sys.stdout.write(result.stdout.decode("utf-8", errors="replace"))
    sys.stderr.write(result.stderr.decode("utf-8", errors="replace"))
    write_record(output, record)
    if result.returncode:
        raise RuntimeError(f"{name} failed (rc={result.returncode}); see retained logs")


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    record = None
    try:
        if args.list_models:
            print(
                json.dumps(
                    [
                        {"asset_id": a.reference, "target": a.filename.split("/")[0]}
                        for a in list_available_assets(args.target)
                    ],
                    indent=2,
                )
            )
            return 0
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit target")
        selection = resolve_selection(
            args.target, asset_id=args.asset_id, model_path=args.model_path
        )
        build_dir = CPP / "build" / selection.target
        binary = (
            args.binary.expanduser().resolve()
            if args.binary
            else build_dir / "asr_demo"
        )
        destination = args.output_dir.expanduser().resolve()

        def command(digest):
            return [
                str(binary),
                "--target",
                selection.target,
                "--asset-id",
                selection.asset.reference,
                "--model-path",
                str(selection.model_path.resolve()),
                "--model-sha256",
                digest,
                "--audio-file",
                str(args.audio_file.expanduser().resolve()),
                "--vocab-file",
                str(args.vocab_file.expanduser().resolve()),
                "--decode-mode",
                args.decode_mode,
                "--output-dir",
                str(destination / "result"),
            ]

        record = {
            "schema": "rdk-model-zoo/asr-native-launch/v1",
            "target": selection.target,
            "asset_id": selection.asset.reference,
            "decode_mode": args.decode_mode,
            "publisher_sha256": selection.asset.sha256,
            "publisher_checksum_verified": False,
            "executed": False,
            "downloaded": False,
            "runtime_metadata_verified": False,
            "status": "prepared",
            "command": command(selection.asset.sha256 or "<observed-model-sha256>"),
            "processes": [],
        }
        if args.dry_run:
            print(json.dumps(record, indent=2))
            return 0
        record["detected_target"] = require_execution_target(selection.target)
        model = selection.model_path
        if not model.is_file() or model.stat().st_size == 0:
            raise ValueError("Missing or empty model file; download explicitly first")
        digest = verify_asset_file(selection.asset, model)
        load_vocabulary(args.vocab_file)
        audio = args.audio_file.expanduser()
        if not audio.is_file() or audio.stat().st_size == 0:
            raise ValueError("Missing or empty audio file")
        if (
            args.output_dir.expanduser().is_symlink()
            or destination.exists()
            or destination.is_symlink()
        ):
            raise ValueError("Output directory must be new")
        if not args.build and (not binary.is_file() or not os.access(binary, os.X_OK)):
            raise ValueError("Native executable missing; use --build or --binary")
        record.update(
            model_sha256=digest,
            audio_sha256=sha256_file(audio),
            vocabulary_sha256=SHA256,
            publisher_checksum_verified=selection.asset.sha256 is not None,
            command=command(digest),
            status="running",
        )
        destination.mkdir(parents=True)
        output = destination
        write_record(output, record)
        if args.build:
            run_logged(
                [
                    "cmake",
                    "-S",
                    str(CPP),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DASR_BUILD_SDK=ON",
                    "-DASR_BUILD_CLI=ON",
                    "-DASR_BUILD_TESTS=OFF",
                ],
                "configure",
                output,
                record,
            )
            run_logged(
                ["cmake", "--build", str(build_dir), "--parallel", "2"],
                "build",
                output,
                record,
            )
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("Native executable missing after build")
        record["binary_sha256"] = sha256_file(binary)
        run_logged(record["command"], "native", output, record)
        record["executed"] = True
        path = output / "result/result.json"
        if (output / "result").is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("Native process returned success without result.json")
        report = json.loads(path.read_text())
        validate_report(report, record)
        if sha256_file(audio) != record["audio_sha256"] or sha256_file(model) != digest:
            raise ValueError("Input/model changed during native run")
        record.update(
            status="completed",
            runtime_metadata_verified=True,
            result_sha256=sha256_file(path),
        )
        write_record(output, record)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        if output is not None and record is not None:
            record.update(status="failed", error=str(error))
            try:
                write_record(output, record)
            except OSError as log_error:
                print(f"Could not save launch report: {log_error}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
