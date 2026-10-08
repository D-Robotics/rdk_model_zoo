# Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
"""Resolve published Paraformer artifacts and run the native executable."""

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
from samples.speech.paraformer.runtime.python.model_binding import (
    SAMPLE_DIR,
    STAGES,
    VOCABULARY_DIGEST,
    resolve_selections,
)
from samples.speech.paraformer.runtime.python.decoding import validate_vocabulary
from samples.speech.paraformer.runtime.cpp.native_report import validate_report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    for stage in STAGES:
        p.add_argument(f"--{stage}-model-path", type=Path)
        p.add_argument(f"--{stage}-asset-id")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/paraformer_features/prepared-manifest.json"),
    )
    p.add_argument(
        "--vocab-file", type=Path, default=SAMPLE_DIR / "model/s100/tokens.json"
    )
    p.add_argument("--max-utts", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/paraformer_cpp"))
    build = p.add_mutually_exclusive_group()
    build.add_argument("--build", action="store_true")
    build.add_argument("--binary", type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--list-models", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return p


def write_report(output, record):
    temporary = output / "launch-report.json.tmp"
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(output / "launch-report.json")


def run_logged(argv, name, output, record):
    process = {
        "argv": argv,
        "cwd": str(ROOT),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["processes"].append(process)
    write_report(output, record)
    try:
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=False)
    except OSError as error:
        process.update(
            returncode=None,
            error=str(error),
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        raise
    if name == "native":
        record["native_process_started"] = True
    for stream in ("stdout", "stderr"):
        path = name + "." + stream + ".log"
        (output / path).write_bytes(getattr(result, stream))
        process[stream + "_log"] = path
    process.update(
        returncode=result.returncode,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_report(output, record)
    if result.returncode:
        raise RuntimeError(f"{name} failed (rc={result.returncode}); see retained logs")
    return result.stdout.decode("utf-8", errors="replace")


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = None
    record = None
    try:
        if args.max_utts < 0:
            raise ValueError("max-utts must be nonnegative")
        paths = {s: getattr(args, s + "_model_path") for s in STAGES}
        ids = {s: getattr(args, s + "_asset_id") for s in STAGES}
        if any(paths.values()) and not all(paths.values()):
            raise ValueError("Provide all three model paths")
        if any(ids.values()) and not all(ids.values()):
            raise ValueError("Provide all three asset IDs")
        if args.dry_run and args.target == "auto":
            raise ValueError("Host dry-run requires an explicit target")
        target = "s100" if args.list_models and args.target == "auto" else args.target
        selections = resolve_selections(
            target,
            model_paths=paths if any(paths.values()) else None,
            asset_ids=ids if any(ids.values()) else None,
        )
        if args.list_models:
            print(
                json.dumps(
                    [
                        {
                            "stage": s.stage,
                            "target": s.target,
                            "asset_id": s.asset.reference,
                        }
                        for s in selections
                    ],
                    indent=2,
                )
            )
            return 0
        manifest = args.manifest.expanduser().resolve()
        vocabulary_path = args.vocab_file.expanduser().resolve()
        destination = args.output_dir.expanduser().resolve()
        build_dir = CPP / "build/s100"
        binary = (
            args.binary.expanduser().resolve()
            if args.binary
            else build_dir / "paraformer_demo"
        )

        def command(digests):
            result = [
                str(binary),
                "--target",
                "s100",
                "--manifest",
                str(manifest),
                "--vocab-file",
                str(vocabulary_path),
                "--output-dir",
                str(destination / "result"),
                "--max-utts",
                str(args.max_utts),
            ]
            for s, digest in zip(selections, digests, strict=True):
                result += [
                    f"--{s.stage}-model-path",
                    str(s.model_path.resolve()),
                    f"--{s.stage}-asset-id",
                    s.asset.reference,
                    f"--{s.stage}-sha256",
                    digest,
                ]
            return result

        record = {
            "schema": "rdk-model-zoo/paraformer-native-launch/v1",
            "status": "prepared",
            "target": "s100",
            "runtime_metadata_verified": False,
            "native_process_started": False,
            "native_process_attempted": False,
            "downloaded": False,
            "models": [
                {
                    "stage": s.stage,
                    "asset_id": s.asset.reference,
                    "publisher_sha256": s.asset.sha256,
                    "publisher_checksum_verified": False,
                }
                for s in selections
            ],
            "command": command(
                [s.asset.sha256 or "<observed-model-sha256>" for s in selections]
            ),
            "processes": [],
        }
        if args.dry_run:
            print(json.dumps(record, indent=2))
            return 0
        record["detected_target"] = require_execution_target("s100")
        for s in selections:
            if not s.model_path.is_file() or s.model_path.stat().st_size == 0:
                raise ValueError(
                    "Missing or empty model; prepare the model package first"
                )
        digests = [verify_asset_file(s.asset, s.model_path) for s in selections]
        if sha256_file(vocabulary_path) != VOCABULARY_DIGEST:
            raise ValueError("Vocabulary SHA-256 mismatch")
        vocabulary = validate_vocabulary(json.loads(vocabulary_path.read_text()))
        manifest_sha = sha256_file(manifest)
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, list) or not entries:
            raise ValueError("Expected nonempty prepared manifest")
        selected_entries = entries[: args.max_utts] if args.max_utts else entries
        if (
            args.output_dir.expanduser().is_symlink()
            or destination.exists()
            or destination.is_symlink()
        ):
            raise ValueError("Output directory must be new")
        if not args.build and (not binary.is_file() or not os.access(binary, os.X_OK)):
            raise ValueError("Native executable missing; use --build or --binary")
        record.update(
            status="running", manifest_sha256=manifest_sha, command=command(digests)
        )
        for model, digest, s in zip(record["models"], digests, selections, strict=True):
            model.update(
                sha256=digest, publisher_checksum_verified=s.asset.sha256 is not None
            )
        destination.mkdir(parents=True)
        output = destination
        write_report(output, record)
        if args.build:
            run_logged(
                [
                    "cmake",
                    "-S",
                    str(CPP),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DPARAFORMER_BUILD_SDK=ON",
                    "-DPARAFORMER_BUILD_IO=ON",
                    "-DPARAFORMER_BUILD_CLI=ON",
                    "-DPARAFORMER_BUILD_TESTS=OFF",
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
        record["binary_sha256"] = sha256_file(binary)
        help_text = run_logged([str(binary), "--help"], "binary-help", output, record)
        if "Backend: native-sdk" not in help_text or "host-fixture" in help_text:
            raise ValueError("Executable is not the native SDK backend")
        record["native_process_attempted"] = True
        text = run_logged(record["command"], "native", output, record)
        print(text, end="")
        path = output / "result/result.json"
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError("Native success without a regular result.json")
        report = json.loads(path.read_text())
        validate_report(
            report,
            selections,
            digests,
            selected_entries,
            manifest,
            manifest_sha,
            vocabulary,
        )
        if (
            sha256_file(manifest) != manifest_sha
            or sha256_file(vocabulary_path) != VOCABULARY_DIGEST
            or any(
                sha256_file(s.model_path) != digest
                for s, digest in zip(selections, digests, strict=True)
            )
        ):
            raise ValueError("Manifest/model/vocabulary changed during run")
        for entry in selected_entries:
            if (
                sha256_file(manifest.parent / entry["feature_file"])
                != entry["feature_sha256"].lower()
            ):
                raise ValueError("Feature changed during run")
        record.update(
            status="completed",
            runtime_metadata_verified=True,
            result_sha256=sha256_file(path),
        )
        write_report(output, record)
        return 0
    except Exception as error:
        if output is not None and record is not None:
            record.update(status="failed", error=str(error))
            try:
                write_report(output, record)
            except OSError as failure:
                print(f"Could not save launch report: {failure}", file=sys.stderr)
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
