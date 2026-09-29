# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Validate explicit calibration and parameterize the preserved S100 template."""

import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from samples._shared.assets import sha256_file
from samples.vision.lanenet.conversion.prepare_calibration import PROTOCOL


def validate_calibration(manifest):
    report = json.loads(manifest.read_text())
    records = report.get("records", [])
    if (
        report.get("protocol") != PROTOCOL
        or report.get("target") != "s100"
        or not records
        or report.get("count") != len(records)
    ):
        raise ValueError("Calibration protocol/target/count mismatch")
    data = (manifest.parent / "data").resolve()
    paths = []
    for record in records:
        path = (manifest.parent / record["tensor"]).resolve()
        if path.parent != data or path.suffix != ".npy" or not path.is_file():
            raise ValueError("Calibration tensor must be a local data/*.npy file")
        if record.get("shape") != [1, 3, 256, 512] or record.get("dtype") != "float32":
            raise ValueError("Declared calibration geometry/type mismatch")
        if sha256_file(path) != record.get("tensor_sha256"):
            raise ValueError("Calibration tensor digest mismatch")
        value = np.load(path, allow_pickle=False)
        if (
            value.shape != (1, 3, 256, 512)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError("Actual calibration tensor geometry/type/values mismatch")
        # Every source pixel is uint8, normalized with the three ImageNet constants.
        low = -np.array([0.485, 0.456, 0.406], np.float32) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        high = (1 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        if (value < low[None, :, None, None] - 1e-6).any() or (
            value > high[None, :, None, None] + 1e-6
        ).any():
            raise ValueError("Calibration outside expected normalized uint8 range")
        paths.append(path)
    if len(set(paths)) != len(paths) or set(data.iterdir()) != set(paths):
        raise ValueError(
            "Calibration data directory has duplicate or unmanifested files"
        )
    return data, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--calibration-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.onnx.is_file() or args.onnx.stat().st_size == 0:
        raise ValueError("A nonempty caller-supplied ONNX is required")
    data, manifest = validate_calibration(args.calibration_manifest)
    template = Path(__file__).with_name("config.yaml")
    config = yaml.safe_load(template.read_text())
    config["model_parameters"].update(
        onnx_model=str(args.onnx.resolve()),
        working_dir=str((args.output / "artifacts").resolve()),
        output_model_file_prefix="lanenet256x512",
    )
    config["calibration_parameters"]["cal_data_dir"] = str(data)
    args.output.mkdir(parents=True)
    cfg = args.output / "config.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    argv = ["hb_compile", "-c", str(cfg.resolve())]
    report = {
        "schema_version": "1.0",
        "target": "s100",
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": sha256_file(args.onnx),
        "onnx_semantics": "caller-declared; no graph validation performed",
        "calibration_manifest": str(args.calibration_manifest.resolve()),
        "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
        "sample_count": manifest["count"],
        "template_sha256": sha256_file(template),
        "config_sha256": sha256_file(cfg),
        "argv": argv,
        "artifact_origin": "caller-converted; not published asset authentication",
        "board_validation": "not-run",
        "compilation": "not-run" if args.prepare_only else "pending",
    }
    if not args.prepare_only:
        result = subprocess.run(
            argv, cwd=args.output.resolve(), text=True, capture_output=True
        )
        (args.output / "stdout.log").write_text(result.stdout)
        (args.output / "stderr.log").write_text(result.stderr)
        report["returncode"] = result.returncode
        if result.returncode != 0:
            report["compilation"] = "failed"
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            raise RuntimeError(f"hb_compile failed with {result.returncode}")
        artifact = args.output / "artifacts/lanenet256x512.hbm"
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise ValueError(
                "Compiler returned zero without expected nonempty artifact"
            )
        report.update(
            compilation="completed; execution not validated",
            artifact=str(artifact.resolve()),
            artifact_sha256=sha256_file(artifact),
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared conversion record at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
