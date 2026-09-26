# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Bind source recipe to explicit ONNX and verified calibration before OE."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from samples._shared.assets import sha256_file
from samples.vision.yolo26_depth.conversion.prepare_calibration import (
    TARGETS,
    VARIANTS,
    select_profile,
)
from samples.vision.yolo26_depth.conversion.mapper import execute, parse_compile_metrics

MARCHES = {"x5": "bayes-e", "s100": "nash-e", "s100p": "nash-m", "s600": "nash-p"}
HERE = Path(__file__).resolve().parent


def template_path(target, variant, profile):
    if (
        target not in TARGETS
        or variant not in VARIANTS
        or profile not in ("nv12", "lite")
    ):
        raise ValueError("Unknown recipe identity")
    if target == "x5":
        if profile != "nv12":
            raise ValueError("No X5 lite recipe")
        return HERE / "ptq_yamls/x5" / f"yolo26{variant}_depth_768.yaml"
    suffix = MARCHES[target].replace("-", "")
    filename = (
        f"yolo26{variant}_depth_lite_{suffix}_768.yaml"
        if profile == "lite"
        else f"yolo26{variant}_depth_{suffix}_768x768_nv12.yaml"
    )
    path = HERE / "ptq_yamls/s" / filename
    if not path.is_file():
        raise ValueError(f"No source recipe for {target}/{variant}/{profile}")
    return path


def configure(target, variant, profile, onnx, calibration, working):
    config = copy.deepcopy(
        yaml.safe_load(template_path(target, variant, profile).read_text())
    )
    config["model_parameters"]["onnx_model"] = str(onnx.resolve())
    config["model_parameters"]["working_dir"] = str(working.resolve())
    config["calibration_parameters"]["cal_data_dir"] = str(calibration.resolve())
    return config


def validate_calibration(path, target, profile):
    data = json.loads(path.read_text())
    # S marches share a calibration representation; X5 uses a different one.
    same_family = (data.get("target") == "x5") == (target == "x5")
    if (
        data.get("target") not in TARGETS
        or not same_family
        or data.get("profile") != profile
        or data.get("size") != 768
    ):
        raise ValueError(
            "Calibration target/profile/size does not match the requested recipe"
        )
    directory = Path(data["tensor_directory"]).resolve()
    records = data.get("records", [])
    if not records or len(records) != data.get("selection", {}).get("count"):
        raise ValueError("Calibration requires a nonempty complete record set")
    expected_names = []
    expected_shape = [3, 768, 768] if target == "x5" else [1, 3, 768, 768]
    expected_dtype = "uint8" if target == "x5" else "float32"
    for record in records:
        name = record["output"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in expected_names
        ):
            raise ValueError(
                "Calibration record must name one unique local tensor file"
            )
        expected_names.append(name)
        tensor_path = directory / name
        if sha256_file(tensor_path) != record["output_sha256"]:
            raise ValueError(f"Calibration digest mismatch: {name}")
        if (
            record.get("shape") != expected_shape
            or record.get("dtype") != expected_dtype
        ):
            raise ValueError(f"Calibration tensor contract mismatch: {name}")
        if target == "x5":
            if (
                tensor_path.suffix != ".bin"
                or tensor_path.stat().st_size != 3 * 768 * 768
            ):
                raise ValueError(f"Expected RGB CHW uint8 binary: {name}")
        else:
            tensor = np.load(tensor_path, allow_pickle=False)
            if (
                tensor.shape != tuple(expected_shape)
                or tensor.dtype != np.float32
                or not np.isfinite(tensor).all()
                or tensor.min() < 0
                or tensor.max() > 1
            ):
                raise ValueError(
                    f"Expected normalized float32 NCHW calibration: {name}"
                )
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != set(
        expected_names
    ):
        raise ValueError(
            "Calibration directory must contain exactly the manifest tensor files"
        )
    return directory, data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--experimental-lite", action="store_true")
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--calibration-manifest", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, required=True, help="New external work directory"
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate files and write config; does not validate ONNX graph or run OE",
    )
    args = p.parse_args(argv)
    profile = select_profile(args.target, args.variant, args.experimental_lite)
    onnx, output = args.onnx.expanduser().resolve(), args.output.expanduser().resolve()
    if not onnx.is_file():
        raise FileNotFoundError(onnx)
    if output.exists():
        raise FileExistsError(output)
    manifest = args.calibration_manifest.expanduser().resolve()
    calibration, data = validate_calibration(manifest, args.target, profile)
    config = configure(
        args.target, args.variant, profile, onnx, calibration, output / "working"
    )
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    report = {
        "schema_version": "1.0",
        "target": args.target,
        "variant": args.variant,
        "profile": profile,
        "experimental": args.target != "x5"
        and profile == "lite"
        and args.variant in ("n", "s", "m"),
        "onnx": str(onnx),
        "onnx_sha256": sha256_file(onnx),
        "onnx_graph_validation": "not-run",
        "calibration_manifest": str(manifest),
        "calibration_manifest_sha256": sha256_file(manifest),
        "calibration_count": len(data["records"]),
        "config": str(config_path),
        "compilation": "not-run",
        "board": "not-run",
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return 0
    logs = output / "reports"
    logs.mkdir()
    if args.target == "x5":
        checker = output / "checker"
        checker.mkdir()
        execute(
            [
                "hb_mapper",
                "checker",
                "--model-type",
                "onnx",
                "--march",
                "bayes-e",
                "--model",
                str(onnx),
                "--input-shape",
                "images",
                "1x3x768x768",
                "--output",
                str(checker),
            ],
            output,
            logs / "checker.log",
        )
        execute(
            [
                "hb_mapper",
                "makertbin",
                "--config",
                str(config_path),
                "--model-type",
                "onnx",
            ],
            output,
            logs / "makertbin.log",
        )
        extension = ".bin"
    else:
        execute(
            ["hb_compile", "--config", str(config_path)],
            output,
            logs / "hb_compile.log",
        )
        extension = ".hbm"
    prefix = config["model_parameters"]["output_model_file_prefix"]
    compiled = output / "working" / (prefix + extension)
    if not compiled.is_file():
        raise FileNotFoundError(
            f"Compiler did not produce expected artifact: {compiled}"
        )
    artifacts = output / "artifacts"
    artifacts.mkdir()
    deployed = artifacts / compiled.name
    shutil.copy2(compiled, deployed)
    report.update(
        compilation="completed",
        artifact=str(deployed),
        artifact_sha256=sha256_file(deployed),
    )
    if args.target == "x5":
        quantized = output / "working" / (prefix + "_quantized_model.onnx")
        if not quantized.is_file():
            raise FileNotFoundError(quantized)
        shutil.copy2(quantized, artifacts / quantized.name)
        report["quantized_onnx_sha256"] = sha256_file(quantized)
        execute(["hb_model_info", str(deployed)], output, logs / "hb_model_info.log")
        report.update(parse_compile_metrics(logs / "makertbin.log"))
        report["onnx_graph_validation"] = (
            "hb_mapper checker completed; not independent numerical validation"
        )
    else:
        report["onnx_graph_validation"] = (
            "hb_compile completed; not independent numerical validation"
        )
    (output / "compile-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
