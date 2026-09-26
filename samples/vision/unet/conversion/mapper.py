#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compile one UNet ResNet ONNX model into a verified RDK X5 binary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
INPUT_NAME = "images"
INPUT_SHAPE = "1x3x512x512"
INPUT_ELEMENTS = 3 * 512 * 512
SAMPLE_BYTES = INPUT_ELEMENTS * np.dtype("<f4").itemsize
OUTPUT_SHAPE = [1, 21, 512, 512]
SCALE_VALUE = 1.0 / 255.0
TEMPLATE_DIR = Path(__file__).resolve().with_name("ptq_yamls")


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a UTF-8 JSON report to a new or run-owned path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def execute(command: list[str], cwd: Path, log_path: Path) -> None:
    """Run one command while streaming combined output to a log and terminal."""

    header = "$ " + " ".join(command) + "\n\n"
    print(header, end="", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        with subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        ) as process:
            if process.stdout is None:
                raise RuntimeError("failed to capture command output")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def tool_version(command: list[str]) -> str:
    """Return the first non-empty version line reported by a tool."""

    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return output.splitlines()[0] if output else "unreported"


def load_template(backbone: str) -> tuple[Path, dict[str, Any]]:
    """Load the checked-in PTQ template for one supported backbone."""

    template_path = TEMPLATE_DIR / f"unet_{backbone}_voc_512x512_nv12.yaml"
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"PTQ template root must be a mapping: {template_path}")
    validate_template(backbone, payload)
    return template_path, payload


def validate_template(backbone: str, config: dict[str, Any]) -> None:
    """Validate the immutable X5 and UNet contracts in a PTQ template."""

    expected_groups = {
        "model_parameters",
        "input_parameters",
        "calibration_parameters",
        "compiler_parameters",
    }
    if set(config) != expected_groups:
        raise ValueError(f"PTQ template must contain exactly {sorted(expected_groups)}")
    model = config["model_parameters"]
    inputs = config["input_parameters"]
    calibration = config["calibration_parameters"]
    compiler = config["compiler_parameters"]
    expected_prefix = f"unet_{backbone}_voc_512x512_nv12"
    checks = {
        "march": model.get("march") == "bayes-e",
        "onnx_model": Path(str(model.get("onnx_model", ""))).name
        == f"unet_{backbone}_voc_512x512.onnx",
        "output_model_file_prefix": model.get("output_model_file_prefix")
        == expected_prefix,
        "input_name": inputs.get("input_name") == INPUT_NAME,
        "input_shape": inputs.get("input_shape") == INPUT_SHAPE,
        "input_type_train": inputs.get("input_type_train") == "rgb",
        "input_layout_train": inputs.get("input_layout_train") == "NCHW",
        "input_type_rt": inputs.get("input_type_rt") == "nv12",
        "norm_type": inputs.get("norm_type") == "data_scale",
        "scale_value": abs(float(inputs.get("scale_value", 0.0)) - SCALE_VALUE) < 1e-15,
        "calibration_type": calibration.get("calibration_type")
        in {"default", "mix", "kl", "max"},
        "cal_data_type": calibration.get("cal_data_type") == "float32",
        "input_source": compiler.get("input_source") == {INPUT_NAME: "pyramid"},
        "core_num": compiler.get("core_num") == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("invalid UNet X5 PTQ template fields: " + ", ".join(failed))
    serialized = yaml.safe_dump(config, sort_keys=False).lower()
    forbidden = ("nash-", "bernoulli2", "hbdk4", "hmct", "compile_perf.py")
    found = [term for term in forbidden if term in serialized]
    if found:
        raise ValueError("non-X5 PTQ terms found: " + ", ".join(found))


def validate_export_report(
    backbone: str,
    onnx_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Require a numerically verified export receipt for the selected ONNX."""

    if not report_path.is_file():
        raise FileNotFoundError(f"missing ONNX export report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("backbone") != backbone:
        raise ValueError("ONNX export report backbone does not match --backbone")
    if report.get("onnx_sha256") != sha256_file(onnx_path):
        raise ValueError("ONNX hash does not match its export report")
    if report.get("x5_ptq_ready") is not True:
        raise ValueError("ONNX export did not pass its numerical runtime check")
    contract = report.get("contract", {})
    if contract.get("input_name") != INPUT_NAME:
        raise ValueError("ONNX export report has an unexpected input name")
    if contract.get("input_shape") != [1, 3, 512, 512]:
        raise ValueError("ONNX export report has an unexpected input shape")
    if contract.get("output_name") != "logits":
        raise ValueError("ONNX export report has an unexpected output name")
    if contract.get("output_shape") != OUTPUT_SHAPE:
        raise ValueError("ONNX export report has an unexpected output shape")
    if contract.get("opsets", {}).get("ai.onnx") != 11:
        raise ValueError("ONNX export report must record opset 11")
    return report


def audit_calibration(calibration_dir: Path) -> dict[str, object]:
    """Validate RGB float32 calibration tensors and return a manifest."""

    if not calibration_dir.is_dir():
        raise NotADirectoryError(calibration_dir)
    files = sorted(calibration_dir.glob("*.bin"))
    if not files:
        raise ValueError(f"no .bin calibration samples found in {calibration_dir}")
    samples: list[dict[str, object]] = []
    for path in files:
        payload = path.read_bytes()
        if len(payload) != SAMPLE_BYTES:
            raise ValueError(
                f"{path} has {len(payload)} bytes; expected {SAMPLE_BYTES} "
                "for RGB float32 CHW [3,512,512]"
            )
        array = np.frombuffer(payload, dtype="<f4")
        if not bool(np.isfinite(array).all()):
            raise ValueError(f"calibration sample contains NaN or Inf: {path}")
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0.0 or maximum > 255.0:
            raise ValueError(
                f"calibration sample must contain unnormalized RGB [0,255]: {path}"
            )
        samples.append(
            {
                "file": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": [3, 512, 512],
                "dtype": "float32-le",
                "minimum": minimum,
                "maximum": maximum,
                "mean": float(array.mean(dtype=np.float64)),
            }
        )
    warnings = []
    if len(samples) < 100:
        warnings.append(
            f"only {len(samples)} calibration samples were provided; "
            "about 100 representative samples are recommended"
        )
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(calibration_dir),
        "selection": {"sample_count": len(samples)},
        "input_contract": {
            "name": INPUT_NAME,
            "shape": [1, 3, 512, 512],
            "stored_shape": [3, 512, 512],
            "dtype": "float32-le",
            "layout": "NCHW",
            "color_space": "RGB",
            "stored_range": [0.0, 255.0],
            "normalization_owner": "PTQ YAML data_scale",
            "scale_value": SCALE_VALUE,
        },
        "samples": samples,
        "warnings": warnings,
    }


def resolve_config(
    template: dict[str, Any],
    onnx_path: Path,
    calibration_dir: Path,
    working_dir: Path,
    calibration_type: str,
    max_percentile: float,
    optimize_level: str,
    jobs: int,
) -> dict[str, Any]:
    """Bind a checked-in template to the artifacts of one isolated run."""

    config = copy.deepcopy(template)
    model = config["model_parameters"]
    calibration = config["calibration_parameters"]
    compiler = config["compiler_parameters"]
    model["onnx_model"] = str(onnx_path)
    model["working_dir"] = str(working_dir)
    calibration["calibration_type"] = calibration_type
    calibration["cal_data_dir"] = str(calibration_dir)
    if calibration_type == "mix":
        calibration["max_percentile"] = max_percentile
    else:
        calibration.pop("max_percentile", None)
    compiler["optimize_level"] = optimize_level
    compiler["jobs"] = jobs
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--export-report",
        type=Path,
        help="numerically passed export report; defaults beside the ONNX",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
        help="directory of RGB float32 CHW .bin samples in range [0,255]",
    )
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--optimize-level",
        choices=("O0", "O1", "O2", "O3"),
        default="O3",
    )
    parser.add_argument(
        "--calibration-type",
        choices=("default", "mix", "kl", "max"),
        default="default",
    )
    parser.add_argument("--max-percentile", type=float, default=1.0)
    return parser.parse_args()


def compile_model(args: argparse.Namespace) -> dict[str, object]:
    """Execute checker, makertbin, artifact gates, and model-info verification."""

    onnx_path = args.onnx.expanduser().resolve()
    calibration_dir = args.calibration.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_path}")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if not 0.0 < args.max_percentile <= 1.0:
        raise ValueError("--max-percentile must be in (0,1]")

    template_path, template = load_template(args.backbone)
    report_path = (
        args.export_report.expanduser().resolve()
        if args.export_report
        else onnx_path.with_suffix(".export.json")
    )
    export_report = validate_export_report(args.backbone, onnx_path, report_path)
    calibration_manifest = audit_calibration(calibration_dir)
    for executable in ("hb_mapper", "hb_model_info"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"{executable} is not available in PATH")

    config_dir = output_path / "config"
    checker_dir = output_path / "checker"
    working_dir = output_path / "working"
    artifact_dir = output_path / "artifacts"
    reports_dir = output_path / "reports"
    for directory in (
        config_dir,
        checker_dir,
        working_dir,
        artifact_dir,
        reports_dir,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    config = resolve_config(
        template=template,
        onnx_path=onnx_path,
        calibration_dir=calibration_dir,
        working_dir=working_dir,
        calibration_type=args.calibration_type,
        max_percentile=args.max_percentile,
        optimize_level=args.optimize_level,
        jobs=args.jobs,
    )
    validate_template(args.backbone, config)
    config_path = config_dir / template_path.name
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    calibration_manifest_path = reports_dir / "calibration-manifest.json"
    write_json(calibration_manifest_path, calibration_manifest)
    receipt_path = reports_dir / "run-receipt.json"
    completed_stage = "config"
    try:
        execute(
            [
                "hb_mapper",
                "checker",
                "--model-type",
                "onnx",
                "--march",
                "bayes-e",
                "--model",
                str(onnx_path),
                "--input-shape",
                INPUT_NAME,
                INPUT_SHAPE,
                "--output",
                str(checker_dir),
            ],
            config_dir,
            reports_dir / "checker.log",
        )
        completed_stage = "checker"
        execute(
            [
                "hb_mapper",
                "makertbin",
                "--config",
                str(config_path),
                "--model-type",
                "onnx",
            ],
            config_dir,
            reports_dir / "makertbin.log",
        )
        completed_stage = "makertbin"

        compiled_models = sorted(working_dir.rglob("*.bin"))
        if len(compiled_models) != 1:
            raise ValueError(
                f"expected exactly one compiled .bin, found {len(compiled_models)}"
            )
        deployed_model = artifact_dir / compiled_models[0].name
        shutil.copy2(compiled_models[0], deployed_model)

        quantized_models = sorted(working_dir.rglob("*quantized_model.onnx"))
        if len(quantized_models) > 1:
            raise ValueError(
                "expected at most one quantized ONNX, " f"found {len(quantized_models)}"
            )
        deployed_quantized = None
        if quantized_models:
            deployed_quantized = artifact_dir / quantized_models[0].name
            shutil.copy2(quantized_models[0], deployed_quantized)

        model_info_log = reports_dir / "hb_model_info.log"
        execute(["hb_model_info", str(deployed_model)], output_path, model_info_log)
        model_info = model_info_log.read_text(encoding="utf-8", errors="replace")
        if re.search(r"BPU\s+march\s*:\s*bayes-e\b", model_info, re.I) is None:
            raise ValueError("hb_model_info did not report 'BPU march: bayes-e'")
        completed_stage = "hb_model_info"

        receipt: dict[str, object] = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "platform": "RDK X5",
            "march": "bayes-e",
            "backbone": args.backbone,
            "completed_stage": completed_stage,
            "onnx": str(onnx_path),
            "onnx_sha256": sha256_file(onnx_path),
            "export_report": str(report_path),
            "export_report_sha256": sha256_file(report_path),
            "export_numerical_check": export_report["numerical_check"],
            "calibration_manifest": str(
                calibration_manifest_path.relative_to(output_path)
            ),
            "calibration_samples": calibration_manifest["selection"]["sample_count"],
            "calibration_warnings": calibration_manifest["warnings"],
            "config": str(config_path.relative_to(output_path)),
            "config_sha256": sha256_file(config_path),
            "model": str(deployed_model.relative_to(output_path)),
            "model_sha256": sha256_file(deployed_model),
            "quantized_onnx": (
                str(deployed_quantized.relative_to(output_path))
                if deployed_quantized
                else None
            ),
            "quantized_onnx_sha256": (
                sha256_file(deployed_quantized) if deployed_quantized else None
            ),
            "tool_versions": {
                "hb_mapper": tool_version(["hb_mapper", "--version"]),
                "hb_model_info": tool_version(["hb_model_info", "--version"]),
            },
        }
        write_json(receipt_path, receipt)
        return receipt
    except Exception as exc:
        failure = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "platform": "RDK X5",
            "march": "bayes-e",
            "backbone": args.backbone,
            "completed_stage": completed_stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(receipt_path, failure)
        raise


def main() -> int:
    """Compile one model and report concise expected failures."""

    args = parse_args()
    try:
        receipt = compile_model(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
