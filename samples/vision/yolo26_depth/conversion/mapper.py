"""Compile one YOLO26 N/S/M/L/X log-depth ONNX for RDK X5."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for a generated artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(command: list[str], cwd: Path, log: Path) -> None:
    """Run one conversion command and capture its combined output.

    Args:
        command: Program and arguments to execute.
        cwd: Working directory for the subprocess.
        log: Destination log file.

    Raises:
        subprocess.CalledProcessError: If the command exits unsuccessfully.
    """
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log.write_text(
        "$ " + " ".join(command) + "\n\n" + result.stdout + "\n--- stderr ---\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def parse_compile_metrics(log: Path) -> dict[str, float | int]:
    """Extract output cosine and performance estimates from Mapper logs.

    Args:
        log: Mapper compilation log.

    Returns:
        Parsed accuracy, latency, frame-rate, and memory estimates.
    """
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = ansi.sub("", log.read_text(encoding="utf-8", errors="replace")).splitlines()
    metrics: dict[str, float | int] = {}
    in_output_table = False
    for line in lines:
        if "The quantized model output:" in line:
            in_output_table = True
            continue
        if in_output_table and line.strip().startswith("output0"):
            metrics["output_cosine_similarity"] = float(line.split()[1])
            in_output_table = False
        match = re.search(
            r"FPS=([0-9.]+), latency = ([0-9.]+) us, DDR = ([0-9]+) bytes", line
        )
        if match:
            metrics["compiler_estimated_fps"] = float(match.group(1))
            metrics["compiler_estimated_latency_us"] = float(match.group(2))
            metrics["compiler_estimated_ddr_bytes"] = int(match.group(3))
    if "output_cosine_similarity" not in metrics:
        raise ValueError(f"output cosine was not found in {log}")
    return metrics


def main() -> None:
    """Run Mapper conversion and write a reproducible compile report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=int, choices=(768,))
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--optimize-level", choices=("O0", "O1", "O2", "O3"), default="O3")
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    calibration_path = args.calibration.resolve()
    output_path = args.output.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not calibration_path.is_dir():
        raise NotADirectoryError(calibration_path)
    if output_path.exists():
        raise FileExistsError(output_path)

    config_dir = output_path / "config"
    checker_dir = output_path / "checker"
    working_dir = output_path / "working"
    artifact_dir = output_path / "artifacts"
    report_dir = output_path / "reports"
    for directory in (config_dir, checker_dir, working_dir, artifact_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    prefix = f"yolo26{args.variant}_depth_bayese_{args.size}x{args.size}_nv12"
    config = {
        "model_parameters": {
            "march": "bayes-e",
            "output_model_file_prefix": prefix,
            "working_dir": str(working_dir),
            "layer_out_dump": False,
            "onnx_model": str(onnx_path),
            "node_info": {
                "/model.23/head/head.3/Conv": {"ON": "BPU", "OutputType": "int16"}
            },
        },
        "input_parameters": {
            "input_name": "images",
            "input_type_train": "rgb",
            "input_layout_train": "NCHW",
            "input_shape": f"1x3x{args.size}x{args.size}",
            "input_type_rt": "nv12",
            "input_batch": 1,
            "norm_type": "data_scale",
            "input_layout_rt": "NHWC",
            "input_space_and_range": "regular",
            "scale_value": "0.003921568627451",
        },
        "calibration_parameters": {
            "calibration_type": "max",
            "cal_data_dir": str(calibration_path),
            "cal_data_type": "uint8",
            "max_percentile": 0.9999,
        },
        "compiler_parameters": {
            "compile_mode": "latency",
            "debug": False,
            "core_num": 1,
            "optimize_level": args.optimize_level,
            "input_source": {"images": "pyramid"},
            "jobs": args.jobs,
        },
    }
    config_path = config_dir / f"{prefix}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    execute(
        [
            "hb_mapper", "checker", "--model-type", "onnx", "--march", "bayes-e",
            "--model", str(onnx_path), "--input-shape", "images",
            f"1x3x{args.size}x{args.size}", "--output", str(checker_dir),
        ],
        config_dir,
        report_dir / "checker.log",
    )
    execute(
        ["hb_mapper", "makertbin", "--config", str(config_path), "--model-type", "onnx"],
        config_dir,
        report_dir / "makertbin.log",
    )

    compiled_model = working_dir / f"{prefix}.bin"
    quantized_model = working_dir / f"{prefix}_quantized_model.onnx"
    if not compiled_model.is_file() or not quantized_model.is_file():
        raise FileNotFoundError("hb_mapper did not produce the expected artifacts")
    deployed_model = artifact_dir / compiled_model.name
    deployed_quantized = artifact_dir / quantized_model.name
    shutil.copy2(compiled_model, deployed_model)
    shutil.copy2(quantized_model, deployed_quantized)
    execute(["hb_model_info", str(deployed_model)], output_path, report_dir / "hb_model_info.log")
    compile_metrics = parse_compile_metrics(report_dir / "makertbin.log")
    report = {
        "schema_version": "1.0",
        "variant": args.variant,
        "size": args.size,
        "onnx": onnx_path.name,
        "calibration": calibration_path.name,
        "config": f"config/{config_path.name}",
        "bin": f"artifacts/{deployed_model.name}",
        "bin_sha256": sha256(deployed_model),
        "quantized_onnx": f"artifacts/{deployed_quantized.name}",
        "quantized_onnx_sha256": sha256(deployed_quantized),
        "optimization": args.optimize_level,
        "calibration_type": "max",
        "max_percentile": 0.9999,
        "int16_output_node": "/model.23/head/head.3/Conv",
        **compile_metrics,
    }
    (report_dir / "compile-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
