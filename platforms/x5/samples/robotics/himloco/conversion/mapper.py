#!/usr/bin/env python3
"""Compile the fused HIMLoco ONNX policy for RDK X5 with OpenExplorer Mapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

INPUT_NAME = "obs_history"
INPUT_SHAPE = "1x270"
SAMPLE_BYTES = 270 * 4
MODEL_PREFIX = "himloco_go2_bayese_1x270"


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one artifact.

    Args:
        path: Artifact to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(command: list[str], cwd: Path, log: Path) -> None:
    """Run one command and stream combined output to the terminal and log.

    Args:
        command: Executable and arguments.
        cwd: Command working directory.
        log: New file receiving combined standard output and error.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: If the command exits unsuccessfully.
    """

    command_header = "$ " + " ".join(command) + "\n\n"
    print(command_header, end="", flush=True)
    with log.open("w", encoding="utf-8") as log_handle:
        log_handle.write(command_header)
        log_handle.flush()
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
                log_handle.write(line)
                log_handle.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def tool_version(command: list[str]) -> str:
    """Return compact version output without creating a build prerequisite.

    Args:
        command: Version command and arguments.

    Returns:
        First reported version line, or ``unreported`` when empty.
    """

    result = subprocess.run(command, text=True, capture_output=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    return output.splitlines()[0] if output else "unreported"


def parse_compile_metrics(log: Path) -> dict[str, float | int | str]:
    """Extract output cosine and compiler estimates from a Mapper log.

    Args:
        log: Mapper build log.

    Returns:
        Output cosine and any available latency, FPS, and DDR estimates.

    Raises:
        ValueError: If output cosine cannot be located.
    """

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    lines = ansi.sub("", log.read_text(encoding="utf-8", errors="replace")).splitlines()
    metrics: dict[str, float | int | str] = {}
    output_table_lines = 0
    for line in lines:
        if "The quantized model output:" in line:
            output_table_lines = 30
            continue
        if output_table_lines:
            output_table_lines -= 1
            fields = line.split()
            numeric_fields = []
            for field in fields[1:]:
                try:
                    numeric_fields.append(float(field))
                except ValueError:
                    continue
            similarities = [value for value in numeric_fields if 0.0 <= value <= 1.0]
            if fields and similarities:
                metrics["output_cosine_similarity"] = similarities[0]
                metrics["output_node"] = fields[0]
                output_table_lines = 0
                continue
        match = re.search(
            r"FPS=([0-9.]+), latency = ([0-9.]+) us, DDR = ([0-9]+) bytes", line
        )
        if match:
            metrics["compiler_estimated_fps"] = float(match.group(1))
            metrics["compiler_estimated_latency_us"] = float(match.group(2))
            metrics["compiler_estimated_ddr_bytes"] = int(match.group(3))
    if "output_cosine_similarity" not in metrics:
        raise ValueError(f"output cosine similarity was not found in {log}")
    return metrics


def validate_calibration(calibration_dir: Path) -> tuple[list[Path], Path, dict]:
    """Validate raw calibration files and their manifest.

    Args:
        calibration_dir: Directory containing raw float32 ``.bin`` files.

    Returns:
        Validated file paths, manifest path, and parsed manifest.

    Raises:
        FileNotFoundError: If the manifest is missing.
        NotADirectoryError: If ``calibration_dir`` is not a directory.
        ValueError: If file sizes or manifest fields do not match the contract.
    """

    if not calibration_dir.is_dir():
        raise NotADirectoryError(calibration_dir)
    files = sorted(calibration_dir.glob("*.bin"))
    if not files:
        raise ValueError(f"no .bin calibration files found in {calibration_dir}")
    invalid_sizes = [path for path in files if path.stat().st_size != SAMPLE_BYTES]
    if invalid_sizes:
        raise ValueError(
            f"calibration files must contain {SAMPLE_BYTES} bytes: {invalid_sizes[0]}"
        )

    manifest_path = calibration_dir.parent / "calibration-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing calibration manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("input_contract", {})
    if contract.get("name") != INPUT_NAME or contract.get("model_shape") != [1, 270]:
        raise ValueError("calibration manifest does not match obs_history [1,270]")
    if manifest.get("selection", {}).get("sample_count") != len(files):
        raise ValueError("calibration manifest/file count mismatch")
    return files, manifest_path, manifest


def build_config(
    onnx_path: Path,
    calibration_dir: Path,
    working_dir: Path,
    calibration_type: str,
    max_percentile: float,
    optimize_level: str,
    jobs: int,
) -> dict:
    """Build the complete Mapper configuration.

    Args:
        onnx_path: Fused fixed-shape ONNX policy.
        calibration_dir: Validated calibration tensor directory.
        working_dir: Mapper working directory.
        calibration_type: Mapper calibration algorithm.
        max_percentile: MIX calibration percentile bound.
        optimize_level: Compiler optimization level.
        jobs: Compiler worker count.

    Returns:
        Complete RDK X5 Mapper configuration mapping.
    """

    return {
        "model_parameters": {
            "onnx_model": str(onnx_path),
            "march": "bayes-e",
            "layer_out_dump": False,
            "working_dir": str(working_dir),
            "output_model_file_prefix": MODEL_PREFIX,
        },
        "input_parameters": {
            "input_name": INPUT_NAME,
            "input_shape": INPUT_SHAPE,
            "input_type_train": "featuremap",
            "input_layout_train": "NHWC",
            "input_type_rt": "featuremap",
            "input_layout_rt": "NHWC",
            "norm_type": "no_preprocess",
        },
        "calibration_parameters": {
            "calibration_type": calibration_type,
            "cal_data_dir": str(calibration_dir),
            "cal_data_type": "float32",
            "max_percentile": max_percentile,
            "per_channel": False,
        },
        "compiler_parameters": {
            "jobs": jobs,
            "compile_mode": "latency",
            "debug": False,
            "optimize_level": optimize_level,
            "core_num": 1,
            "input_source": {INPUT_NAME: "ddr"},
        },
    }


def main() -> None:
    """Create an isolated Mapper run and write a reproducible compile receipt.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(
        description="Compile HIMLoco ONNX for RDK X5 Bayes-e."
    )
    parser.add_argument(
        "--onnx", required=True, type=Path, help="Fused fixed-shape ONNX model."
    )
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Directory containing raw obs_history/*.bin calibration files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New run directory; existing paths are rejected.",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Mapper compile jobs.")
    parser.add_argument(
        "--optimize-level", choices=("O0", "O1", "O2", "O3"), default="O3"
    )
    parser.add_argument(
        "--calibration-type", choices=("default", "kl", "max", "mix"), default="mix"
    )
    parser.add_argument("--max-percentile", type=float, default=1.0)
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    calibration_dir = args.calibration.resolve()
    output_path = args.output.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if not 0.0 < args.max_percentile <= 1.0:
        raise ValueError("--max-percentile must be in (0, 1]")
    for executable in ("hb_mapper", "hb_model_info"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"{executable} is not available in PATH")

    calibration_files, manifest_path, manifest = validate_calibration(calibration_dir)
    config_dir = output_path / "config"
    checker_dir = output_path / "checker"
    working_dir = output_path / "working"
    artifact_dir = output_path / "artifacts"
    report_dir = output_path / "reports"
    for directory in (config_dir, checker_dir, working_dir, artifact_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    config = build_config(
        onnx_path=onnx_path,
        calibration_dir=calibration_dir,
        working_dir=working_dir,
        calibration_type=args.calibration_type,
        max_percentile=args.max_percentile,
        optimize_level=args.optimize_level,
        jobs=args.jobs,
    )
    config_path = config_dir / f"{MODEL_PREFIX}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    checker_log = report_dir / "checker.log"
    makertbin_log = report_dir / "makertbin.log"
    model_info_log = report_dir / "hb_model_info.log"
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
        checker_log,
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
        config_dir,
        makertbin_log,
    )

    compiled_models = sorted(working_dir.rglob("*.bin"))
    if len(compiled_models) != 1:
        raise ValueError(f"expected one compiled BIN, found {len(compiled_models)}")
    deployed_model = artifact_dir / compiled_models[0].name
    shutil.copy2(compiled_models[0], deployed_model)

    quantized_models = sorted(working_dir.rglob("*quantized_model.onnx"))
    if len(quantized_models) > 1:
        raise ValueError(
            f"expected at most one quantized ONNX, found {len(quantized_models)}"
        )
    deployed_quantized = None
    if quantized_models:
        deployed_quantized = artifact_dir / quantized_models[0].name
        shutil.copy2(quantized_models[0], deployed_quantized)

    execute(["hb_model_info", str(deployed_model)], output_path, model_info_log)
    model_info_text = model_info_log.read_text(encoding="utf-8", errors="replace")
    if "bayes-e" not in model_info_text.lower():
        raise ValueError("hb_model_info did not confirm bayes-e")

    compile_metrics = parse_compile_metrics(makertbin_log)
    report = {
        "schema_version": "1.0",
        "platform": "RDK X5",
        "march": "bayes-e",
        "input_contract": {
            "name": INPUT_NAME,
            "shape": [1, 270],
            "dtype": "float32",
            "runtime_type": "featuremap",
            "source": "ddr",
        },
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": sha256(manifest_path),
        "calibration_samples": len(calibration_files),
        "calibration_source_sha256": manifest.get("source_sha256"),
        "config": str(config_path.relative_to(output_path)),
        "model": str(deployed_model.relative_to(output_path)),
        "model_sha256": sha256(deployed_model),
        "quantized_onnx": (
            str(deployed_quantized.relative_to(output_path))
            if deployed_quantized
            else None
        ),
        "quantized_onnx_sha256": (
            sha256(deployed_quantized) if deployed_quantized else None
        ),
        "calibration_type": args.calibration_type,
        "max_percentile": args.max_percentile,
        "optimize_level": args.optimize_level,
        "jobs": args.jobs,
        "tool_versions": {
            "hb_mapper": tool_version(["hb_mapper", "--version"]),
            "hb_model_info": tool_version(["hb_model_info", "--version"]),
        },
        **compile_metrics,
    }
    report_path = report_dir / "compile-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
