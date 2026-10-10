# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Prepare one reviewable YOLOE conversion; compilation is explicit and separately recorded."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from utils.py_utils.assets import sha256_file
from samples.vision.yoloe.runtime.python.cli import resolve_selection
from samples.vision.yoloe.conversion.calibration import select_images, write_calibration
from samples.vision.yoloe.conversion.contract import inspect_graph
from samples.vision.yoloe.conversion.configuration import make_config


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def compile_prepared(root, selection, config, executable):
    """Record a real command; an exit-zero artifact is still precision-unverified."""
    argv = (
        [
            executable,
            "makertbin",
            "--model-type",
            "onnx",
            "--config",
            str(root / "config.yaml"),
        ]
        if selection.target == "x5"
        else [executable, "-c", str(root / "config.yaml")]
    )
    record = {
        "argv": argv,
        "cwd": str(root),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "log": "compile.log",
    }
    try:
        with (root / "compile.log").open("w") as log:
            proc = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        record["returncode"] = proc.returncode
    except OSError as error:
        record.update(returncode=None, error=str(error))
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    extension = ".bin" if selection.target == "x5" else ".hbm"
    path = Path(config["model_parameters"]["working_dir"]) / (
        config["model_parameters"]["output_model_file_prefix"] + extension
    )
    if record.get("returncode") != 0:
        return "compile_failed", record, None
    if not path.is_file() or not path.stat().st_size:
        record["error"] = f"Compiler did not produce a nonempty artifact: {path}"
        return "compile_failed", record, None
    return (
        "compiled_unverified",
        record,
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size},
    )


def prepare_conversion(
    *,
    onnx_path,
    names_path,
    target,
    variant,
    images,
    output_dir,
    sample_count=100,
    compile_model=False,
    compiler=None,
):
    """Snapshot validated inputs, prepare calibration/YAML and optionally compile.

    Output directories are new and failures are preserved for diagnosis. No
    downloads or board access. Variant is declared, not proven from graph shapes.
    """
    import yaml

    selection = resolve_selection(target, variant=variant)
    root = Path(output_dir).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Refusing existing conversion directory: {root}")
    source = Path(onnx_path).expanduser().resolve()
    names = Path(names_path).expanduser().resolve()
    graph = inspect_graph(source, names, selection.variant)
    paths = select_images(images, sample_count)
    executable = None
    if compile_model:
        executable = shutil.which(
            compiler or ("hb_mapper" if selection.target == "x5" else "hb_compile")
        )
        if executable is None:
            raise ValueError(
                "OE compiler unavailable; use its matching toolchain environment."
            )
        executable = str(Path(executable).resolve())
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "target": selection.target,
        "variant_declared": selection.variant,
        "source_asset_id": selection.asset.reference,
        "status": "preparing",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "graph": graph,
        "expected_output_dtype": "float32",
        "observed_output_dtype": None,
        "board": "not-run",
        "dataset_accuracy": "not-run",
        "compiler": None,
        "artifact": None,
        "warnings": [],
    }
    try:
        (root / "source").mkdir()
        shutil.copyfile(source, root / "source/model.onnx")
        shutil.copyfile(names, root / "source/classes.names")
        if (
            sha256_file(root / "source/model.onnx") != graph["onnx_sha256"]
            or sha256_file(root / "source/classes.names") != graph["names_sha256"]
        ):
            raise ValueError("Source changed while snapshotting conversion inputs.")
        records = write_calibration(
            paths, root / "calibration", selection.target, selection.variant
        )
        write_json(root / "calibration.json", records)
        config, warnings = make_config(selection, root, graph)
        report["warnings"] = warnings + (
            ["fewer than 20 calibration images; no accuracy acceptance"]
            if len(records) < 20
            else []
        )
        report["calibration"] = {
            "requested": sample_count,
            "actual": len(records),
            "format": (
                "raw-float32-rgb-0..255"
                if selection.target == "x5"
                else "npy-float32-rgb-0..1"
            ),
        }
        (root / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        report["config_sha256"] = sha256_file(root / "config.yaml")
        report["status"] = "config_only"
        if compile_model:
            report["status"], report["compiler"], report["artifact"] = compile_prepared(
                root, selection, config, executable
            )
    except Exception as error:
        report.update(
            status="preparation_failed", error=f"{type(error).__name__}: {error}"
        )
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "conversion.json", report)
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True, type=Path)
    p.add_argument("--names", required=True, type=Path)
    p.add_argument("--target", required=True, choices=("x5", "s100", "s100p"))
    p.add_argument("--variant", required=True)
    p.add_argument("--cal-images", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sample-count", type=int, default=100)
    p.add_argument("--compile", action="store_true")
    p.add_argument(
        "--compiler",
        default=None,
        help="Explicit executable; otherwise hb_mapper (X5) or hb_compile (S).",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = prepare_conversion(
            onnx_path=args.onnx,
            names_path=args.names,
            target=args.target,
            variant=args.variant,
            images=args.cal_images,
            output_dir=args.output_dir,
            sample_count=args.sample_count,
            compile_model=args.compile,
            compiler=args.compiler,
        )
        print(json.dumps(report, indent=2))
        return 1 if report["status"] == "compile_failed" else 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
