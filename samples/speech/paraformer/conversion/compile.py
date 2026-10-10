"""Explicit OE compilation of a validated workspace into a separate new run."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from utils.py_utils.assets import sha256_file
from samples.speech.paraformer.runtime.python.cli import write_json
from samples.speech.paraformer.conversion.configuration import STAGES, PREFIXES
from samples.speech.paraformer.conversion.workspace import verify_prepared


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New compile run; partial runs never overwritten",
    )
    parser.add_argument(
        "--compiler",
        default="hb_compile",
        help="OE executable name or path, default hb_compile",
    )
    return parser


def compile_workspace(args):
    import yaml

    root = args.workspace.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if ";" in str(root):
        raise ValueError(
            "Workspace path cannot contain the OE calibration separator semicolon"
        )
    if output.exists():
        raise ValueError("Compile output directory must be new")
    executable = shutil.which(args.compiler)
    if executable is None:
        raise ValueError(
            "OE hb_compile is unavailable; use the matching toolchain environment"
        )
    executable = str(Path(executable).resolve())
    prepared, prepared_sha, configs = verify_prepared(root)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "rdk-model-zoo/paraformer-compile/v1",
        "status": "compiling",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "preparation_sha256": prepared_sha,
        "target": "s100",
        "march": "nash-e",
        "compiler": executable,
        "compiler_sha256": sha256_file(Path(executable)),
        "stages": {},
        "board": "not-run",
        "sdk_metadata": "not-run",
        "dataset_accuracy": "not-run",
        "warnings": prepared.get("warnings", []),
    }
    try:
        for stage in STAGES:
            config = configs[stage]
            config["model_parameters"]["onnx_model"] = str(
                root / f"source/{stage}.onnx"
            )
            config["model_parameters"]["working_dir"] = str(output / stage)
            config["calibration_parameters"]["cal_data_dir"] = ";".join(
                str(root / relative)
                for relative in config["calibration_parameters"]["cal_data_dir"].split(
                    ";"
                )
            )
            config_path = output / f"{stage}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            record = {
                "argv": [executable, "-c", str(config_path)],
                "cwd": str(output),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": sha256_file(config_path),
                "stdout": f"{stage}.stdout.log",
                "stderr": f"{stage}.stderr.log",
                "status": "running",
            }
            report["stages"][stage] = record
            write_json(output / "compile-report.json", report)
            try:
                with (output / record["stdout"]).open("wb") as stdout, (
                    output / record["stderr"]
                ).open("wb") as stderr:
                    process = subprocess.run(
                        record["argv"], cwd=output, stdout=stdout, stderr=stderr
                    )
            except OSError as error:
                record.update(
                    status="failed",
                    returncode=None,
                    error=str(error),
                    finished_utc=datetime.now(timezone.utc).isoformat(),
                )
                raise
            record.update(
                returncode=process.returncode,
                finished_utc=datetime.now(timezone.utc).isoformat(),
            )
            if process.returncode != 0:
                record["status"] = "failed"
                raise RuntimeError(
                    f"{stage} compiler failed (rc={process.returncode}); see retained logs"
                )
            artifact = output / stage / f"{PREFIXES[stage]}.hbm"
            if not artifact.is_file() or artifact.stat().st_size == 0:
                record["status"] = "failed"
                raise RuntimeError(
                    f"{stage} compiler returned zero without a nonempty expected HBM"
                )
            record.update(
                status="compiled_unverified",
                artifact={
                    "path": str(artifact.relative_to(output)),
                    "bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                },
            )
            ptq = output / stage / f"{PREFIXES[stage]}_ptq_model.onnx"
            record["ptq_model"] = (
                {"path": str(ptq.relative_to(output)), "sha256": sha256_file(ptq)}
                if ptq.is_file()
                else None
            )
            write_json(output / "compile-report.json", report)
        _, final_prepared_sha, _ = verify_prepared(root)
        if final_prepared_sha != prepared_sha:
            raise ValueError("Preparation report changed during compilation")
        report["status"] = "compiled_unverified"
    except Exception as error:
        report.update(status="compile_failed", error=str(error))
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "compile-report.json", report)
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        report = compile_workspace(args)
    except Exception as error:
        print(f"Compilation failed: {error}", file=sys.stderr)
        return 2
    print(f"{report['status']}: {args.output_dir / 'compile-report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
