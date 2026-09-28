"""Preview, validate, build and execute the native HIMLoco application."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
CPP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from samples.robotics.himloco.runtime.python.model_binding import (
    resolve_selection,
    SAMPLE_DIR,
)
from samples._shared.assets import verify_asset_file
from samples._shared.platforms import require_execution_target


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target", choices=("auto", "x5", "s100", "s100p", "s600"), default="auto"
    )
    p.add_argument("--asset-id")
    p.add_argument("--model-path", "--model_path", type=Path)
    p.add_argument(
        "--input-path",
        "--input_path",
        type=Path,
        default=SAMPLE_DIR / "test_data/obs_history",
    )
    p.add_argument(
        "--output-dir", "--output_dir", type=Path, default=Path("outputs/himloco_cpp")
    )
    p.add_argument("--report", type=Path)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--priority", type=int, default=-1)
    p.add_argument("--build-dir", type=Path, default=CPP / "build")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--list-models", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if not 0 <= args.warmup <= 1000000 or not -1 <= args.priority <= 255:
            raise ValueError("warmup must be 0..1000000; priority must be -1 or 0..255")
        if args.dry_run and args.target == "auto":
            raise ValueError("Host preview requires --target x5")
        target = "x5" if args.list_models and args.target == "auto" else args.target
        selection = resolve_selection(
            target, model_path=args.model_path, asset_id=args.asset_id
        )
        if args.list_models:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "filename": selection.asset.filename,
                    },
                    indent=2,
                )
            )
            return 0
        build = args.build_dir.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        report = (
            args.report.expanduser().resolve()
            if args.report
            else output / "report.json"
        )
        native = [
            str(build / "himloco_cpp"),
            "--target",
            "x5",
            "--asset-id",
            selection.asset.reference,
            "--model-path",
            str(selection.model_path.resolve()),
            "--input-path",
            str(args.input_path.expanduser().resolve()),
            "--output-dir",
            str(output),
            "--report",
            str(report),
            "--warmup",
            str(args.warmup),
            "--priority",
            str(args.priority),
        ]
        configure = [
            "cmake",
            "-S",
            str(CPP),
            "-B",
            str(build),
            "-DHIMLOCO_BUILD_SDK=ON",
            "-DHIMLOCO_BUILD_CLI=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        compile_command = [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            str(os.cpu_count() or 1),
        ]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": selection.target,
                        "asset_id": selection.asset.reference,
                        "built": False,
                        "downloaded": False,
                        "sdk_loaded": False,
                        "configure_argv": configure,
                        "build_argv": compile_command,
                        "native_argv": native,
                    },
                    indent=2,
                )
            )
            return 0
        require_execution_target(selection.target)
        verify_asset_file(selection.asset, selection.model_path)
        if (
            output.exists()
            or output.is_symlink()
            or report.exists()
            or report.is_symlink()
            or report == output
        ):
            raise ValueError("Output directory and report must be new and distinct")
        for command in (configure, compile_command):
            subprocess.run(command, check=True)
        return subprocess.run(native, check=False).returncode
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"HIMLoco native launcher: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
