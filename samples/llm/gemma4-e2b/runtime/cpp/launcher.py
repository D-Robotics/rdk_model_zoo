# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Native Gemma orchestration; preparation is explicit, preview is offline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CPP = Path(__file__).resolve().parent
sys.path.insert(0, str(CPP.parents[4]))
from samples._shared.platforms import require_execution_target, resolve_target

APPS = {
    "main": "main",
    "server": "gemma4_server",
    "demo": "gemma4_demo",
    "text_bench": "gemma4_text_bench",
    "golden_verify": "gemma4_golden_verify",
}
APPS.update({binary: binary for binary in tuple(APPS.values())})


def execution_environment(target, home, build):
    """Copy source runtime settings without altering the parent environment."""
    env = os.environ.copy()
    env.update(GEMMA4_HOME=str(home), GEMMA4_TARGET=target, GEMMA4_BUILD_DIR=str(build))
    if target == "s600":
        env.pop("LD_LIBRARY_PATH", None)
        env.pop("GEMMA4_USE_DNN_V3", None)
        env.setdefault("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Gemma C++ launcher. Prepare models/dependencies separately. "
        "Launcher options precede APP; everything after APP goes to the native binary."
    )
    parser.add_argument(
        "--target",
        default="auto",
        help="auto/s100/s100p/s600; S100 requires manual HBMs",
    )
    parser.add_argument(
        "--home", type=Path, default=Path(os.environ.get("GEMMA4_HOME", "~/gemma4_e2b"))
    )
    parser.add_argument("--build-dir", type=Path, default=CPP / "build")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print command and settings without execution",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="build only; never install packages or download models",
    )
    parser.add_argument("app", nargs="?", default=os.environ.get("GEMMA4_APP", "main"))
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        target = resolve_target(args.target)
        if target not in ("s100", "s100p", "s600"):
            raise ValueError("Gemma requires s100/s100p/s600; no X5 runtime")
        if args.app not in APPS:
            raise ValueError(
                "Unknown app; choose main/server/demo/text_bench/golden_verify"
            )
        home, build = (
            args.home.expanduser().resolve(),
            args.build_dir.expanduser().resolve(),
        )
        native_args = (
            args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
        )
        native = [str(build / APPS[args.app]), *native_args]
        if args.build and native_args:
            raise ValueError(
                "--build is build-only and does not accept native arguments"
            )
        env = execution_environment(target, home, build)
        build_command = ["bash", str(CPP / "build.sh")]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": target,
                        "home": str(home),
                        "executed": False,
                        "action": "build" if args.build else "run",
                        "build_argv": build_command,
                        "native_argv": native,
                        "environment": {
                            key: env[key]
                            for key in (
                                "GEMMA4_HOME",
                                "GEMMA4_TARGET",
                                "GEMMA4_BUILD_DIR",
                            )
                        },
                        "asset_validation": "not performed; use matching target HBMs",
                    },
                    indent=2,
                )
            )
            return 0
        require_execution_target(target)
        if args.build:
            return subprocess.run(build_command, env=env, check=False).returncode
        if not Path(native[0]).is_file() or not os.access(native[0], os.X_OK):
            raise ValueError(
                "Native executable missing; prepare dependencies then run --build"
            )
        return subprocess.run(native, env=env, check=False).returncode
    except (ValueError, OSError) as error:
        print(f"Gemma launcher: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
