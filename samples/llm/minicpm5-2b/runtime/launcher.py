# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Select the MiniCPM SDK family; never download or build implicitly."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

RUNTIME = Path(__file__).resolve().parent
SAMPLE = RUNTIME.parent
sys.path.insert(0, str(SAMPLE.parents[2]))
from samples._shared.platforms import require_execution_target, resolve_target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="auto")
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("MODEL_DIR"))
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=os.environ.get("OELLM_RUNTIME_ROOT")
        or (
            str(Path(os.environ["OELLM_SDK_ROOT"]) / "oellm_runtime")
            if os.environ.get("OELLM_SDK_ROOT")
            else None
        ),
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--timeout",
        type=float,
        default=os.environ.get("INFERENCE_TIMEOUT", "120"),
        help="S100/S100P execution limit in seconds; source default 120",
    )
    parser.add_argument(
        "native_args", nargs=argparse.REMAINDER, help="put -- before native arguments"
    )
    args = parser.parse_args(argv)
    try:
        target = resolve_target(args.target)
        if target not in ("s100", "s100p", "s600"):
            raise ValueError("MiniCPM requires s100/s100p/s600")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("--timeout must be positive and finite")
        native = (
            args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
        )
        if args.build and native:
            raise ValueError("--build does not accept inference arguments")
        backend = "cpp" if target == "s600" else "legacy"
        source = RUNTIME / backend
        build = (args.build_dir or source / ("build-" + target)).expanduser().resolve()
        model = (args.model_dir or SAMPLE / "model" / target).expanduser().resolve()
        sdk = args.runtime_root.expanduser().resolve() if args.runtime_root else None
        binary = build / "main"
        if target == "s600":
            command = [str(binary), "--model_path=" + str(model), *native]
        else:
            command = [
                str(binary),
                "--model-path",
                str(model / f"minicpm5-2b_ctx4096_{target}.hbm"),
                "--tokenizer-path",
                str(model / "tokenizer"),
                "--template-path",
                str(model / "tokenizer/simple-chat.jinja"),
                *native,
            ]
        commands = [
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-DOELLM_RUNTIME_ROOT=" + (str(sdk) if sdk else "<set-runtime-root>"),
                "-DMINICPM_TARGET=" + target,
            ],
            [
                "cmake",
                "--build",
                str(build),
                "--parallel",
                "4" if target == "s600" else "2",
            ],
        ]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target": target,
                        "backend": backend,
                        "sdk_family": "2.0.0-beta1" if target == "s600" else "1.0.0",
                        "model_dir": str(model),
                        "runtime_root": str(sdk) if sdk else None,
                        "action": "build" if args.build else "run",
                        "executed": False,
                        "argv": command,
                        "commands": commands if args.build else [command],
                        "timeout_seconds": None if target == "s600" else args.timeout,
                    },
                    indent=2,
                )
            )
            return 0
        require_execution_target(target)
        header = "oellm_runtime_basic/oellm_runtime.h" if target == "s600" else "xlm.h"
        if sdk is None or not (sdk / "include" / header).is_file():
            raise ValueError(
                f"Prepare the matching SDK and set --runtime-root; expected include/{header}"
            )
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(sdk / "lib") + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
        if target == "s600":
            env["HB_DNN_USER_DEFINED_L2M_SIZES"] = "6:6:6:6"
        if args.build:
            for cmd in commands:
                result = subprocess.run(cmd, env=env, check=False)
                if result.returncode:
                    return result.returncode
            return 0
        if not binary.is_file():
            raise ValueError(
                "Native binary missing; prepare dependencies then run --build explicitly"
            )
        return subprocess.run(
            command,
            cwd=source,
            env=env,
            timeout=None if target == "s600" else args.timeout,
            check=False,
        ).returncode
    except subprocess.TimeoutExpired:
        print("ERROR: inference timeout expired", file=sys.stderr)
        return 124
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
