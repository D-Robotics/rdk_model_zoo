# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0

"""Explicit preparation of the one published S100 R3D-18 asset."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples._shared.assets import download_asset  # noqa: E402
_binding = importlib.import_module("samples.vision.3dresnet.runtime.python.model_binding")
SUPPORTED_TARGETS = _binding.SUPPORTED_TARGETS
resolve_selection = _binding.resolve_selection

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def download_target(target: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> str:
    selection = resolve_selection(target)
    destination = Path(output_dir).expanduser() / selection.asset.filename
    digest = download_asset(selection.asset, destination)
    print(f"Downloaded {selection.asset.reference} to {destination}")
    print(f"Observed SHA-256: {digest}")
    if selection.asset.sha256 is None:
        print("Publisher SHA-256 is unknown; observed digest does not independently verify origin.")
    return digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the published R3D-18 HBM asset.")
    parser.add_argument("--target", choices=SUPPORTED_TARGETS, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        download_target(args.target, args.output_dir)
    except (ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
