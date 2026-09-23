# Copyright (c) 2026 D-Robotics Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download one published HGNetV2 artifact through the shared manifest.

The script deliberately contains only stable target/variant-to-reference
mappings. URL, format, and optional publisher hash facts are read from the
platform manifest by ``samples._shared.assets``. It is safe to import this
module on a host without the board SDK; network access occurs only when
``download_target`` is called.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

# Direct execution from any working directory is supported. This is the one
# entrypoint-local import-path adjustment; importing the module itself remains
# SDK-free and side-effect free.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from samples._shared.assets import download_asset, resolve_asset


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
VARIANTS = ('b0', 'b1', 'b2', 'b3', 'b4')
ASSET_REFERENCES = {
    ('x5', 'b0'): 'x5:hgnetv2:hgnetv2_b0_224x224_nv12.bin',
    ('x5', 'b1'): 'x5:hgnetv2:hgnetv2_b1_224x224_nv12.bin',
    ('x5', 'b2'): 'x5:hgnetv2:hgnetv2_b2_224x224_nv12.bin',
    ('x5', 'b3'): 'x5:hgnetv2:hgnetv2_b3_224x224_nv12.bin',
    ('x5', 'b4'): 'x5:hgnetv2:hgnetv2_b4_224x224_nv12.bin',
}
TARGETS = ("x5",)


def asset_reference(target: str, variant: str = 'b0') -> str:
    """Return the exact manifest reference for a supported target/variant."""

    key = (str(target).strip().lower(), str(variant).strip().lower())
    try:
        return ASSET_REFERENCES[key]
    except KeyError as exc:
        available = ", ".join("/".join(k) for k in ASSET_REFERENCES)
        raise ValueError(
            f"Unsupported HGNetV2 target/variant {target}; "
            f"published combinations: {available}."
        ) from exc


def download_target(
    target: str,
    output_dir: Optional[str | Path] = None,
    *,
    variant: str = 'b0',
) -> str:
    """Download an exact target artifact and return its observed SHA-256.

    X5 files use the flat model directory. Existing files are verified by
    the shared downloader and are never silently replaced.
    """

    asset = resolve_asset(asset_reference(target, variant))
    root = Path(output_dir).expanduser() if output_dir is not None else DEFAULT_OUTPUT_DIR
    destination = root / Path(asset.filename)
    digest = download_asset(asset, destination)
    print(f"Downloaded {asset.reference} to {destination}")
    print(f"Observed SHA-256: {digest}")
    if asset.sha256 is None:
        print("Publisher SHA-256 is not recorded; origin is not independently verified.")
    return digest


def build_parser() -> argparse.ArgumentParser:
    """Build the dependency-free command-line parser."""

    parser = argparse.ArgumentParser(
        description="Download one manifest-backed HGNetV2 deployment artifact."
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="x5",
        help="Target artifact to fetch (default: x5).",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default='b0',
        help="Model variant to fetch (default: b0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory receiving the manifest filename (default: this model directory).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, download one artifact, and return a shell status."""

    args = build_parser().parse_args(argv)
    try:
        download_target(args.target, args.output_dir, variant=args.variant)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
