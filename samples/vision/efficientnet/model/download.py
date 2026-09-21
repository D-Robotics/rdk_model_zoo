"""Download one published EfficientNet artifact through the shared manifest.

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
VARIANTS = ('b2', 'b3', 'b4', 'lite0', 'lite1', 'lite2', 'lite3', 'lite4')
ASSET_REFERENCES = {
    ('x5', 'b2'): 'x5:efficientnet:EfficientNet_B2_224x224_nv12.bin',
    ('x5', 'b3'): 'x5:efficientnet:EfficientNet_B3_224x224_nv12.bin',
    ('x5', 'b4'): 'x5:efficientnet:EfficientNet_B4_224x224_nv12.bin',
    ('s100', 'lite0'): 's:efficientnet:s100/efficientnet_lite0_224x224_nv12.hbm',
    ('s100', 'lite1'): 's:efficientnet:s100/efficientnet_lite1_240x240_nv12.hbm',
    ('s100', 'lite2'): 's:efficientnet:s100/efficientnet_lite2_260x260_nv12.hbm',
    ('s100', 'lite3'): 's:efficientnet:s100/efficientnet_lite3_300x300_nv12.hbm',
    ('s100', 'lite4'): 's:efficientnet:s100/efficientnet_lite4_380x380_nv12.hbm',
    ('s600', 'lite0'): 's:efficientnet:s600/efficientnet_lite0_224x224_nv12.hbm',
    ('s600', 'lite1'): 's:efficientnet:s600/efficientnet_lite1_240x240_nv12.hbm',
    ('s600', 'lite2'): 's:efficientnet:s600/efficientnet_lite2_260x260_nv12.hbm',
    ('s600', 'lite3'): 's:efficientnet:s600/efficientnet_lite3_300x300_nv12.hbm',
    ('s600', 'lite4'): 's:efficientnet:s600/efficientnet_lite4_380x380_nv12.hbm',
}
TARGETS = ("x5", "s100", "s600")


def asset_reference(target: str, variant: str = 'b2') -> str:
    """Return the exact manifest reference for a supported target/variant."""

    key = (str(target).strip().lower(), str(variant).strip().lower())
    try:
        return ASSET_REFERENCES[key]
    except KeyError as exc:
        available = ", ".join("/".join(k) for k in ASSET_REFERENCES)
        raise ValueError(
            f"Unsupported EfficientNet target/variant {target}; "
            f"published combinations: {available}."
        ) from exc


def download_target(
    target: str,
    output_dir: Optional[str | Path] = None,
    *,
    variant: str = 'b2',
) -> str:
    """Download an exact target artifact and return its observed SHA-256.

    ``output_dir`` preserves the source sample's layout: X5 is flat while the
    S-series artifact remains under ``s100/`` or ``s600/``. Existing files are
    verified by the shared downloader and are never silently replaced.
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
        description="Download one manifest-backed EfficientNet deployment artifact."
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
        default='b2',
        help="Model variant to fetch (default: b2).",
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
