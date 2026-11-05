"""Download one published MobileNetV3 artifact through the shared manifest.

Map a target and variant to one manifest reference and download that file.
URLs and publisher hashes come from the platform manifest
(``utils.py_utils.assets``). Importing this module does not load the board
SDK or use the network; ``download_target`` performs the download.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

# Allow running this file from any working directory.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.py_utils.assets import download_asset, resolve_asset


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
VARIANTS = ('large', 'small')
# Input sizes: large 224x224, small 224x224.  File names carry the toolchain march token
# (bayese, nashe, nashm, nashp), so each file identifies its chip.
ASSET_REFERENCES = {
    ('s100', 'large'): 's:mobilenetv3:s100/mobilenetv3_large_100_nashe_224x224_nv12.hbm',
    ('s100p', 'large'): 's:mobilenetv3:s100p/mobilenetv3_large_100_nashm_224x224_nv12.hbm',
    ('s600', 'large'): 's:mobilenetv3:s600/mobilenetv3_large_100_nashp_224x224_nv12.hbm',
    ('s100', 'small'): 's:mobilenetv3:s100/mobilenetv3_small_100_nashe_224x224_nv12.hbm',
    ('s100p', 'small'): 's:mobilenetv3:s100p/mobilenetv3_small_100_nashm_224x224_nv12.hbm',
    ('s600', 'small'): 's:mobilenetv3:s600/mobilenetv3_small_100_nashp_224x224_nv12.hbm',
}
TARGETS = ('s100', 's100p', 's600')


def asset_reference(target: str, variant: str = 'large') -> str:
    """Return the exact manifest reference for a supported target/variant."""

    key = (str(target).strip().lower(), str(variant).strip().lower())
    try:
        return ASSET_REFERENCES[key]
    except KeyError as exc:
        available = ", ".join("/".join(k) for k in ASSET_REFERENCES)
        raise ValueError(
            f"Unsupported MobileNetV3 target/variant {target}; "
            f"published combinations: {available}."
        ) from exc


def download_target(
    target: str,
    output_dir: Optional[str | Path] = None,
    *,
    variant: str = 'large',
) -> str:
    """Download an exact target artifact and return its observed SHA-256.

    ``output_dir`` preserves the sample's layout: X5 is flat while the S-series
    artifact stays under ``s100/``, ``s100p/`` or ``s600/``. Existing files are
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
        description="Download one manifest-backed MobileNetV3 deployment artifact."
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default='s100',
        help="Target artifact to fetch (default: s100).",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default='large',
        help="Model variant to fetch (default: large).",
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
