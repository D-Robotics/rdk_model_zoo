"""Download one published ResNet artifact through the shared manifest.

The script deliberately contains only stable target/variant-to-reference
mappings. URL, format, and optional publisher hash facts are read from the
platform manifest by ``utils.py_utils.assets``. It is safe to import this
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

from utils.py_utils.assets import download_asset, resolve_asset


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
VARIANTS = ('18', '34', '50', '101', '152')
# Input sizes: 18 224x224, 34 224x224, 50 224x224, 101 224x224, 152 224x224.  File names carry the toolchain march token
# (bayese, nashe, nashm, nashp), so each file identifies its chip.
ASSET_REFERENCES = {
    ('x5', '18'): 'x5:resnet:resnet18_bayese_224x224_nv12.bin',
    ('s100', '18'): 's:resnet:s100/resnet18_nashe_224x224_nv12.hbm',
    ('s100p', '18'): 's:resnet:s100p/resnet18_nashm_224x224_nv12.hbm',
    ('s600', '18'): 's:resnet:s600/resnet18_nashp_224x224_nv12.hbm',
    ('x5', '34'): 'x5:resnet:resnet34_bayese_224x224_nv12.bin',
    ('s100', '34'): 's:resnet:s100/resnet34_nashe_224x224_nv12.hbm',
    ('s100p', '34'): 's:resnet:s100p/resnet34_nashm_224x224_nv12.hbm',
    ('s600', '34'): 's:resnet:s600/resnet34_nashp_224x224_nv12.hbm',
    ('x5', '50'): 'x5:resnet:resnet50_bayese_224x224_nv12.bin',
    ('s100', '50'): 's:resnet:s100/resnet50_nashe_224x224_nv12.hbm',
    ('s100p', '50'): 's:resnet:s100p/resnet50_nashm_224x224_nv12.hbm',
    ('s600', '50'): 's:resnet:s600/resnet50_nashp_224x224_nv12.hbm',
    ('x5', '101'): 'x5:resnet:resnet101_bayese_224x224_nv12.bin',
    ('s100', '101'): 's:resnet:s100/resnet101_nashe_224x224_nv12.hbm',
    ('s100p', '101'): 's:resnet:s100p/resnet101_nashm_224x224_nv12.hbm',
    ('s600', '101'): 's:resnet:s600/resnet101_nashp_224x224_nv12.hbm',
    ('x5', '152'): 'x5:resnet:resnet152_bayese_224x224_nv12.bin',
    ('s100', '152'): 's:resnet:s100/resnet152_nashe_224x224_nv12.hbm',
    ('s100p', '152'): 's:resnet:s100p/resnet152_nashm_224x224_nv12.hbm',
    ('s600', '152'): 's:resnet:s600/resnet152_nashp_224x224_nv12.hbm',
}
TARGETS = ("x5", "s100", "s100p", "s600")


def asset_reference(target: str, variant: str = '18') -> str:
    """Return the exact manifest reference for a supported target/variant."""

    key = (str(target).strip().lower(), str(variant).strip().lower())
    try:
        return ASSET_REFERENCES[key]
    except KeyError as exc:
        available = ", ".join("/".join(k) for k in ASSET_REFERENCES)
        raise ValueError(
            f"Unsupported ResNet target/variant {target}; "
            f"published combinations: {available}."
        ) from exc


def download_target(
    target: str,
    output_dir: Optional[str | Path] = None,
    *,
    variant: str = '18',
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
        description="Download one manifest-backed ResNet deployment artifact."
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
        default='18',
        help="Model variant to fetch (default: 18).",
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
