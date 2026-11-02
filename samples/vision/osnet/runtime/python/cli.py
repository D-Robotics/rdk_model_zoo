"""OSNet command-line selection, decoded input loading and feature delivery."""

import argparse
import json
from pathlib import Path
import struct

from samples.vision.osnet.model.artifacts import load_manifest

SAMPLE = Path(__file__).resolve().parents[2]


def build_parser():
    """Build the SDK-free OSNet argument parser."""
    parser = argparse.ArgumentParser(description="Extract OSNet person appearance features on RDK X5/S series")
    parser.add_argument("--target", choices=("x5", "s100", "s100p", "s600", "auto"), default="x5", help="Execution target; auto requires recognized physical hardware")
    parser.add_argument("--variant", choices=tuple(load_manifest()["artifacts"]), default="spatial-rgb-late8", help="Registered quantization variant")
    parser.add_argument("--model-path", type=Path, default=None, help="Local BIN/HBM; defaults to the registered model filename")
    parser.add_argument("--test-img", type=Path, default=None, help="Person crop image; defaults to the first prepared C++ fixture crop")
    parser.add_argument("--output", type=Path, default=Path("feature.npy"), help="Output NumPy feature")
    parser.add_argument("--raw", action="store_true", help="Disable L2 normalization")
    parser.add_argument("--dry-run", action="store_true", help="Show selection without loading the SDK")
    parser.add_argument("--list-models", action="store_true", help="List registered artifacts without loading the SDK")
    return parser


def model_path(args):
    """Resolve a local artifact path without downloading or loading it."""
    return args.model_path or SAMPLE / "model" / load_manifest()["artifacts"][args.variant]["filename"]


def print_selection(args):
    """Print preparation metadata; this does not identify physical hardware."""
    path = model_path(args)
    print(json.dumps({"target": load_manifest()["artifacts"][args.variant].get("target", "x5"), "requested_target": args.target, "variant": args.variant,
                      "model_path": str(path), "local_file_present": path.is_file(),
                      "artifact": load_manifest()["artifacts"][args.variant]}, indent=2))


def read_image(path=None, *, fixture_path=None):
    """Load a user crop or the first decoded RGB crop in a prepared fixture.

    Args:
        path: Optional image path understood by Pillow.
        fixture_path: Optional fixture override for host tests.

    Returns:
        PIL.Image.Image: Owned RGB image; no file handle remains open.

    Raises:
        OSError: The file is unavailable or Pillow cannot read it.
        ValueError: The fixture header or pixel payload is invalid.
    """
    from PIL import Image

    if path is not None:
        with Image.open(path) as image:
            return image.convert("RGB")
    fixture = fixture_path or SAMPLE / "test_data/fixture/images.dat"
    with Path(fixture).open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError("Truncated OSNet fixture header")
        count, width, height = struct.unpack("<III", header)
        if not 1 <= count <= 1000 or not 1 <= width <= 16384 or not 1 <= height <= 16384:
            raise ValueError("Invalid OSNet fixture dimensions")
        pixels = stream.read(width * height * 3)
        if len(pixels) != width * height * 3:
            raise ValueError("Truncated OSNet fixture pixels")
    return Image.frombytes("RGB", (width, height), pixels)


def save_feature(feature, output):
    """Write the owned 512-element feature outside the model stages."""
    import numpy as np

    np.save(output, feature)
    print(f"Saved {feature.shape} float32 feature to {output}")
