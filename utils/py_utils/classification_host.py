"""Host-side classification contracts, image preparation, and dataset checks.

The same RGB crop is usable by a float evaluator, calibration writer, and a
board adapter. Heavy export frameworks are deliberately not imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    """Hash file bytes without loading a model or dataset into memory.

    Args:
        path: Existing file.

    Returns:
        Lowercase SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write a new JSON artifact, rejecting overwrite and nonfinite values.

    Args:
        path: New output file.
        value: JSON-serializable receipt.

    Raises:
        FileExistsError: The artifact already exists.
    """
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def validate_contract(contract: dict) -> None:
    """Reject unsupported geometry, normalization, and output contracts.

    Args:
        contract: Explicit classification input/output settings.

    Raises:
        ValueError: A setting cannot be implemented faithfully.
    """
    size = contract["size"]
    crop = contract["crop_pct"]
    if not isinstance(size, int) or size <= 0 or size % 2:
        raise ValueError("size must be a positive even integer")
    if not math.isfinite(crop) or not 0 < crop <= 1:
        raise ValueError("crop_pct must be in (0, 1]")
    for name in ("mean", "std"):
        values = np.asarray(contract[name], dtype=np.float32)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain three finite values")
        if name == "std" and np.any(values <= 0):
            raise ValueError("std must be positive")
    expected = {"geometry": "resize_shorter_center_crop", "interpolation": "pil_bicubic",
                "color": "RGB", "layout": "NCHW", "dtype": "float32",
                "class_count": 1000, "output": "logits", "batch": 1}
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"Unsupported {key}: {contract.get(key)!r}")


def prepare_rgb(source: Path | Image.Image, contract: dict) -> np.ndarray:
    """Resize the shorter edge and center-crop with timm/PIL rounding rules.

    Args:
        source: Image path or an open PIL image.
        contract: Validated classification contract.

    Returns:
        Contiguous uint8 RGB HWC crop. No normalization or NV12 roundtrip.
    """
    validate_contract(contract)
    if isinstance(source, Image.Image):
        rgb = source.convert("RGB")
    else:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
    size = contract["size"]
    short = int(size / contract["crop_pct"])
    width, height = rgb.size
    if width <= height:
        resized = (short, int(short * height / width))
    else:
        resized = (int(short * width / height), short)
    if resized != rgb.size:
        # Pillow 9.0 on the S100 image predates the Resampling enum.
        resampling = getattr(Image, "Resampling", Image)
        rgb = rgb.resize(resized, resampling.BICUBIC)
    left = int(round((resized[0] - size) / 2.0))
    top = int(round((resized[1] - size) / 2.0))
    return np.ascontiguousarray(rgb.crop((left, top, left + size, top + size)))


def float_input(rgb: np.ndarray, contract: dict) -> np.ndarray:
    """Normalize a prepared RGB crop once, producing batch-one NCHW.

    Args:
        rgb: Uint8 RGB crop from prepare_rgb.
        contract: Explicit mean/std and input size.

    Returns:
        Contiguous float32 array shaped [1, 3, size, size].

    Raises:
        ValueError: The supplied crop has the wrong dtype or shape.
    """
    if rgb.dtype != np.uint8 or rgb.shape != (contract["size"], contract["size"], 3):
        raise ValueError("Expected a prepared uint8 RGB crop")
    data = rgb.astype(np.float32) / np.float32(255)
    data = (data - np.asarray(contract["mean"], dtype=np.float32)) / np.asarray(
        contract["std"], dtype=np.float32)
    return np.ascontiguousarray(data.transpose(2, 0, 1)[None])


def calibration_input(rgb: np.ndarray, contract: dict, platform: str) -> np.ndarray:
    """Prepare calibration bytes for the selected installed OE loader.

    Args:
        rgb: Prepared uint8 RGB crop.
        contract: Classification normalization contract.
        platform: x5, s100, s100p, or s600.

    Returns:
        Float32 NCHW: raw RGB [0,255] on X5; original ONNX domain on S.

    Raises:
        ValueError: The platform is unknown.
    """
    normalized = float_input(rgb, contract)
    if platform == "x5":
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    if platform in ("s100", "s100p", "s600"):
        return normalized
    raise ValueError(f"Unknown platform: {platform}")


def load_dataset(manifest_path: Path, root: Path, *, expected_count: int,
                 labeled: bool = True) -> tuple[dict, list[tuple[Path, dict]]]:
    """Validate a complete manifest, every image hash, and numeric labels.

    Args:
        manifest_path: Dataset manifest with ordered images and hashes.
        root: Root containing manifest-relative images (symlinks are allowed).
        expected_count: Frozen full-set image count; no implicit truncation.
        labeled: Require zero-based ImageNet class IDs when true.

    Returns:
        Parsed metadata and ordered (path, record) pairs.

    Raises:
        ValueError: Counts, hashes, paths, or labels are invalid.
    """
    manifest = json.loads(manifest_path.read_text())
    records = manifest["images"]
    if expected_count <= 0 or manifest["count"] != expected_count or len(records) != expected_count:
        raise ValueError("Full dataset count mismatch")
    if labeled and manifest.get("class_count") != 1000:
        raise ValueError("Expected 1000 ImageNet classes")
    seen = set()
    result = []
    for record in records:
        relative = Path(record.get("path", record.get("name", "")))
        if not str(relative) or relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Image path must be relative to the dataset root")
        if str(relative) in seen:
            raise ValueError(f"Duplicate image: {relative}")
        seen.add(str(relative))
        path = root / relative
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Image hash mismatch: {relative}")
        if labeled:
            label = record.get("label_id")
            if type(label) is not int or not 0 <= label < 1000:
                raise ValueError(f"Invalid class ID: {label}")
        result.append((path, record))
    return manifest, result


def top5_ids(scores: np.ndarray) -> list[int]:
    """Rank one finite 1000-class output with deterministic tie breaking.

    Args:
        scores: A flat vector or batch-one logits tensor.

    Returns:
        Five descending-score indices; ties use ascending class ID.

    Raises:
        ValueError: Output is malformed or nonfinite.
    """
    scores = np.asarray(scores)
    if scores.size != 1000 or scores.shape not in ((1000,), (1, 1000), (1, 1000, 1, 1)):
        raise ValueError(f"Expected one 1000-class output, got {scores.shape}")
    scores = scores.reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite model output")
    return np.argsort(-scores, kind="stable")[:5].tolist()
