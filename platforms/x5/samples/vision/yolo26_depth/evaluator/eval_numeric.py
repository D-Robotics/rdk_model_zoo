"""Compare one X5 depth output with an official floating-point reference."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    """Calculate a SHA256 digest for a comparison artifact.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colorize(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    """Colorize depth with one shared range for visual comparison.

    Args:
        depth: Depth map to visualize.
        low: Lower visualization bound.
        high: Upper visualization bound.

    Returns:
        BGR Turbo visualization.
    """
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    gray = np.asarray(normalized * 255.0, dtype=np.uint8)
    return cv2.applyColorMap(255 - gray, cv2.COLORMAP_TURBO)


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit one visualization into a fixed contact-sheet cell.

    Args:
        image: BGR image to place in the cell.
        width: Target cell width.
        height: Target cell height.

    Returns:
        Letterboxed BGR cell image.
    """
    source_height, source_width = image.shape[:2]
    ratio = min(width / source_width, height / source_height)
    resized_width = round(source_width * ratio)
    resized_height = round(source_height * ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 32, dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def fidelity(candidate: np.ndarray, reference: np.ndarray) -> dict:
    """Calculate pixel-wise fidelity against an official reference.

    Args:
        candidate: X5 relative-depth result.
        reference: Floating-point reference depth.

    Returns:
        Dictionary containing error and similarity metrics.
    """
    candidate64 = candidate.astype(np.float64)
    reference64 = reference.astype(np.float64)
    delta = candidate64 - reference64
    denominator = np.linalg.norm(candidate64.ravel()) * np.linalg.norm(reference64.ravel())
    ratio = np.maximum(
        candidate64 / np.clip(reference64, 1e-6, None),
        reference64 / np.clip(candidate64, 1e-6, None),
    )
    return {
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "mean_relative_abs": float((np.abs(delta) / np.clip(reference64, 1e-6, None)).mean()),
        "delta1": float((ratio < 1.25).mean()),
        "cosine_similarity": float(np.dot(candidate64.ravel(), reference64.ravel()) / denominator),
        "max_abs": float(np.abs(delta).max()),
    }


def main() -> None:
    """Parse arguments and write a single-image comparison report."""
    parser = argparse.ArgumentParser(description="Compare official and X5 depth output on one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--x5", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    official = np.load(args.official).astype(np.float32)
    x5 = np.load(args.x5).astype(np.float32)
    if official.shape != x5.shape or image.shape[:2] != official.shape:
        raise ValueError(
            f"shape mismatch: image={image.shape[:2]}, official={official.shape}, x5={x5.shape}"
        )

    valid = np.isfinite(official) & np.isfinite(x5) & (official > 1e-6) & (x5 > 1e-6)
    median_scale = float(np.median(official[valid]) / np.median(x5[valid]))
    x5_aligned = x5 * median_scale
    low, high = np.percentile(official[valid], [2.0, 98.0])

    official_color = colorize(official, float(low), float(high))
    x5_color = colorize(x5_aligned, float(low), float(high))
    relative_error = np.zeros_like(official, dtype=np.float32)
    relative_error[valid] = np.abs(x5_aligned[valid] - official[valid]) / official[valid]
    error_gray = np.asarray(np.clip(relative_error / 0.5, 0.0, 1.0) * 255.0, dtype=np.uint8)
    error_color = cv2.applyColorMap(error_gray, cv2.COLORMAP_INFERNO)

    official_overlay = cv2.addWeighted(image, 0.45, official_color, 0.55, 0.0)
    x5_overlay = cv2.addWeighted(image, 0.45, x5_color, 0.55, 0.0)
    np.save(args.output / "x5_depth_median_aligned.npy", x5_aligned)
    cv2.imwrite(str(args.output / "official_depth_common_range.png"), official_color)
    cv2.imwrite(str(args.output / "x5_depth_median_aligned_common_range.png"), x5_color)
    cv2.imwrite(str(args.output / "absolute_relative_error.png"), error_color)
    cv2.imwrite(str(args.output / "official_overlay_common_range.png"), official_overlay)
    cv2.imwrite(str(args.output / "x5_overlay_median_aligned_common_range.png"), x5_overlay)

    cell_width, cell_height, header_height = 405, 540, 46
    titles = ["Input", "Official FP32", "X5 PTQ (median-aligned)", "Absolute relative error"]
    images = [image, official_color, x5_color, error_color]
    sheet = np.full((cell_height + header_height, cell_width * len(images), 3), 245, dtype=np.uint8)
    for index, (title, visual) in enumerate(zip(titles, images, strict=True)):
        left = index * cell_width
        cv2.putText(
            sheet,
            title,
            (left + 10, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        sheet[header_height:, left : left + cell_width] = fit(visual, cell_width, cell_height)
    cv2.imwrite(str(args.output / "bus_official_vs_x5.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])

    report = {
        "schema_version": "1.0",
        "reference": "official Ultralytics YOLO26n Depth FP32 predictor",
        "candidate": "RDK X5 PTQ N-768 max-p9999 O3 int16-tailconv",
        "image": str(args.image),
        "image_sha256": sha256(args.image),
        "official_output": str(args.official),
        "official_output_sha256": sha256(args.official),
        "x5_output": str(args.x5),
        "x5_output_sha256": sha256(args.x5),
        "shape": list(official.shape),
        "raw": fidelity(x5, official),
        "median_scale_x5_to_official": median_scale,
        "median_aligned": fidelity(x5_aligned, official),
        "common_color_range_percentile_2_98": [float(low), float(high)],
    }
    (args.output / "comparison-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
