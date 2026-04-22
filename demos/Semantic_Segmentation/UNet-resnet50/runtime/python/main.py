"""Main entry point for UNet-resnet50 semantic segmentation inference.

This script demonstrates how to use the UNet model for single-image
semantic segmentation with command-line configurable parameters.

Typical Usage:
    # Run with default parameters
    python main.py

    # Run with custom model and image
    python main.py --model-path /path/to/model.bin --img-path /path/to/image.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

from unet import UNet, UNetConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="UNet-resnet50 Semantic Segmentation Inference"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to BPU Quantized *.bin Model. If not provided, a local default path is used.",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        help="Path to the input image for inference.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="unet_result.jpg",
        help="Path to save the visualization result.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default="unet_mask.png",
        help="Path to save the raw segmentation mask.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=21,
        help="Number of segmentation classes (including background).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Opacity of the overlay mask in visualization (0.0 to 1.0).",
    )

    return parser.parse_args()


def main() -> int:
    """Main inference entry point.

    Returns:
        int: Exit code (0 for success, non-zero for failure).
    """
    args = parse_args()

    # Initialize configuration and model
    config = UNetConfig(
        model_path=args.model_path,
        num_classes=args.num_classes,
    )
    model = UNet(config)

    # Load input image
    image = cv2.imread(args.img_path)
    if image is None:
        print(f"Error: Failed to load image from {args.img_path}", file=sys.stderr)
        return 1

    # Run inference
    mask = model.predict(image)

    # Visualize result
    result = model.visualize(image, mask, alpha=args.alpha)

    # Save outputs
    cv2.imwrite(args.save_path, result)
    cv2.imwrite(args.mask_path, mask * 12)  # Scale for visibility

    print(f"Result saved to: {args.save_path}")
    print(f"Mask saved to: {args.mask_path}")
    print(f"Predicted classes: {np.unique(mask)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
