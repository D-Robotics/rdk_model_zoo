"""MobileSAM dual-bin full-mask inference entry script."""

from __future__ import annotations

import argparse
import logging
import os

import cv2

from mobile_sam import MobileSAMConfig, MobileSAMSegment, draw_mask_result

logging.basicConfig(level=logging.INFO, format="[%(name)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("MobileSAM")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_ENCODER_PATH = os.path.join(SAMPLE_DIR, "model", "mobile_sam_image_encoder_norm_512x512_allint16.bin")
DEFAULT_DECODER_PATH = os.path.join(SAMPLE_DIR, "model", "mobile_sam_decoder_512_box_default.bin")
DEFAULT_TEST_IMAGE = os.path.join(SAMPLE_DIR, "test_data", "dogs.jpg")
DEFAULT_RESULT_IMAGE = os.path.join(SAMPLE_DIR, "test_data", "mobile_sam_full_mask_result.jpg")
DEFAULT_MASK_IMAGE = os.path.join(SAMPLE_DIR, "test_data", "mobile_sam_binary_mask.png")


def save_outputs(result, image, result_path: str, mask_path: str) -> None:
    """Save the overlay image and binary mask from a prediction result.

    Args:
        result: Prediction dictionary returned by `MobileSAMSegment.predict`.
        image: Original BGR input image.
        result_path: Destination path for the mask-overlay visualization.
        mask_path: Destination path for the binary mask image.

    Raises:
        RuntimeError: If either output image cannot be written.
    """

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    overlay = draw_mask_result(image, result["mask"], result["iou"], result["mask_index"])
    if not cv2.imwrite(result_path, overlay):
        raise RuntimeError(f"Failed to save result image: {result_path}")
    if not cv2.imwrite(mask_path, result["mask"].astype("uint8") * 255):
        raise RuntimeError(f"Failed to save mask image: {mask_path}")


def parse_box(value: str) -> tuple[float, float, float, float]:
    """Parse a box prompt from `x1,y1,x2,y2` CLI text.

    Args:
        value: Comma-separated box coordinates in resized 512x512 image space.

    Returns:
        Four floating-point box coordinates as `(x1, y1, x2, y2)`.

    Raises:
        argparse.ArgumentTypeError: If the value does not contain four numbers.
    """

    parts = [float(x) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("box must be x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    """Parse CLI arguments and run the complete MobileSAM demo."""

    parser = argparse.ArgumentParser(description="MobileSAM dual-bin full mask demo")
    parser.add_argument("--encoder-model-path", type=str, default=DEFAULT_ENCODER_PATH, help="Path to quantized encoder .bin model.")
    parser.add_argument("--decoder-model-path", type=str, default=DEFAULT_DECODER_PATH, help="Path to quantized decoder .bin model.")
    parser.add_argument("--test-img", type=str, default=DEFAULT_TEST_IMAGE, help="Path to input image.")
    parser.add_argument("--img-save-path", type=str, default=DEFAULT_RESULT_IMAGE, help="Path to save mask overlay image.")
    parser.add_argument("--mask-save-path", type=str, default=DEFAULT_MASK_IMAGE, help="Path to save binary mask image.")
    parser.add_argument("--box", type=parse_box, default=(185.0, 120.0, 380.0, 445.0), help="Box prompt as x1,y1,x2,y2 in 512x512 image coordinates.")
    parser.add_argument("--priority", type=int, default=0, help="Model scheduling priority.")
    args = parser.parse_args()

    image = cv2.imread(args.test_img, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.test_img)

    config = MobileSAMConfig(encoder_model_path=args.encoder_model_path, decoder_model_path=args.decoder_model_path, box=args.box)
    model = MobileSAMSegment(config)
    model.set_scheduling_params(priority=args.priority)
    result = model.predict(image)
    save_outputs(result, image, args.img_save_path, args.mask_save_path)
    logger.info('Saving results to "%s"', args.img_save_path)
    logger.info('Predicted IoU %.4f, mask index %d', result["iou"], result["mask_index"])


if __name__ == "__main__":
    main()
