# Copyright (c) 2025 D-Robotics Corporation
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

"""PaddleOCR text detection and recognition inference entry script.

This script runs a two-stage OCR pipeline on a single input image using
BPU-quantized PaddleOCR models (.hbm) and saves a side-by-side result image
showing detected text boxes on the left and recognized text on the right.

The default models are PP-OCRv6, with the model variant selected automatically
from the SoC name read from ``/sys/class/boardinfo/soc_name``:

    - RDK S100  -> rdk_s100/paddle_ocr/...
    - RDK S100P -> rdk_s100/paddle_ocr/...   (reuses the S100 build)
    - RDK S600  -> rdk_s600/paddle_ocr/...
    - unknown   -> rdk_s100/paddle_ocr/...   (fallback)

Workflow:
    1) Parse CLI arguments.
    2) Resolve the model variant from the current SoC.
    3) Download detection and recognition models if missing.
    4) Load character list from label file (prepend blank token).
    5) Initialize PaddleOCRDet and run detection pipeline.
    6) Initialize PaddleOCRRec and run recognition on each cropped region.
    7) Visualize: white canvas with recognized text beside the box image.
    8) Save result to disk.

Notes:
    - The project root is appended to sys.path to import shared utilities
      under ``utils/py_utils/``.

Example:
    python3 main.py \\
        --test-img ../../test_data/gt_2322.jpg \\
        --label-file ../../test_data/ppocrv6_dict.txt \\
        --img-save-path result.jpg
"""

import os
import cv2
import sys
import argparse
import numpy as np

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/inspect.py
#   utils/py_utils/file_io.py
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.file_io as file_io
import utils.py_utils.visualize as vis_utils
from paddle_ocr import PaddleOCRDet, PaddleOCRDetConfig, PaddleOCRRec, PaddleOCRRecConfig


def _resolve_model_soc() -> str:
    """Map the runtime SoC name to the PP-OCRv6 model variant directory.

    PP-OCRv6 is published under ``rdk_s100/paddle_ocr/`` and
    ``rdk_s600/paddle_ocr/``. S100P boards reuse the S100 build, and any
    unknown / unreadable SoC name falls back to S100.

    Returns:
        ``"s100"`` or ``"s600"``.
    """
    soc = inspect.get_soc_name().lower().strip("()").strip()
    if soc == "s600":
        return "s600"
    return "s100"


def main() -> None:
    """Run the PaddleOCR detection + recognition pipeline on a single image.

    This function parses command-line arguments, resolves the SoC-specific
    model variant, downloads missing models, runs the two-stage OCR pipeline,
    visualizes the results, and saves the output image.

    Returns:
        None
    """
    model_soc = _resolve_model_soc()
    det_model_url = (
        "https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
        f"rdk_{model_soc}/paddle_ocr/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm"
    )
    rec_model_url = (
        "https://archive.d-robotics.cc/downloads/rdk_model_zoo/"
        f"rdk_{model_soc}/paddle_ocr/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm"
    )

    parser = argparse.ArgumentParser(
        description="PaddleOCR text detection and recognition demo (PP-OCRv6).")

    parser.add_argument('--det-model-path', type=str,
                        default=f'/opt/hobot/model/{model_soc}/basic/PP-OCRv6_det_infer-deploy_640x640_nv12.hbm',
                        help='Path to BPU quantized detection model (*.hbm).')
    parser.add_argument('--rec-model-path', type=str,
                        default=f'/opt/hobot/model/{model_soc}/basic/PP-OCRv6_rec_infer-deploy_48x320_rgb.hbm',
                        help='Path to BPU quantized recognition model (*.hbm).')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model scheduling priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../test_data/gt_2322.jpg',
                        help='Path to the test input image.')
    parser.add_argument('--label-file', type=str, default='../../test_data/ppocrv6_dict.txt',
                        help='Path to the character vocabulary file (one token per line).')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save the final result image.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold for the text detection mask (0.0–1.0).')
    parser.add_argument('--ratio-prime', type=float, default=2.7,
                        help='Contour expansion ratio for bounding box dilation.')

    opt = parser.parse_args()

    # Download models if missing
    file_io.download_model_if_needed(opt.det_model_path, det_model_url)
    file_io.download_model_if_needed(opt.rec_model_path, rec_model_url)

    # Build character list: read label file line by line, prepend blank token.
    # PaddleOCR vocab layout is [blank, *dict, ' ']; append trailing space to
    # match the model output class count (e.g. 18710 = 1 blank + 18708 dict + 1 space).
    with open(opt.label_file, 'r', encoding='utf-8') as f:
        char_list = ['blank'] + [line.rstrip('\n') for line in f]
    char_list.append(' ')

    # --- Detection stage ---------------------------------------------------
    det_config = PaddleOCRDetConfig(
        model_path=opt.det_model_path,
        ratio_prime=opt.ratio_prime,
        threshold=opt.threshold,
    )
    det = PaddleOCRDet(det_config)
    det.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    inspect.print_model_info(det.model)

    # Load input image
    img = file_io.load_image(opt.test_img)

    # Run detection pipeline
    input_tensor = det.pre_process(img)
    det_outputs = det.forward(input_tensor)
    img_h, img_w = img.shape[:2]
    img_boxes, cropped_images, boxes_list = det.post_process(det_outputs, img, img_w, img_h)

    # --- Recognition stage -------------------------------------------------
    rec_config = PaddleOCRRecConfig(model_path=opt.rec_model_path)
    rec = PaddleOCRRec(rec_config)
    rec.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    inspect.print_model_info(rec.model)

    recognized_texts = []
    for i, crop in enumerate(cropped_images):
        rec_input = rec.pre_process(crop)
        rec_outputs = rec.forward(rec_input)
        text = rec.post_process(rec_outputs, char_list)
        recognized_texts.append(text)
        print(f"[{i}] Prediction: {text}")

    # --- Visualization -----------------------------------------------------
    font_path = "../../test_data/FangSong.ttf"

    white_canvas = np.ones_like(img) * 255
    img_with_text = vis_utils.draw_text(white_canvas, recognized_texts, boxes_list,
                                        font_path, font_size=35, color=(0, 0, 255))

    combined = np.hstack((img_boxes, img_with_text))
    cv2.imwrite(opt.img_save_path, combined)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
