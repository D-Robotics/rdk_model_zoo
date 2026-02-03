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

"""YOLOv5X image inference entry script.

This script runs a BPU-quantized YOLOv5X (.hbm) model on a single input image.

Workflow:
    1) Parse CLI arguments.
    2) Download the model file if missing (based on SoC type).
    3) Create YOLOv5Config and initialize YoloV5X runtime wrapper.
    4) Preprocess image -> BPU inference -> postprocess detections.
    5) Draw bounding boxes and save the result image.

Notes:
    - The project root is appended to sys.path to import shared utilities under
      `utils/py_utils/`.
    - This script is designed to be executed from the sample directory so that
      relative paths (e.g., `../../../../../`) resolve correctly.

Example:
    python main.py \
        --test-img ../../../../../datasets/coco/assets/kite.jpg \
        --img-save-path result.jpg \
        --score-thres 0.25 \
        --nms-thres 0.45
"""
import os
import cv2
import sys
import argparse

# Add project root to sys.path so we can import utility modules.
# Source files:
#   utils/py_utils/preprocess_utils.py
#   utils/py_utils/postprocess_utils.py
# (Check these files for preprocessing/postprocessing implementations.)
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yolov5 import YoloV5X, YOLOv5Config


def main() -> None:
    """Run YOLOv5X object detection on a single image.

    This function parses command-line arguments, loads the YOLOv5X model,
    preprocesses the input image, performs inference on the BPU,
    postprocesses detection results, and saves the output image with
    bounding boxes.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'../../model/yolov5x_672x672_nv12.hbm',
                        help="""Path to BPU Quantized *.hbm Model.""")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.")
    parser.add_argument('--test-img', type=str, default='../../../../../datasets/coco/assets/kite.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default='../../../../../datasets/coco/coco_classes.names',
                        help='Path to load COCO label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with detection results.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold for filtering detections.')

    opt = parser.parse_args()

    # Download model if missing (manual step, URL should be provided)
    if soc == "s600":
        download_url = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s600/ultralytics_YOLO/yolov5x_672x672_nv12.hbm"
    else:
        download_url = "https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_s100/ultralytics_YOLO/yolov5x_672x672_nv12.hbm"

    file_io.download_model_if_needed(opt.model_path, download_url)

    # Init config parameter
    config = YOLOv5Config(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres
    )

    # Instantiate YOLOv5X model
    yolov5x = YoloV5X(config)

    # Configure runtime scheduling (BPU cores, priority)
    yolov5x.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(yolov5x.model)

    # Load label names (e.g., COCO class names)
    coco_names = file_io.load_class_names(opt.label_file)

    # Load input image
    img = file_io.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Preprocess image to match model input
    input_array = yolov5x.pre_process(img)
    # input_array = yolov5x.pre_process(img, )

    # Run inference
    outputs = yolov5x.forward(input_array)

    # Postprocess outputs to get boxes, class IDs, scores
    boxes, scores, cls_ids = yolov5x.post_process(outputs, img_w, img_h)

    # boxes, cls_ids, scores = yolov5x(img)

    # visualize detection results on image
    image = visualize.draw_boxes(img, boxes, cls_ids, scores, coco_names, visualize.rdk_colors)

    # Save the resulting image
    cv2.imwrite(opt.img_save_path, image)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
