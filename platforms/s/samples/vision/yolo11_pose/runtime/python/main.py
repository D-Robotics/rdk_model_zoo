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

"""YOLO11-Pose pose estimation image inference entry script.

This script runs a BPU-quantized YOLO11-Pose (.hbm) model on a single input image
and produces detection boxes, keypoints, and a visualization result.

Workflow:
    1) Parse CLI arguments.
    2) Download the model file if missing (based on SoC type).
    3) Create YoloV11PoseConfig and initialize YoloV11Pose runtime wrapper.
    4) Preprocess image -> BPU inference -> postprocess detections and keypoints.
    5) Draw boxes and keypoints, then save the result image.

Notes:
    - The project root is appended to sys.path to import shared utilities under
      `utils/py_utils/`.
    - This script is designed to be executed from the sample directory so that
      relative paths (e.g., `../../../../../`) resolve correctly.

Example:
    python main.py \\
        --test-img ../../test_data/bus.jpg \\
        --img-save-path result.jpg \\
        --score-thres 0.25 \\
        --nms-thres 0.7 \\
        --kpt-conf-thres 0.5
"""

import os
import cv2
import sys
import argparse

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yolo11pose import YoloV11Pose, YoloV11PoseConfig


def main() -> None:
    """Run YOLO11-Pose estimation on a single image.

    This function parses command-line arguments, loads the YOLO11-Pose model,
    preprocesses the input image, performs BPU inference, postprocesses results
    (boxes + keypoints), and saves the visualization image.

    Returns:
        None
    """
    soc = inspect.get_soc_name().lower()
    model_suffix = "nashp" if soc == "s600" else "nashe"

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'/opt/hobot/model/{soc}/basic/yolo11n_pose_{model_suffix}_640x640_nv12.hbm',
                        help='Path to BPU Quantized *.hbm Model.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../test_data/bus.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default='../../test_data/coco_classes.names',
                        help='Path to load COCO label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with detection and keypoint results.')
    parser.add_argument('--nms-thres', type=float, default=0.7,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold for filtering detections.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5,
                        help='Keypoint confidence threshold for visualization.')

    opt = parser.parse_args()

    # Select model download URL based on SoC platform
    download_url = f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_{soc}/ultralytics_YOLO/yolo11n_pose_{model_suffix}_640x640_nv12.hbm"

    file_io.download_model_if_needed(opt.model_path, download_url)

    # Init config
    config = YoloV11PoseConfig(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres,
    )

    # Instantiate model
    yolov11_pose = YoloV11Pose(config)

    # Configure runtime scheduling
    yolov11_pose.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(yolov11_pose.model)

    # Load resources
    coco_names = file_io.load_class_names(opt.label_file)
    img = file_io.load_image(opt.test_img)
    img_h, img_w = img.shape[:2]

    # Preprocess -> Inference -> Postprocess
    input_array = yolov11_pose.pre_process(img)
    outputs = yolov11_pose.forward(input_array)
    boxes, scores, cls_ids, kpts_xy, kpts_score = yolov11_pose.post_process(outputs, img_w, img_h)

    # Visualize: boxes + keypoints
    visualize.draw_boxes(img, boxes, cls_ids, scores, coco_names, visualize.rdk_colors)
    visualize.draw_keypoints(img, kpts_xy, kpts_score, kpt_conf_thresh=opt.kpt_conf_thres)

    # Save result
    cv2.imwrite(opt.img_save_path, img)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
