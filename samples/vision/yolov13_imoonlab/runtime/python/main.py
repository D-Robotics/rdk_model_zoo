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

"""YOLOv13 single-image inference entry."""

import os
import cv2
import sys
import argparse

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from yolov13 import YOLOv13Config, YoloV13


def main() -> None:
    """Parse arguments, run detection, and save the rendered result."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default='../../model/s100/yolo13n_detect_nashe_640x640_nv12.hbm',
                        help='Path to BPU Quantized *.hbm Model.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../test_data/kite.jpg',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default='../../test_data/coco_classes.names',
                        help='Path to load COCO label file.')
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output image with detection results.')
    parser.add_argument('--nms-thres', type=float, default=0.45,
                        help='IoU threshold for Non-Maximum Suppression.')
    parser.add_argument('--score-thres', type=float, default=0.25,
                        help='Confidence score threshold for filtering detections.')

    opt = parser.parse_args()

    config = YOLOv13Config(
        model_path=opt.model_path,
        score_thres=opt.score_thres,
        nms_thres=opt.nms_thres
    )
    yolov13 = YoloV13(config)
    yolov13.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    inspect.print_model_info(yolov13.model)
    coco_names = file_io.load_class_names(opt.label_file)
    img = file_io.load_image(opt.test_img)
    boxes, scores, cls_ids = yolov13.predict(img)
    image = visualize.draw_boxes(
        img, boxes, cls_ids, scores, coco_names, visualize.rdk_colors)
    cv2.imwrite(opt.img_save_path, image)
    print(f"[Saved] Result saved to: {opt.img_save_path}")


if __name__ == "__main__":
    main()
