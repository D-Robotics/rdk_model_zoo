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

"""
ConvNeXt Inference Entry Script.

This script demonstrates the standard BPU inference pipeline for ConvNeXt
on a single input image, following the RDK Model Zoo engineering standards.

Workflow:
    1) Parse CLI arguments for model, data, and parameters.
    2) Initialize ConvNeXtConfig and ConvNeXt model wrapper.
    3) Configure runtime scheduling (BPU cores, priority).
    4) Load data and execute full pipeline: Preprocess -> Forward -> Postprocess.
    5) Visualize and save the resulting image with Top-K predictions.

Notes:
    - The project root is appended to sys.path to import shared utilities under
      `utils/py_utils/`.
    - This script is designed to be executed from the sample directory.

Example:
    python main.py \
        --model-path ../../model/ConvNeXt_atto_224x224_nv12.bin \
        --test-img ../../test_data/cheetah.JPEG \
        --topk 5
"""

import os
import cv2
import sys
import argparse
from typing import List, Dict, Optional

# Add project root to sys.path so we can import utility modules.
sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.file_io as file_io
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from convnext import ConvNeXtConfig, ConvNeXt


def main() -> None:
    """Run ConvNeXt inference on a single image.
    
    This function orchestrates the complete inference process:
    - Argument parsing
    - Model initialization
    - Image loading and label loading
    - Inference execution
    - Result visualization and saving
    """
    
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="ConvNeXt Classification Inference")

    # Model configuration
    parser.add_argument('--model-path', type=str, 
                        default="../../model/ConvNeXt_atto_224x224_nv12.bin", 
                        help="Path to the BPU quantized *.bin model.")
    parser.add_argument('--label-file', type=str, 
                        default="../../../../../datasets/imagenet/imagenet_classes.names", 
                        help="Path to the ImageNet class names file.")
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255).')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help="BPU core indexes to run inference.")
    
    # Test data and results
    parser.add_argument('--test-img', type=str, 
                        default="../../test_data/cheetah.JPEG", 
                        help="Path to the test input image.")
    parser.add_argument('--img-save-path', type=str, default='result.jpg',
                        help='Path to save output result image.')
    
    # Inference parameters
    parser.add_argument('--resize-type', type=int, default=1, 
                        help="Resize strategy (0: direct, 1: letterbox).")
    parser.add_argument('--topk', type=int, default=5, 
                        help="Number of top results to return.")

    args = parser.parse_args()

    # 2. Initialize configuration and model
    # Note: ConvNeXt internally handles label loading from label-file if provided in classes_path
    config = ConvNeXtConfig(
        model_path=args.model_path,
        classes_path=args.label_file,
        resize_type=args.resize_type,
        topk=args.topk
    )
    model = ConvNeXt(config)

    # 3. Configure runtime scheduling (BPU cores, priority)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)

    # 4. Print basic model info
    inspect.print_model_info(model.model)

    # 5. Load data
    img = file_io.load_image(args.test_img)
    if img is None:
        print(f"[Error] Failed to load image: {args.test_img}")
        return

    # 6. Execute the inference pipeline
    # results returns (topk_idx, topk_prob, topk_labels)
    topk_idx, topk_prob, topk_labels = model.predict(img)

    # 7. Visualization and Export
    print(f"\n[Info] Inference completed. Top-{args.topk} results:")
    for i in range(len(topk_idx)):
        cls_name = topk_labels[i] if topk_labels else f"ID {topk_idx[i]}"
        print(f"  [{i+1}] {cls_name:30} | Prob: {topk_prob[i]:.4f}")

    # Draw results on image using standard visualization tool
    # Need to convert topk_labels to List[str] if it isn't already
    results = list(zip(topk_idx.tolist(), topk_prob.tolist()))
    image_res = visualize.draw_classification(img, results, topk_labels if topk_labels else {})
    cv2.imwrite(args.img_save_path, image_res)
    print(f"\n[Info] Saving results to {args.img_save_path}")


if __name__ == "__main__":
    main()
