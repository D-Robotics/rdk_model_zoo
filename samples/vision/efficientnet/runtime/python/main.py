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

import os
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
from efficientnet import EfficientNet, EfficientNetConfig


def main() -> None:
    """
    @brief Run EfficientNet classification on a single image.

    This function parses command-line arguments, loads the EfficientNet model,
    preprocesses the image, performs inference, postprocesses the results,
    and prints Top-K results.

    @return None
    """
    soc = inspect.get_soc_name().lower()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-path', type=str,
                        default=f'../../model/efficientnet_lite0_224x224_nv12.hbm',
                        help='Path to BPU Quantized *.hbm Model.')
    parser.add_argument('--priority', type=int, default=0,
                        help='Model priority (0~255). 0 is lowest, 255 is highest.')
    parser.add_argument('--bpu-cores', nargs='+', type=int, default=[0],
                        help='List of BPU core indexes to run inference, e.g., --bpu-cores 0 1.')
    parser.add_argument('--test-img', type=str, default='../../../../../datasets/imagenet/asset/scottish_deerhound.JPEG',
                        help='Path to load test image.')
    parser.add_argument('--label-file', type=str, default='../../../../../datasets/imagenet/imagenet_classes.names',
                        help='Path to load label file.')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top predictions to display.')
    parser.add_argument('--resize-type', type=int, default=1,
                        help='Resize strategy: 0 (direct), 1 (letterbox).')

    opt = parser.parse_args()

    # Download model if missing
    download_url = f"https://archive.d-robotics.cc/downloads/rdk_model_zoo/rdk_{soc}/EfficientNet/efficientnet_lite0_224x224_nv12.hbm"

    file_io.download_model_if_needed(opt.model_path, download_url)

    # Init config parameter
    config = EfficientNetConfig(
        model_path=opt.model_path,
        topk=opt.topk,
        resize_type=opt.resize_type
    )

    # Instantiate EfficientNet model
    efficientnet = EfficientNet(config)

    # Configure runtime scheduling (BPU cores, priority)
    efficientnet.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)

    # Print basic model info
    inspect.print_model_info(efficientnet.model)

    # Load label names
    labels = file_io.load_labels(opt.label_file)

    # Load input image
    img = file_io.load_image(opt.test_img)

    # Run prediction pipeline
    probs, indices = efficientnet.predict(img)

    # Print Results
    print(f"\nTop-{opt.topk} Results:")
    for i in range(len(indices)):
        idx = indices[i]
        score = probs[i]
        label_name = labels.get(idx, f"Class {idx}")
        print(f"  {label_name}: {score:.4f}")


if __name__ == "__main__":
    main()