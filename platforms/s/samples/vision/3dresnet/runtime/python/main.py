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

"""3D ResNet-18 video action classification sample entry point."""

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
import utils.py_utils.visualize as visualize
from resnet3d import ResNet3D, ResNet3DConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the 3DResNet sample."""
    parser = argparse.ArgumentParser(
        description="Run 3D ResNet-18 video action classification."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../../model/s100/r3d_18.hbm",
        help="Path to the BPU quantized HBM model.",
    )
    parser.add_argument(
        "--test-clip",
        type=str,
        default="../../test_data/video0.npy",
        help="Path to the preprocessed video clip in .npy format.",
    )
    parser.add_argument(
        "--label-file",
        type=str,
        default="../../test_data/kinetics_classnames.json",
        help="Path to the Kinetics-400 class name mapping JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to print.",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Model scheduling priority in the range 0 to 255.",
    )
    parser.add_argument(
        "--bpu-cores",
        nargs="+",
        type=int,
        default=[0],
        help="BPU core indexes used by hbm_runtime.",
    )
    return parser.parse_args()


def load_labels(label_file: str) -> Dict[int, str]:
    """Load the Kinetics class-name mapping used by the original demo."""
    with open(label_file, "r", encoding="utf-8") as f:
        kinetics_classnames = json.load(f)
    return {
        int(class_id): str(name).replace('"', "")
        for name, class_id in kinetics_classnames.items()
    }


def main() -> None:
    """Run video action classification on a pre-extracted clip."""
    args = parse_args()

    config = ResNet3DConfig(model_path=args.model_path)
    model = ResNet3D(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    inspect.print_model_info(model.model)

    clip = np.load(args.test_clip)
    idx2label = load_labels(args.label_file)
    results = model.predict(clip, top_k=args.top_k)
    visualize.print_classification_results(results, idx2label, topk=args.top_k)


if __name__ == "__main__":
    main()
