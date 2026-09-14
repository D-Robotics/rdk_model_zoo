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

"""PointNet point cloud segmentation sample entry point."""

import argparse
import os
import sys
from typing import Dict

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../../../../../"))
import utils.py_utils.inspect as inspect
from pointnet import PointNet, PointNetConfig


CHAIR_PARTS = ["back", "seat", "leg", "arm"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the PointNet sample."""
    parser = argparse.ArgumentParser(
        description="Run PointNet point cloud part segmentation."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../../model/s100/pointnet.hbm",
        help="Path to the BPU quantized HBM model.",
    )
    parser.add_argument(
        "--test-pts",
        type=str,
        default="../../test_data/chair.pts",
        help="Path to the input point cloud in .pts format.",
    )
    parser.add_argument(
        "--img-save-path",
        type=str,
        default="result.png",
        help="Path to save the segmented point cloud visualization.",
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


def create_point_cloud_axes(point_set: np.ndarray):
    """Create a 3D matplotlib axis for point cloud visualization."""
    fig = plt.figure(dpi=192, figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    x_axis = point_set[:, 0]
    y_axis = point_set[:, 2]
    z_axis = point_set[:, 1]

    max_range = (
        np.array(
            [
                x_axis.max() - x_axis.min(),
                y_axis.max() - y_axis.min(),
                z_axis.max() - z_axis.min(),
            ]
        ).max()
        * 0.5
    )
    mid_x = (x_axis.max() + x_axis.min()) * 0.5
    mid_y = (y_axis.max() + y_axis.min()) * 0.5
    mid_z = (z_axis.max() + z_axis.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tick_params(labelsize=5)
    ax.set_xlabel("X", fontsize=10)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_zlabel("Z", fontsize=10)
    return ax


def summarize_parts(pred_labels: np.ndarray) -> Dict[str, int]:
    """Count predicted points for each chair part."""
    return {
        name: int(np.sum(pred_labels == idx))
        for idx, name in enumerate(CHAIR_PARTS)
    }


def save_original_view(point_set: np.ndarray, output_path: str) -> None:
    """Save the normalized input point cloud visualization."""
    x_axis = point_set[:, 0]
    y_axis = point_set[:, 2]
    z_axis = point_set[:, 1]
    ax = create_point_cloud_axes(point_set)
    ax.scatter3D(x_axis, y_axis, z_axis, s=5, cmap="jet", marker="o", label="chair")
    ax.set_title("3D Point Cloud")
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig(output_path, bbox_inches="tight", dpi=192)
    plt.close()


def save_segmentation_view(
    point_set: np.ndarray,
    pred_labels: np.ndarray,
    output_path: str,
) -> None:
    """Save the predicted point cloud segmentation visualization."""
    x_axis = point_set[:, 0]
    y_axis = point_set[:, 2]
    z_axis = point_set[:, 1]
    ax = create_point_cloud_axes(point_set)
    for idx, name in enumerate(CHAIR_PARTS):
        mask = pred_labels == idx
        ax.scatter(
            x_axis[mask],
            y_axis[mask],
            z_axis[mask],
            s=5,
            cmap="jet",
            marker="o",
            label=name,
        )
    ax.set_title("3D Segmentation Result")
    plt.legend(loc="upper right", fontsize=8)
    plt.savefig(output_path, bbox_inches="tight", dpi=192)
    plt.close()


def main() -> None:
    """Run PointNet part segmentation on one chair point cloud."""
    args = parse_args()

    config = PointNetConfig(model_path=args.model_path)
    model = PointNet(config)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    inspect.print_model_info(model.model)

    points = PointNet.load_point_cloud(args.test_pts)
    original_path = os.path.splitext(args.img_save_path)[0] + "_orig.png"
    save_original_view(points, original_path)

    pred_labels = model.predict(points)
    save_segmentation_view(points, pred_labels, args.img_save_path)

    print("Part point counts:")
    for part_name, count in summarize_parts(pred_labels).items():
        print(f"  {part_name}: {count}")
    print(f"[Saved] Original point cloud: {original_path}")
    print(f"[Saved] Segmentation result: {args.img_save_path}")


if __name__ == "__main__":
    main()
