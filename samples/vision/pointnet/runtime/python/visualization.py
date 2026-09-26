# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Source PointNet plot helpers; imported only for explicit CLI visualization."""
from typing import Dict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
