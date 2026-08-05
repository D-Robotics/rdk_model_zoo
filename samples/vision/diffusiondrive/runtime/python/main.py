# Copyright (c) 2026 D-Robotics Corporation
# Licensed under the Apache License, Version 2.0.

"""Run the DiffusionDrive S600 trajectory-planning sample."""

from __future__ import annotations

import argparse
import os

import numpy as np

from diffusiondrive import DiffusionDrive, DiffusionDriveConfig, INPUT_NAMES, render_result


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
DEFAULT_MODEL_PATH = os.path.join(SAMPLE_DIR, "model", "s600", "diffusiondrive_r34_256x1024_s600.hbm")
DEFAULT_INPUT_PATH = os.path.join(SAMPLE_DIR, "test_data", "reference_inputs.npz")
DEFAULT_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "diffusiondrive_outputs.npz")
DEFAULT_IMAGE_PATH = os.path.join(SCRIPT_DIR, "diffusiondrive_result.png")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the DiffusionDrive sample.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="DiffusionDrive trajectory planning on RDK S600")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the S600 DiffusionDrive HBM model.")
    parser.add_argument("--input-npz", type=str, default=DEFAULT_INPUT_PATH, help="Path to four float32 NAVSIM input tensors in NPZ format.")
    parser.add_argument("--output-npz", type=str, default=DEFAULT_OUTPUT_PATH, help="Path used to save decoded output tensors.")
    parser.add_argument(
        "--img-save-path",
        "--output-image",
        dest="img_save_path",
        type=str,
        default=DEFAULT_IMAGE_PATH,
        help="Path used to save the inference visualization.",
    )
    parser.add_argument("--agent-score-thres", type=float, default=0.5, help="Sigmoid threshold for predicted agents.")
    parser.add_argument("--priority", type=int, default=0, help="Model scheduling priority.")
    parser.add_argument("--bpu-cores", nargs="+", type=int, default=[0], help="BPU core indexes used by hbm_runtime.")
    return parser.parse_args()


def main() -> None:
    """Load test tensors, execute BPU inference, and save decoded results.

    Returns:
        None.
    """

    args = parse_args()
    with np.load(args.input_npz, allow_pickle=False) as archive:
        features = {name: np.asarray(archive[name], dtype=np.float32) for name in INPUT_NAMES}

    model = DiffusionDrive(DiffusionDriveConfig(args.model_path, args.agent_score_thres))
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    result = model.predict(features)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
    np.savez(args.output_npz, **result)
    os.makedirs(os.path.dirname(os.path.abspath(args.img_save_path)), exist_ok=True)
    render_result(features, result, args.img_save_path)

    print("Model:", model.model_name)
    print("Trajectory [x, y, heading]:")
    print(result["trajectory"][0])
    print("Predicted agents:", int(result["agent_mask"].sum()))
    values, counts = np.unique(result["bev_labels"], return_counts=True)
    print("BEV class pixels:", dict(zip(values.tolist(), counts.tolist())))
    print("[Saved] Outputs:", args.output_npz)
    print("[Saved] Visualization:", args.img_save_path)


if __name__ == "__main__":
    main()
