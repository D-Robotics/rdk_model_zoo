#!/usr/bin/env python3

# Copyright (c) 2026 D-Robotics Corporation
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

"""DINOv2 Model Conversion Script (Mapper)

This script automates the model conversion process for D-Robotics RDK
S100/S100P/S600 (Nash architecture).

Key Features:
- Exports the FAIR DINOv2 ViT-S/14 checkpoint to a BPU-friendly ONNX graph.
- Automates calibration data preparation (Image -> NPY, ImageNet-normalized).
- Generates the hb_compile configuration with the validated recipe:
  featuremap float32 input + all-int16 + default (KL) calibration.
- Invokes `hb_compile` and copies the .hbm artifact to the output directory.

The quantization recipe is fixed on purpose: int8 activations and
max-percentile calibration were measured to fail the 0.99 cosine bar on the
raw self-supervised DINOv2 backbone (see conversion/README.md for the full
measured matrix).

Usage:
    python3 mapper.py \
        --weights ./dinov2_vits14_pretrain.pth \
        --repo ./dinov2 \
        --cal-images ./cal_images \
        --march nash-e \
        --output-dir ./output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("dinov2_mapper")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DINOV2_SOURCE_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHTS_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHTS_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
RESIZE_SIZE = 256
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The measured recipe: featuremap input, all-int16, KL (default) calibration.
# calibration_type is intentionally NOT written into the yaml: hmct's default
# "modelwise search" calibration is what makes int16 pass on DINOv2.
QUANT_CONFIG = """
  quant_config:
    {
        "model_config": {
            "all_node_type": "int16",
            "model_output_type": "int16",
        }
    }"""


def parse_args() -> argparse.Namespace:
    """Parse mapper command-line arguments.

    Returns:
        argparse.Namespace: Conversion options supplied by the user.
    """

    parser = argparse.ArgumentParser(description="D-Robotics Model Mapper for DINOv2")
    parser.add_argument("--weights", type=str, default="./dinov2_vits14_pretrain.pth", help="Path to dinov2_vits14_pretrain.pth (or its URL).")
    parser.add_argument("--weights-sha256", type=str, default=WEIGHTS_SHA256, help="Expected SHA-256 for the checkpoint.")
    parser.add_argument("--repo", type=str, default="./dinov2", help="Path to a local clone of facebookresearch/dinov2.")
    parser.add_argument("--repo-revision", type=str, default=DINOV2_SOURCE_REVISION, help="Required git revision for the DINOv2 source clone.")
    parser.add_argument("--cal-images", type=str, default="./cal_images", help="Directory containing calibration images (jpg/png). 50 diverse real images recommended.")
    parser.add_argument("--march", type=str, default="nash-e", choices=["nash-e", "nash-m", "nash-p"], help="Target Nash march.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save the converted model.")
    parser.add_argument("--jobs", type=int, default=16, help="Number of parallel compilation jobs.")
    parser.add_argument("--save-cache", action="store_true", help="Keep temporary intermediate files (workspace).")
    return parser.parse_args()


def check_toolchain() -> None:
    """Verify that hb_compile is available in the current environment.

    Raises:
        SystemExit: If hb_compile cannot be executed.
    """

    try:
        subprocess.run(["hb_compile", "--help"], capture_output=True, text=True, check=True)
        logger.info("hb_compile is available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("hb_compile is not available. Please run inside the RDK S100/S600 OE docker image.")
        sys.exit(1)


def export_onnx(opt: argparse.Namespace, ws: str) -> str:
    """Run the ONNX export step and return the exported model path.

    Args:
        opt: Parsed conversion options.
        ws: Temporary conversion workspace.

    Returns:
        str: Path to the generated ONNX model.
    """

    onnx_path = os.path.join(ws, "dinov2_vits14_224.onnx")
    script = os.path.join(SCRIPT_DIR, "onnx_export", "export_dinov2.py")
    cmd = [
        sys.executable, script,
        "--weights", opt.weights,
        "--weights-sha256", opt.weights_sha256,
        "--repo", opt.repo,
        "--repo-revision", opt.repo_revision,
        "--out", onnx_path,
    ]
    logger.info("Exporting ONNX: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return onnx_path


def resize_short_side_and_center_crop(image: np.ndarray) -> np.ndarray:
    """Resize an RGB image and crop it to DINOv2 evaluation geometry.

    Args:
        image: RGB image in HWC layout with a non-zero height and width.

    Returns:
        np.ndarray: Bicubic-resized, centered 224 by 224 RGB image.
    """

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Calibration image must have non-zero dimensions.")
    if height < width:
        resized_height = RESIZE_SIZE
        resized_width = int(width * RESIZE_SIZE / height)
    else:
        resized_width = RESIZE_SIZE
        resized_height = int(height * RESIZE_SIZE / width)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
    y0 = (resized_height - IMAGE_SIZE) // 2
    x0 = (resized_width - IMAGE_SIZE) // 2
    return resized[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE]


def prepare_calibration(opt: argparse.Namespace, ws: str) -> str:
    """Convert calibration images to preprocessed float32 NPY tensors.

    The tensors match the runtime input contract: BGR to RGB, bicubic
    short-side resize to 256, center crop to 224, [0, 1] scaling, ImageNet
    mean/std normalization, and NCHW layout.

    Args:
        opt: Parsed conversion options containing the calibration directory.
        ws: Temporary conversion workspace.

    Returns:
        str: Directory containing preprocessed float32 NPY tensors.

    Raises:
        SystemExit: If the calibration directory has no usable images.
    """

    if not os.path.isdir(opt.cal_images) or not os.listdir(opt.cal_images):
        logger.error("Calibration folder '%s' invalid.", opt.cal_images)
        sys.exit(1)

    names = sorted(
        n for n in os.listdir(opt.cal_images)
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not names:
        logger.error("No valid images found in '%s'.", opt.cal_images)
        sys.exit(1)
    logger.info("Using %d calibration images.", len(names))

    cal_dir = os.path.join(ws, "calibration_data_norm")
    os.makedirs(cal_dir, exist_ok=True)
    manifest = []
    for idx, name in enumerate(names):
        img = cv2.imread(os.path.join(opt.cal_images, name), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Skipping undecodable calibration image: %s", name)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_short_side_and_center_crop(img)
        tensor = np.transpose(img, (2, 0, 1))[None, :, :, :].astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
        np.save(os.path.join(cal_dir, f"{idx:04d}.npy"), np.ascontiguousarray(tensor))
        with open(os.path.join(opt.cal_images, name), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        manifest.append({"file": name, "npy": f"{idx:04d}.npy", "sha256": digest})

    if not manifest:
        logger.error("No decodable calibration images found in '%s'.", opt.cal_images)
        sys.exit(1)
    logger.info("Prepared %d calibration tensors.", len(manifest))

    with open(os.path.join(ws, "calibration_manifest.json"), "w") as fh:
        json.dump({"count": len(manifest), "images": manifest}, fh, indent=1)
    return cal_dir


def generate_yaml(opt: argparse.Namespace, onnx_path: str, cal_dir: str, ws: str) -> str:
    """Generate the hb_compile configuration with the validated recipe.

    Args:
        opt: Parsed conversion options.
        onnx_path: Absolute path to the exported ONNX model.
        cal_dir: Directory containing calibration tensors.
        ws: Temporary conversion workspace.

    Returns:
        str: Path to the generated compiler configuration.
    """

    march_str = opt.march.replace("-", "")
    prefix = f"dinov2_vits14_224_int16_{march_str}"
    # All paths in the yaml are absolute: hb_compile resolves them against its
    # own working directory, not against the yaml location.
    working_dir = os.path.join(os.path.abspath(ws), f"out_{march_str}")

    yaml_content = f'''model_parameters:
  onnx_model: '{onnx_path}'
  march: "{opt.march}"
  layer_out_dump: False
  working_dir: '{working_dir}'
  output_model_file_prefix: '{prefix}'
input_parameters:
  input_name: "input"
  input_type_rt: 'featuremap'
  input_type_train: 'featuremap'
  input_layout_train: 'NCHW'
calibration_parameters:
  cal_data_dir: '{cal_dir}'
  cal_data_type: 'float32'
{QUANT_CONFIG}
compiler_parameters:
  extra_params: {{'input_no_padding': True, 'output_no_padding': True}}
  compile_mode: 'latency'
  core_num: 1
  debug: False
  jobs: {opt.jobs}
  optimize_level: 'O2'
  advice: 1
'''
    yaml_path = os.path.join(ws, "config.yaml")
    with open(yaml_path, "w") as fh:
        fh.write(yaml_content)
    logger.info("Generated %s", yaml_path)
    return yaml_path


def run_compile(yaml_path: str) -> None:
    """Invoke hb_compile with the generated configuration.

    Args:
        yaml_path: Path to the hb_compile configuration file.

    Raises:
        SystemExit: If compilation fails.
    """

    logger.info("Running hb_compile ...")
    result = subprocess.run(["hb_compile", "--config", os.path.abspath(yaml_path)])
    if result.returncode != 0:
        logger.error("hb_compile failed.")
        sys.exit(result.returncode)


def collect_artifacts(opt: argparse.Namespace, ws: str) -> None:
    """Copy the .hbm artifact and the compile log to the output directory.

    Args:
        opt: Parsed conversion options.
        ws: Temporary conversion workspace.

    Raises:
        SystemExit: If the expected compiled HBM is absent.
    """

    march_str = opt.march.replace("-", "")
    working_dir = os.path.join(os.path.abspath(ws), f"out_{march_str}")
    hbm = os.path.join(working_dir, f"dinov2_vits14_224_int16_{march_str}.hbm")
    if not os.path.isfile(hbm):
        logger.error("Expected hbm not found: %s", hbm)
        sys.exit(1)
    dst = os.path.join(opt.output_dir, f"dinov2_vits14_224_int16_{march_str}.hbm")
    shutil.copy2(hbm, dst)
    log_src = os.path.join(working_dir, "hb_compile.log")
    if os.path.isfile(log_src):
        shutil.copy2(log_src, os.path.join(opt.output_dir, f"hb_compile_{march_str}.log"))
    logger.info("Saved %s", dst)


def main() -> None:
    """Run the complete DINOv2 conversion workflow.

    Raises:
        SystemExit: If a conversion step cannot complete.
    """
    opt = parse_args()
    check_toolchain()

    ws = os.path.abspath(os.path.join(opt.output_dir, ".temporary_workspace"))
    if os.path.isdir(ws):
        shutil.rmtree(ws)
    os.makedirs(ws, exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)

    onnx_path = export_onnx(opt, ws)
    cal_dir = prepare_calibration(opt, ws)
    yaml_path = generate_yaml(opt, os.path.abspath(onnx_path), os.path.abspath(cal_dir), ws)
    run_compile(yaml_path)
    collect_artifacts(opt, ws)

    if not opt.save_cache:
        shutil.rmtree(ws, ignore_errors=True)
    logger.info("Done.")


if __name__ == "__main__":
    main()
