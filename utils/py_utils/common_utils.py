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

# flake8: noqa: E501

import cv2
import numpy as np

# List of predefined RGB color tuples used for bounding box visualization.
rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def load_image(img_path: str) -> np.ndarray:
    """
    @brief Load an image from file path using OpenCV.
    @param img_path Path to the image file.
    @return Image as a NumPy ndarray in BGR format.
    @throws FileNotFoundError if the image cannot be loaded.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    return img


def load_class_names(path: str) -> list:
    """
    @brief Load class names from a file.
    @param path Path to the label file, each line contains a class name.
    @return List of class name strings.
    """
    with open(path, 'r') as f:
        # Strip whitespace and filter out empty lines
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def zscore_normalize_lastdim(x: np.ndarray) -> np.ndarray:
    """
    @brief Normalize input array along the last dimension.
    @details This function performs standard score normalization (z-score).
    @param x Input NumPy array of shape (..., channels).
    @return Normalized array with mean 0 and variance 1 per vector.
    """
    mean = np.mean(x, axis=-1, keepdims=True)        # Compute mean per sample
    var = np.var(x, axis=-1, keepdims=True)          # Compute variance per sample
    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))  # Z-score normalization


def print_model_info(models: object) -> None:
    """Print detailed information about input and \
        output tensors of all models in the system."""

    # 1. Model Name List
    print("=== Model Name List ===")
    model_names = models.model_names
    print(model_names)

    # 2. Total Number of Models
    print("\n=== Model Count ===")
    print(models.model_count)

    # 3. Input Count per Model
    print("\n=== Input Counts ===")
    input_counts = models.input_counts
    for model, count in input_counts.items():
        print(f"{model}: {count}")

    # 4. Input Names per Model
    print("\n=== Input Names ===")
    input_names = models.input_names
    for model, inputs in input_names.items():
        print(f"{model}:")
        for name in inputs:
            print(f"  - {name}")

    # 5. Input Tensor Shapes
    print("\n=== Input Tensor Shapes ===")
    input_shapes = models.input_shapes
    for model, inputs in input_shapes.items():
        print(f"{model}:")
        for name, shape in inputs.items():
            print(f"  {name} -> shape: {shape}")

    # 6. Input Tensor Data Types
    print("\n=== Input Tensor Types ===")
    input_types = models.input_dtypes
    for model, inputs in input_types.items():
        print(f"{model}:")
        for name, dtype in inputs.items():
            print(f"  {name} -> dtype: {dtype.name}")

    # 7. Input Quantization Information
    print("\n=== Input Quantization Info ===")
    input_quanti_info = models.input_quants
    for model, inputs in input_quanti_info.items():
        print(f"{model}:")
        for name, info in inputs.items():
            print(f"  {name}:")
            print(f"    quanti_type: {info.quant_type.name}")
            print(f"    quantize_axis: {info.axis}")
            print(f"    scale_data: {info.scale.tolist()}")
            print(f"    zero_point_data: {info.zero_point.tolist()}")

    # 8. Input Tensor Stride
    print("\n=== Input Tensor Stride ===")
    input_strides = models.input_strides
    for model, inputs in input_strides.items():
        print(f"{model}:")
        for name, stride in inputs.items():
            print(f"  {name} -> stride: {stride}")

    # 9. Input Descriptions
    input_descs = models.input_descs
    for model, inputs in input_descs.items():
        for name, desc in inputs.items():
            print(f"[Input] {model}.{name} desc: {desc}")

    print("\n================ OUTPUT TESTS ================\n")

    # 1. Output Count per Model
    print("=== Output Counts ===")
    output_counts = models.output_counts
    for model, count in output_counts.items():
        print(f"{model}: {count}")

    # 2. Output Names per Model
    print("\n=== Output Names ===")
    output_names = models.output_names
    for model, outputs in output_names.items():
        print(f"{model}:")
        for name in outputs:
            print(f"  - {name}")

    # 3. Output Tensor Shapes
    print("\n=== Output Tensor Shapes ===")
    output_shapes = models.output_shapes
    for model, outputs in output_shapes.items():
        print(f"{model}:")
        for name, shape in outputs.items():
            print(f"  {name} -> shape: {shape}")

    # 4. Output Tensor Data Types
    print("\n=== Output Tensor Types ===")
    output_types = models.output_dtypes
    for model, outputs in output_types.items():
        print(f"{model}:")
        for name, dtype in outputs.items():
            print(f"  {name} -> dtype: {dtype.name}")

    # 5. Output Quantization Information
    print("\n=== Output Quantization Info ===")
    output_quanti = models.output_quants
    for model, outputs in output_quanti.items():
        print(f"{model}:")
        for name, info in outputs.items():
            print(f"  {name}:")
            print(f"    quanti_type: {info.quant_type.name}")
            print(f"    quantize_axis: {info.axis}")
            print(f"    scale_data: {info.scale}")
            print(f"    zero_point_data: {info.zero_point}")

    # 6. Output Tensor Stride
    print("\n=== Output Tensor Stride ===")
    output_stride = models.output_strides
    for model, outputs in output_stride.items():
        print(f"{model}:")
        for name, stride in outputs.items():
            print(f"  {name} -> stride: {stride}")

    # 7. Output Descriptions
    output_descs = models.output_descs
    for model, outputs in output_descs.items():
        for name, desc in outputs.items():
            print(f"[Output] {model}.{name} desc: {desc}")

    # # Get and Print Model Description Info
    # print("\nModel Description:")
    # model_desc = models.model_descs
    # for model_name, desc in model_desc.items():
    #     print(f" - {model_name}: {desc}")

    # # Get and Print HBM Description Info
    # print("\nHBM Description:")
    # hbm_desc = models.hbm_descs
    # for file_name, desc in hbm_desc.items():
    #     print(f" - {file_name}: {desc}")

        # Get and PrintScheduling Params
    print("\n=== Scheduling Parameters ===")
    sched_params = models.sched_params
    for model_name, sched in sched_params.items():
        print(f"{model_name}:")
        print(f"  priority    : {sched.priority}")
        print(f"  customId    : {sched.customId}")
        print(f"  bpu_cores   : {sched.bpu_cores}")
        print(f"  deviceId    : {sched.deviceId}")
