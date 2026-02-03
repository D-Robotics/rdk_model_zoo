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

"""
inspect: Runtime and system inspection utilities.

This module provides helper functions for inspecting runtime and system-level
information at execution time. It is primarily used to query platform metadata
and to introspect loaded models, exposing detailed information for debugging,
verification, and demonstration purposes.

Key Features:
    - Query basic system or platform identification information.
    - Inspect runtime-managed models and print detailed metadata.
    - Assist developers in understanding model inputs, outputs, and scheduling
      configurations.

Typical Usage:
    >>> from inspect import get_soc_name
    >>> soc = get_soc_name()

    >>> from inspect import print_model_info
    >>> print_model_info(models)

Notes:
    - This module is intended for inspection, debugging, and informational
      purposes only, and should not be relied on for performance-critical logic.
    - The output format is human-readable and may change as inspection
      requirements evolve.
"""


def get_soc_name() -> str:
    """Get the SoC (System-on-Chip) name of the current device.

    The SoC name is read from the system file
    `/sys/class/boardinfo/soc_name`. If the file cannot be read, a default
    value is returned.

    Returns:
        The SoC name as a string. Returns `"s100"` if reading the system
        information fails.
    """
    soc_path = "/sys/class/boardinfo/soc_name"
    try:
        # Attempt to read the SoC name from the system information file
        with open(soc_path, "r") as f:
            # Strip trailing whitespace and return the content
            return f.read().strip()
    except Exception:
        # Return a default SoC name when the system file cannot be read
        return "s100"


def print_model_info(models: object) -> None:

    """Print detailed input and output tensor information for all models.

    This utility function prints comprehensive metadata for each model
    managed by the runtime, including:

    - Model names and total model count
    - Input tensor information:
        - Names, shapes, data types
        - Quantization parameters
        - Strides and descriptions
    - Output tensor information:
        - Names, shapes, data types
        - Quantization parameters
        - Strides and descriptions
    - Model-level and HBM-level descriptions
    - Runtime scheduling parameters (priority, BPU cores, etc.)

    Args:
        models: Runtime model container object that provides model metadata
            such as inputs, outputs, quantization info, and scheduling
            parameters.

    Returns:
        None
    """
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

    # Get and Print Model Description Info
    print("\nModel Description:")
    model_desc = models.model_descs
    for model_name, desc in model_desc.items():
        print(f" - {model_name}: {desc}")

    # Get and Print HBM Description Info
    print("\nHBM Description:")
    hbm_desc = models.hbm_descs
    for file_name, desc in hbm_desc.items():
        print(f" - {file_name}: {desc}")

        # Get and PrintScheduling Params
    print("\n=== Scheduling Parameters ===")
    sched_params = models.sched_params
    for model_name, sched in sched_params.items():
        print(f"{model_name}:")
        print(f"  priority    : {sched.priority}")
        print(f"  customId    : {sched.customId}")
        print(f"  bpu_cores   : {sched.bpu_cores}")
        print(f"  deviceId    : {sched.deviceId}")
