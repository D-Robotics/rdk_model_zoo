#!/usr/bin/env python3
# Copyright (c) 2026 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""Emit the PTQ/compile configs for the final per-target contract.

Eight configs under ``build/<target>/<part>/config.yaml``:

* X5 (bayes-e, ``hb_mapper makertbin``): PartA = f0-only opset-11 graph with
  the NV12 pyramid input node carrying mean/scale; PartB = batch-150 graph
  (fc6 as conv) with a float featuremap RoI input quantized on-chip with the
  recorded calibration threshold.
* S100/S100P/S600 (nash-e/nash-m/nash-p, ``hb_compile``): PartA = f0-only
  opset-16 graph, split NV12 input; PartB = packed ROI-bank graph with a
  float32 DDR input quantized on-chip (HzCalibration).

Run inside the matching compile container (paths only; the content is the
verified configuration — see README.md for the toolchain pins).
"""
from __future__ import annotations

import argparse
from pathlib import Path

S_TARGETS = {"s100": "nash-e", "s100p": "nash-m", "s600": "nash-p"}

PARTA_X5 = """model_parameters:
  onnx_model: {onnx}/part_a_f0only.op11.onnx
  march: bayes-e
  working_dir: {build}/x5/parta
  output_model_file_prefix: frcnn_mbv3_320_parta
input_parameters:
  input_name: images
  input_type_rt: nv12
  input_type_train: rgb
  input_layout_train: NCHW
  input_shape: 1x3x320x320
  norm_type: data_mean_and_scale
  mean_value: 123.675 116.28 103.53
  scale_value: 0.0171247538317 0.0175070028011 0.0174291938998
  input_space_and_range: bt601_video
calibration_parameters:
  cal_data_dir: {calib}/parta_x5
  cal_data_type: float32
  calibration_type: default
compiler_parameters:
  optimize_level: O3
  compile_mode: latency
  core_num: 1
  input_source:
    images: pyramid
"""

PARTB_X5 = """model_parameters:
  onnx_model: {onnx}/part_b_batch150.op11.onnx
  march: bayes-e
  working_dir: {build}/x5/partb
  output_model_file_prefix: frcnn_mbv3_320_partb
input_parameters:
  input_name: pooled
  input_type_rt: featuremap
  input_type_train: featuremap
  input_layout_train: NCHW
  input_layout_rt: NCHW
  input_shape: 150x256x7x7
  norm_type: no_preprocess
calibration_parameters:
  cal_data_dir: {calib}/partb
  cal_data_type: float32
  calibration_type: default
compiler_parameters:
  optimize_level: O3
  compile_mode: latency
  core_num: 1
  input_source:
    pooled: ddr
"""

PARTA_S = """model_parameters:
  onnx_model: {onnx}/part_a_f0only.op16.onnx
  march: {march}
  working_dir: {build}/{target}/parta
  output_model_file_prefix: frcnn_mbv3_320_parta
  debug_mode: ''
  output_nodes: ''
  remove_node_name: ''
  remove_node_type: ''
input_parameters:
  input_name: images
  input_type_rt: nv12
  input_type_train: rgb
  input_layout_train: NCHW
  input_shape: 1x3x320x320
  norm_type: data_mean_and_scale
  mean_value: 123.675 116.28 103.53
  scale_value: 0.0171247538317 0.0175070028011 0.0174291938998
  std_value: ''
  separate_batch: false
  input_space_and_range: bt601_video
calibration_parameters:
  cal_data_dir: {calib}/parta_s
  cal_data_type: float32
  calibration_type: default
  quant_config: {{}}
compiler_parameters:
  optimize_level: O2
  compile_mode: latency
  core_num: 1
  jobs: 8
  advice: 0
  balance_factor: 0
  max_time_per_fc: 0
  cache_mode: disable
  cache_path: ''
"""

PARTB_S = """model_parameters:
  onnx_model: {onnx}/part_b_packed.op16.onnx
  march: {march}
  working_dir: {build}/{target}/partb
  output_model_file_prefix: frcnn_mbv3_320_partb
  debug_mode: ''
  output_nodes: ''
  remove_node_name: ''
  remove_node_type: ''
input_parameters:
  input_name: bank
  input_type_rt: featuremap
  input_type_train: featuremap
  input_layout_train: NCHW
  input_shape: 1x256x1050x7
  norm_type: no_preprocess
  std_value: ''
  separate_batch: false
calibration_parameters:
  cal_data_dir: {calib}/partb_packed_s
  cal_data_type: float32
  calibration_type: default
  quant_config: {{}}
compiler_parameters:
  optimize_level: O2
  compile_mode: latency
  core_num: 1
  jobs: 8
  advice: 0
  balance_factor: 0
  max_time_per_fc: 0
  cache_mode: disable
  cache_path: ''
  extra_params: {{'input_no_padding': True, 'output_no_padding': True}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", default="onnx", help="export_split.py output directory")
    parser.add_argument("--calib", default="calib", help="prepare_calibration.py output directory")
    parser.add_argument("--build", default="build", help="config/working directory root")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    fmt = {"onnx": (here / args.onnx).resolve(),
           "calib": (here / args.calib).resolve(),
           "build": (here / args.build).resolve()}

    paths: list[Path] = []
    parts = [("x5", "parta", PARTA_X5), ("x5", "partb", PARTB_X5)]
    for target, march in S_TARGETS.items():
        parts.append((target, "parta", PARTA_S.replace("{march}", march).replace("{target}", target)))
        parts.append((target, "partb", PARTB_S.replace("{march}", march).replace("{target}", target)))
    for target, part, template in parts:
        config = fmt["build"] / target / part / "config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(template.format(**fmt))
        paths.append(config)
        print(f"wrote {config}")


if __name__ == "__main__":
    main()
