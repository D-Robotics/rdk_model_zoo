# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLO26 Model Conversion Tool.

This script automates the process of converting a float ONNX model to a
quantized BPU .bin model using the D-robotics OpenExplore toolchain.

Main Workflow:
1. Validate requirements (hb_mapper, libraries).
2. Analyze ONNX model (extract input name, shape, and type).
3. Prepare calibration data (resize, normalize, and save as binary).
4. Generate hb_mapper configuration (YAML).
5. Execute hb_mapper and manage output artifacts.

Notes:
    - This program must run in an environment where 'hb_mapper' is available
      (e.g., D-robotics OpenExplore Docker).
"""
import os
import argparse
import logging
import subprocess
import shutil
from typing import Optional
logging.basicConfig(level=logging.DEBUG, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('YOLO26_Mapper')
WORK_DIR = os.getcwd()

def resolve_path(path: str, base_dir: Optional[str]=None) -> str:
    """Resolve a path to an absolute path.

    Args:
        path: The path string to resolve.
        base_dir: Optional base directory for relative resolution.

    Returns:
        Absolute path as a string.
    """
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = WORK_DIR
    return os.path.abspath(os.path.join(base_dir, path))

def main() -> None:
    """Main execution block for model conversion."""
    parser = argparse.ArgumentParser(description='Automated ONNX to BPU Model Conversion')
    parser.add_argument('--onnx', type=str, required=True, help='Path to the source float ONNX model.')
    parser.add_argument('--cal-images', type=str, default='./cal_images', help='Path to calibration images (20~50 recommended).')
    parser.add_argument('--output-dir', type=str, default='.', help='Target directory for the converted .bin model.')
    parser.add_argument('--quantized', type=str, default='int8', choices=['int8', 'int16'], help='Quantization precision level.')
    parser.add_argument('--jobs', type=int, default=16, help='Number of parallel jobs for model combination.')
    parser.add_argument('--optimize-level', type=str, default='O3', choices=['O0', 'O1', 'O2', 'O3'], help='Compiler optimization level.')
    parser.add_argument('--cal-sample', type=bool, default=True, help='Whether to sample from the calibration image pool.')
    parser.add_argument('--cal-sample-num', type=int, default=20, help='Number of images to sample for calibration.')
    parser.add_argument('--save-cache', action='store_true', help='Preserve temporary workspace and logs after conversion.')
    parser.add_argument('--ws', type=str, default='.temporary_workspace', help='Directory for temporary conversion artifacts.')
    opt = parser.parse_args()
    global cv2, np, ort
    import cv2
    import numpy as np
    import onnxruntime as ort
    opt.onnx = resolve_path(opt.onnx)
    opt.cal_images = resolve_path(opt.cal_images)
    if opt.output_dir == '.':
        opt.output_dir = os.path.dirname(opt.onnx)
    else:
        opt.output_dir = resolve_path(opt.output_dir)
    opt.ws = resolve_path(opt.ws)
    logger.info(f'Starting conversion for: {opt.onnx}')
    try:
        subprocess.run(['hb_mapper', '--version'], capture_output=True, text=True, check=True)
        logger.info('hb_mapper tool is verified.')
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error('hb_mapper is not available. Please run inside OpenExplore Docker.')
        exit(1)
    try:
        if not os.path.exists(opt.onnx):
            logger.error(f'ONNX file not found: {opt.onnx}')
            exit(1)
        session = ort.InferenceSession(opt.onnx, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            logger.error(f'Error: Model has {len(inputs)} inputs, expected 1.')
            exit(1)
        input_tensor = inputs[0]
        input_shape = input_tensor.shape
        input_type = input_tensor.type
        if input_type != 'tensor(float)':
            logger.error(f'Error: Input type {input_type} is not float32.')
            exit(1)
        if len(input_shape) != 4:
            logger.error(f'Error: Input shape {input_shape} is not NCHW.')
            exit(1)
        height, width = (input_shape[2], input_shape[3])
        logger.info(f'Model Input Resolution: {width}x{height}')
        del session
    except Exception as e:
        logger.error(f'Failed to analyze ONNX model: {e}')
        exit(1)
    if not os.path.exists(opt.cal_images) or not os.listdir(opt.cal_images):
        logger.error(f'Invalid calibration image path: {opt.cal_images}')
        exit(1)
    img_names = [n for n in os.listdir(opt.cal_images) if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not img_names:
        logger.error('No valid images found in calibration directory.')
        exit(1)
    if len(img_names) > opt.cal_sample_num and opt.cal_sample:
        img_names = list(np.random.choice(img_names, size=opt.cal_sample_num, replace=False))
    if os.path.exists(opt.ws):
        shutil.rmtree(opt.ws)
    cal_data_dir = os.path.join(opt.ws, 'calibration_data')
    os.makedirs(cal_data_dir, exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)
    model_base_name = os.path.splitext(os.path.basename(opt.onnx))[0]
    output_prefix = f'{model_base_name}_bayese_{width}x{height}_nv12'
    int16_opt = ',set_all_nodes_int16' if opt.quantized == 'int16' else ''
    bpu_output_dir = os.path.join(opt.ws, 'bpu_model_output')
    yaml_content = f"""model_parameters:\n  onnx_model: '{opt.onnx}'\n  march: "bayes-e"\n  layer_out_dump: False\n  working_dir: '{bpu_output_dir}'\n  output_model_file_prefix: '{output_prefix}'\ninput_parameters:\n  input_name: ""\n  input_type_rt: 'nv12'\n  input_type_train: 'rgb'\n  input_layout_train: 'NCHW'\n  norm_type: 'data_scale'\n  scale_value: 0.003921568627451\ncalibration_parameters:\n  cal_data_dir: '{cal_data_dir}'\n  cal_data_type: 'float32'\n  calibration_type: 'default'\n  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_opt}\ncompiler_parameters:\n  jobs: {opt.jobs}\n  compile_mode: 'latency'\n  debug: true\n  optimize_level: '{opt.optimize_level}'\n"""
    with open(os.path.join(opt.ws, 'config.yaml'), 'w') as f:
        f.write(yaml_content)
    logger.info('Generating binary calibration data...')
    for name in img_names:
        img = cv2.imread(os.path.join(opt.cal_images, name))
        if img is None:
            continue
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (width, height))
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        input_tensor.tofile(os.path.join(cal_data_dir, f'{name}.rgbchw'))
    orig_cwd = os.getcwd()
    try:
        os.chdir(opt.ws)
        cmd = 'hb_mapper makertbin --config config.yaml --model-type onnx'
        logger.info(f'Running: {cmd}')
        if os.system(cmd) != 0:
            logger.error('Model conversion failed.')
            exit(1)
        bin_src = os.path.join(bpu_output_dir, f'{output_prefix}.bin')
        bin_dst = os.path.join(opt.output_dir, f'{output_prefix}.bin')
        if os.path.exists(bin_src):
            shutil.move(bin_src, bin_dst)
            logger.info(f'BPU Model saved to: {bin_dst}')
        log_src = os.path.join(opt.ws, 'hb_mapper_makertbin.log')
        if os.path.exists(log_src):
            shutil.move(log_src, os.path.join(opt.output_dir, 'hb_mapper_makertbin.log'))
    finally:
        os.chdir(orig_cwd)
    if not opt.save_cache:
        shutil.rmtree(opt.ws)
        logger.info('Cleaned up temporary workspace.')
    logger.info('Conversion Workflow Completed.')
if __name__ == '__main__':
    main()
