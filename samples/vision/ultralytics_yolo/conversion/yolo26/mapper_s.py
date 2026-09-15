# Copyright (c) 2025 D-Robotics Corporation
# SPDX-License-Identifier: Apache-2.0
"""
YOLO26 Model Conversion Script (Mapper)

This script automates the model conversion process for D-Robotics RDK platforms.
It supports RDK S100/S100P/S600 (Nash architecture).

Key Features:
- Automates calibration data preparation (Image -> NPY).
- Generates architecture-specific configuration files (yaml).
- Invokes the appropriate compiler tool:
    - `hb_compile` for Nash (S100/S100P/S600) -> Output: *.hbm

Usage:
    python3 mapper.py --onnx model.onnx --cal-images ./images --march nash-e
"""
import os
import argparse
import logging
import subprocess
import shutil
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.getcwd()
logging.basicConfig(level=logging.DEBUG, format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('MZOO')

def resolve_path(path, base_dir=None):
    """Resolve path to absolute path, handling relative paths correctly."""
    if os.path.isabs(path):
        return path
    if base_dir is None:
        base_dir = WORK_DIR
    return os.path.abspath(os.path.join(base_dir, path))

def main():
    parser = argparse.ArgumentParser(description='D-Robotics Model Mapper for YOLO26')
    parser.add_argument('--cal-images', type=str, default='./cal_images', help='Directory containing calibration images (jpg/png). 20-50 images recommended.')
    parser.add_argument('--onnx', type=str, default='./yolo11n.onnx', help='Path to the source float ONNX model.')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save the converted model.')
    parser.add_argument('--march', type=str, default='nash-e', help='Target Architecture: "nash-e" (RDK S100), "nash-m" (RDK S100P), "nash-p" (RDK S600). Default: nash-e')
    parser.add_argument('--quantized', type=str, default='int8', help='Quantization precision: "int8" (default) or "int16".')
    parser.add_argument('--jobs', type=int, default=16, help='Number of parallel compilation jobs.')
    parser.add_argument('--optimize-level', type=str, default='O2', help='Optimization level. Nash: O0-O2.')
    parser.add_argument('--cal-sample', type=bool, default=True, help='Enable random sampling of calibration images.')
    parser.add_argument('--cal-sample-num', type=int, default=20, help='Number of images to sample for calibration.')
    parser.add_argument('--save-cache', type=bool, default=False, help='Keep temporary intermediate files (workspace).')
    parser.add_argument('--cal', type=str, default='.calibration_data_temporary_folder', help='Internal temp folder for calibration data.')
    parser.add_argument('--ws', type=str, default='.temporary_workspace', help='Internal temporary workspace.')
    opt = parser.parse_args()
    global cv2, np, ort
    import cv2
    import numpy as np
    import onnxruntime as ort
    logger.info(f'Arguments: {opt}')
    if 'nash' in opt.march:
        if opt.optimize_level == 'O3':
            logger.error(f"Error: Optimization level 'O3' is not supported for {opt.march} architecture. Please use O0, O1, or O2.")
            exit(-1)
        logger.info(f'Detected Nash architecture ({opt.march}). Using run_nash workflow.')
        run_nash(opt)
    else:
        logger.error(f"Error: Unsupported march type '{opt.march}'. Supported: nash-e, nash-m, nash-p.")
        exit(-1)

def run_nash(opt):
    """
    Workflow for RDK S100/S100P/S600 (Nash Architecture).
    Uses `hb_compile` toolchain.
    Generates `.hbm` model files.
    Calibration data format: `.npy` files (normalized 0-1).
    """
    opt.onnx = resolve_path(opt.onnx)
    opt.cal_images = resolve_path(opt.cal_images)
    if opt.output_dir == '.':
        opt.output_dir = os.path.dirname(opt.onnx)
    else:
        opt.output_dir = resolve_path(opt.output_dir)
    opt.ws = resolve_path(opt.ws)
    logger.info(f'Resolved paths:')
    logger.info(f'  ONNX model: {opt.onnx}')
    logger.info(f'  Calibration images: {opt.cal_images}')
    logger.info(f'  Output directory: {opt.output_dir}')
    try:
        subprocess.run(['hb_compile', '--help'], capture_output=True, text=True, check=True)
        logger.info('hb_compile is available.')
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error('hb_compile is not available. Please ensure RDK S100/S600 Toolchain is installed/sourced.')
        exit(1)
    session = None
    width = 640
    height = 640
    try:
        logger.info(f'Validating ONNX model: {opt.onnx}')
        if not os.path.exists(opt.onnx):
            logger.error(f'ONNX file not found: {opt.onnx}')
            exit(1)
        session = ort.InferenceSession(opt.onnx, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            logger.error('Error: Model must have exactly 1 input.')
            exit(1)
        input_tensor = inputs[0]
        input_shape = input_tensor.shape
        if len(input_shape) != 4:
            logger.error('Error: Input must be NCHW.')
            exit(1)
        height = input_shape[2]
        width = input_shape[3]
        logger.info(f'Model Input Size: {width}x{height}')
    except Exception as e:
        logger.error(f'Error analyzing ONNX model: {e}')
        exit(1)
    finally:
        if session is not None:
            del session
    if not os.path.exists(opt.cal_images) or not os.listdir(opt.cal_images):
        logger.error(f"Calibration folder '{opt.cal_images}' invalid.")
        exit(1)
    img_names = [n for n in os.listdir(opt.cal_images) if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not img_names:
        logger.error('No valid images found.')
        exit(1)
    if len(img_names) > opt.cal_sample_num and opt.cal_sample:
        sampled_indices = np.random.choice(len(img_names), size=opt.cal_sample_num, replace=False)
        img_names = [img_names[i] for i in sampled_indices]
        logger.info(f'Sampled {opt.cal_sample_num} images.')
    if os.path.exists(opt.ws) and os.path.isdir(opt.ws):
        shutil.rmtree(opt.ws)
    try:
        cal_data_dir = os.path.join(opt.ws, opt.cal)
        os.makedirs(cal_data_dir, exist_ok=True)
    except Exception as e:
        logger.error(f'Workspace creation failed: {e}')
        exit(1)
    os.makedirs(opt.output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(opt.onnx))[0]
    march_str = opt.march.replace('-', '')
    output_model_prefix = f'{model_name}_{march_str}_{width}x{height}_nv12'
    bpu_output_dir = os.path.join(opt.ws, 'bpu_model_output')
    quant_config_part = ''
    if opt.quantized == 'int16':
        quant_config_part = '\n  quant_config: {\n        "model_config": {\n            "all_node_type": "int16",\n            "model_output_type": "int16",\n        }\n    }'
    yaml_content = f"""model_parameters:\n  onnx_model: '{opt.onnx}'\n  march: "{opt.march}"\n  layer_out_dump: False\n  working_dir: '{bpu_output_dir}'\n  output_model_file_prefix: '{output_model_prefix}'\ninput_parameters:\n  input_name: ""\n  input_type_rt: 'nv12'\n  input_type_train: 'rgb'\n  input_layout_train: 'NCHW'\n  norm_type: 'data_scale'\n  scale_value: 0.003921568627451\ncalibration_parameters:\n  cal_data_dir: '{cal_data_dir}'\n  cal_data_type: 'float32'\n  calibration_type: 'default'{quant_config_part}\ncompiler_parameters:\n  extra_params: {{'input_no_padding': True, 'output_no_padding': True}}  \n  jobs: {opt.jobs}\n  compile_mode: 'latency'\n  debug: true\n  optimize_level: '{opt.optimize_level}'\n"""
    config_path = os.path.join(opt.ws, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as file:
        file.write(yaml_content)
    logger.info(f'Generated config: {config_path}')
    logger.info('Processing calibration data (normalizing to 0-1)...')
    for img_name in img_names:
        img_path = os.path.join(opt.cal_images, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (width, height))
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32) / 255.0
        dst_path = os.path.join(cal_data_dir, img_name + '.npy')
        np.save(dst_path, input_tensor)
    original_cwd = os.getcwd()
    try:
        os.chdir(opt.ws)
        cmd = 'hb_compile --config config.yaml'
        logger.info(f'Running: {cmd}')
        if os.system(cmd) != 0:
            logger.error('hb_compile failed.')
            exit(1)
        output_bin_path = os.path.join(bpu_output_dir, f'{output_model_prefix}.hbm')
        final_output_path = os.path.join(opt.output_dir, f'{output_model_prefix}.hbm')
        if os.path.exists(output_bin_path):
            shutil.move(output_bin_path, final_output_path)
            logger.info(f'Success! Model saved to: {final_output_path}')
        else:
            logger.error(f'Output .hbm file not found at {output_bin_path}')
            exit(1)
        log_src = os.path.join(opt.ws, 'hb_compile.log')
        log_dst = os.path.join(opt.output_dir, 'hb_compile.log')
        if os.path.exists(log_src):
            shutil.move(log_src, log_dst)
    finally:
        os.chdir(original_cwd)
    if not opt.save_cache:
        shutil.rmtree(opt.ws)
        logger.info('Workspace cleaned.')
if __name__ == '__main__':
    main()
