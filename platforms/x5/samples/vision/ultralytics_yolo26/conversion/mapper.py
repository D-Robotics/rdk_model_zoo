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

# Pre-import checks and dynamic installs
try:
    import cv2
except ImportError:
    os.system('pip install opencv-python')
    import cv2

try:
    import numpy as np
except ImportError:
    os.system('pip install numpy')
    import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    os.system('pip install onnxruntime')
    import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("YOLO26_Mapper")

# Global path tracking
WORK_DIR = os.getcwd()


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
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
    parser = argparse.ArgumentParser(description="Automated ONNX to BPU Model Conversion")
    
    # Required/Common Args
    parser.add_argument('--onnx', type=str, required=True, 
                        help='Path to the source float ONNX model.')
    parser.add_argument('--cal-images', type=str, default='./cal_images', 
                        help='Path to calibration images (20~50 recommended).') 
    parser.add_argument('--output-dir', type=str, default='.', 
                        help='Target directory for the converted .bin model.')
    
    # Optimization & Quantization Args
    parser.add_argument('--quantized', type=str, default="int8", choices=["int8", "int16"],
                        help='Quantization precision level.')
    parser.add_argument('--jobs', type=int, default=16, 
                        help='Number of parallel jobs for model combination.')
    parser.add_argument('--optimize-level', type=str, default='O3', choices=['O0', 'O1', 'O2', 'O3'],
                        help='Compiler optimization level.')
    
    # Calibration & Cache Control
    parser.add_argument('--cal-sample', type=bool, default=True, 
                        help='Whether to sample from the calibration image pool.') 
    parser.add_argument('--cal-sample-num', type=int, default=20, 
                        help='Number of images to sample for calibration.') 
    parser.add_argument('--save-cache', action='store_true', 
                        help='Preserve temporary workspace and logs after conversion.') 
    
    # Advanced / Internal Paths
    parser.add_argument('--ws', type=str, default='.temporary_workspace', 
                        help='Directory for temporary conversion artifacts.')
    
    opt = parser.parse_args()
    
    # Path Resolution
    opt.onnx = resolve_path(opt.onnx)
    opt.cal_images = resolve_path(opt.cal_images)
    if opt.output_dir == '.':
        opt.output_dir = os.path.dirname(opt.onnx)
    else:
        opt.output_dir = resolve_path(opt.output_dir)
    opt.ws = resolve_path(opt.ws)
    
    logger.info(f"Starting conversion for: {opt.onnx}")

    # 1. Environment Check (hb_mapper)
    try:
        subprocess.run(['hb_mapper', '--version'], capture_output=True, text=True, check=True)
        logger.info("hb_mapper tool is verified.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("hb_mapper is not available. Please run inside OpenExplore Docker.")
        exit(1)

    # 2. ONNX Model Analysis
    try:
        if not os.path.exists(opt.onnx):
            logger.error(f"ONNX file not found: {opt.onnx}")
            exit(1)
            
        session = ort.InferenceSession(opt.onnx, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        
        if len(inputs) != 1:
            logger.error(f"Error: Model has {len(inputs)} inputs, expected 1.")
            exit(1)
        
        input_tensor = inputs[0]
        input_shape = input_tensor.shape
        input_type = input_tensor.type
        
        if input_type != 'tensor(float)':
            logger.error(f"Error: Input type {input_type} is not float32.")
            exit(1)
        
        if len(input_shape) != 4:
            logger.error(f"Error: Input shape {input_shape} is not NCHW.")
            exit(1)

        height, width = input_shape[2], input_shape[3]
        logger.info(f"Model Input Resolution: {width}x{height}")
        
        # Clean up session
        del session
    except Exception as e:
        logger.error(f"Failed to analyze ONNX model: {e}")
        exit(1)

    # 3. Calibration Data Verification
    if not os.path.exists(opt.cal_images) or not os.listdir(opt.cal_images):
        logger.error(f"Invalid calibration image path: {opt.cal_images}")
        exit(1)
    
    img_names = [n for n in os.listdir(opt.cal_images) if n.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not img_names:
        logger.error("No valid images found in calibration directory.")
        exit(1)
        
    if len(img_names) > opt.cal_sample_num and opt.cal_sample:
        img_names = list(np.random.choice(img_names, size=opt.cal_sample_num, replace=False))
    
    # 4. Workspace Setup
    if os.path.exists(opt.ws):
        shutil.rmtree(opt.ws)
    
    cal_data_dir = os.path.join(opt.ws, 'calibration_data')
    os.makedirs(cal_data_dir, exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)

    # 5. Configuration (YAML) Generation
    model_base_name = os.path.splitext(os.path.basename(opt.onnx))[0]
    output_prefix = f"{model_base_name}_bayese_{width}x{height}_nv12"
    int16_opt = ",set_all_nodes_int16" if opt.quantized == "int16" else ""
    bpu_output_dir = os.path.join(opt.ws, 'bpu_model_output')
    
    yaml_content = f'''model_parameters:
  onnx_model: '{opt.onnx}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{bpu_output_dir}'
  output_model_file_prefix: '{output_prefix}'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{cal_data_dir}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_opt}
compiler_parameters:
  jobs: {opt.jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{opt.optimize_level}'
'''
    with open(os.path.join(opt.ws, "config.yaml"), "w") as f:
        f.write(yaml_content)

    # 6. Prepare Binary Calibration Blobs
    logger.info("Generating binary calibration data...")
    for name in img_names:
        img = cv2.imread(os.path.join(opt.cal_images, name))
        if img is None: continue
        # Preprocessing matching model expectation
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (width, height))
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
        input_tensor.tofile(os.path.join(cal_data_dir, f"{name}.rgbchw"))

    # 7. Execute Conversion
    orig_cwd = os.getcwd()
    try:
        os.chdir(opt.ws)
        cmd = "hb_mapper makertbin --config config.yaml --model-type onnx"
        logger.info(f"Running: {cmd}")
        if os.system(cmd) != 0:
            logger.error("Model conversion failed.")
            exit(1)

        # Move artifacts
        bin_src = os.path.join(bpu_output_dir, f"{output_prefix}.bin")
        bin_dst = os.path.join(opt.output_dir, f"{output_prefix}.bin")
        if os.path.exists(bin_src):
            shutil.move(bin_src, bin_dst)
            logger.info(f"BPU Model saved to: {bin_dst}")
        
        log_src = os.path.join(opt.ws, "hb_mapper_makertbin.log")
        if os.path.exists(log_src):
            shutil.move(log_src, os.path.join(opt.output_dir, "hb_mapper_makertbin.log"))
    finally:
        os.chdir(orig_cwd)

    # 8. Cleanup
    if not opt.save_cache:
        shutil.rmtree(opt.ws)
        logger.info("Cleaned up temporary workspace.")
    
    logger.info("Conversion Workflow Completed.")


if __name__ == "__main__":
    main()
