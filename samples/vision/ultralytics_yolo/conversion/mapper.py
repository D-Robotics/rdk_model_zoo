#!/user/bin/env python3

# Copyright (c) 2025, WuChao D-Robotics.
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

# 注意: 此程序在OpenExplore的Docker或者py环境中运行
# Attention: This program runs on OpenExplore Docker or python Environment.

import os
import argparse
import logging 
import subprocess
import shutil

# 获取脚本所在目录和当前工作目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.getcwd()

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

# 日志模块配置
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("MZOO")

def resolve_path(path, base_dir=None):
    """解析路径，支持相对路径和绝对路径"""
    if os.path.isabs(path):
        return path
    
    if base_dir is None:
        base_dir = WORK_DIR
    
    return os.path.abspath(os.path.join(base_dir, path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal-images', type=str, default='./cal_images', help='*.jpg, *.png calibration images path, 20 ~ 50 pictures is OK.') 
    parser.add_argument('--onnx', type=str, default='./yolo11n.onnx', help='origin float onnx model path.')
    parser.add_argument('--output-dir', type=str, default='.', help='output directory for converted model.')
    # default below
    parser.add_argument('--quantized', type=str, default="int8", help='int8 first / int16 first')
    parser.add_argument('--jobs', type=int, default=16, help='model combine jobs.')
    parser.add_argument('--optimize-level', type=str, default='O3', help='O0, O1, O2, O3')
    parser.add_argument('--cal-sample', type=bool, default=True, help='sample calibration data or not.') 
    parser.add_argument('--cal-sample-num', type=int, default=20, help='num of sample calibration data.') 
    parser.add_argument('--save-cache', type=bool, default=False, help='remove bpu output files or not.') 
    # private settings
    parser.add_argument('--cal', type=str, default='.calibration_data_temporary_folder', help='calibration_data_temporary_folder')
    parser.add_argument('--ws', type=str, default='.temporary_workspace', help='temporary workspace')
    opt = parser.parse_args()
    
    # 首先打印原始参数
    logger.info(opt)
    
    # 解析所有路径为绝对路径
    opt.onnx = resolve_path(opt.onnx)
    opt.cal_images = resolve_path(opt.cal_images)
    
    # 如果输出目录是默认值（当前目录），则设置为ONNX文件同级目录
    if opt.output_dir == '.':
        opt.output_dir = os.path.dirname(opt.onnx)
        logger.info(f"Output directory set to ONNX file directory: {opt.output_dir}")
    else:
        opt.output_dir = resolve_path(opt.output_dir)
    
    opt.ws = resolve_path(opt.ws)
    
    logger.info(f"Resolved paths:")
    logger.info(f"  ONNX model: {opt.onnx}")
    logger.info(f"  Calibration images: {opt.cal_images}")
    logger.info(f"  Output directory: {opt.output_dir}")
    logger.info(f"  Workspace: {opt.ws}")

    # check hb_mapper
    try:
        subprocess.run(['hb_mapper', '--version'], capture_output=True, text=True, check=True)
        logger.info("hb_mapper is available and working.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("hb_mapper is not available.")
        exit(1)

    # check onnx file
    session = None
    width = 640
    height = 640
    try:
        logger.info(f"Loading ONNX model from: {opt.onnx}")
        if not os.path.exists(opt.onnx):
            logger.error(f"ONNX file not found: {opt.onnx}")
            exit(1)
            
        session = ort.InferenceSession(opt.onnx, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        # single input check
        if len(inputs) != 1:
            logger.error(f"Error: Model has {len(inputs)} inputs, expected exactly 1.")
            exit(1)
        logger.debug("Model has a single input.")
        
        input_tensor = inputs[0]
        input_shape = input_tensor.shape
        input_type = input_tensor.type
        
        # float32 input check
        if input_type != 'tensor(float)':
            logger.error(f"Error: Input type is {input_type}, expected 'tensor(float)' (fp32).")
            exit(1)
        logger.debug("Input data type is float32 (tensor(float)).")
        
        # NCHW input check
        if len(input_shape) != 4:
            logger.error(f"Error: Input shape has {len(input_shape)} dimensions, expected 4 (NCHW).")
            exit(1)
        logger.debug("NCHW check success.")

        # get input_h, input_w
        height = input_shape[2]
        width = input_shape[3]
        assert isinstance(height, int), "input height dtype error."
        assert isinstance(width, int), "input width dtype error"

    except FileNotFoundError:
        logger.error(f"Error: Model file not found at '{opt.onnx}'.")
        exit(1)
    except Exception as e:
        logger.error(f"Error analyzing ONNX model: {e}")
        exit(1)
    finally:
        if session is not None:
            del session
            logger.debug("ONNX Runtime session released.")

    # check cal-images folder
    if not os.path.exists(opt.cal_images):
        logger.error(f"cal-images folder: '{opt.cal_images}' does not exist, please check!")
        exit(1)
    if len(os.listdir(opt.cal_images)) == 0:
        logger.error(f"cal-images folder: '{opt.cal_images}' is empty, please check!")
        exit(1)
    
    # check cal-images file
    img_cnt = 0
    img_names = []
    for name in os.listdir(opt.cal_images):
        if name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_cnt += 1
            img_names.append(name)
        else:
            logger.warning(f"cal-images folder: '{opt.cal_images}' contains non-image files, skipping: {name}")
    
    if img_cnt == 0:
        logger.error(f"cal-images folder: '{opt.cal_images}' contains no valid images, please check!")
        exit(1)
        
    if img_cnt > opt.cal_sample_num and opt.cal_sample:
        sampled_indices = np.random.choice(len(img_names), size=opt.cal_sample_num, replace=False)
        img_names = [img_names[i] for i in sampled_indices]
        logger.info(f"Sampling enabled. Sampled {opt.cal_sample_num} images from {img_cnt} total images.")
    
    img_cnt = len(img_names)
    if img_cnt < 20:
        logger.warning(f"There are {img_cnt} ( < 20 ) images in the calibration dataset, which may cause the calibration to fail.")
    if img_cnt > 50:
        logger.warning(f"There are {img_cnt} ( > 50 ) images in the calibration dataset, which may cost a long time to calibrate.")
    
    # workspace folder check and setup
    if os.path.exists(opt.ws) and os.path.isdir(opt.ws):
        logger.info(f"Folder '{opt.ws}' exists, removing...")
        try:
            shutil.rmtree(opt.ws)
            logger.info(f"Folder '{opt.ws}' removed successfully.")
        except Exception as e:
            logger.error(f"Remove folder '{opt.ws}' error: {e}")
            exit(1)
    
    try:
        cal_data_dir = os.path.join(opt.ws, opt.cal)
        os.makedirs(cal_data_dir, exist_ok=True)
        logger.info(f"Workspace '{opt.ws}' created successfully.")
    except Exception as e:
        logger.error(f"Create folder '{opt.ws}' error: {e}")
        exit(1)

    # ensure output directory exists
    os.makedirs(opt.output_dir, exist_ok=True)

    # 获取模型文件名用于生成输出文件名
    model_name = os.path.splitext(os.path.basename(opt.onnx))[0]
    output_model_prefix = f"{model_name}_bayese_{width}x{height}_nv12"
    
    # int16
    int16_config_str = ",set_all_nodes_int16" if opt.quantized == "int16" else ""

    # 使用绝对路径生成配置文件
    cal_data_dir = os.path.join(opt.ws, opt.cal)
    bpu_output_dir = os.path.join(opt.ws, 'bpu_model_output')
    
    yaml_content = f'''model_parameters:
  onnx_model: '{opt.onnx}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{bpu_output_dir}'
  output_model_file_prefix: '{output_model_prefix}'
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
  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_config_str}
compiler_parameters:
  jobs: {opt.jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{opt.optimize_level}'
'''

    # 在workspace中创建配置文件
    config_path = os.path.join(opt.ws, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write(yaml_content)
    logger.info(f"Configuration file created: {config_path}")

    # prepare calibration data
    logger.info("Preparing calibration data...")
    for img_name in img_names:
        img_path = os.path.join(opt.cal_images, img_name)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue
            
        # 此处的前处理以ONNX的前处理为基础，总的来说是和训练时的前处理保持一致
        # 如果yaml中有配置mean和scale, 则此处无须计算mean和scale.
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR2RGB
        input_tensor = cv2.resize(input_tensor, (width, height)) # resize
        input_tensor = np.transpose(input_tensor, (2, 0, 1))    # HWC2CHW
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # CHW -> NCHW
        dst_path = os.path.join(cal_data_dir, img_name + '.rgbchw')
        input_tensor.tofile(dst_path)
    logger.info("Calibration data has been successfully generated.")

    # 切换到workspace目录执行转换
    original_cwd = os.getcwd()
    try:
        os.chdir(opt.ws)
        logger.info(f"Changed working directory to: {opt.ws}")
        
        # mapper conversion
        cmd = f"hb_mapper makertbin --config config.yaml --model-type onnx"
        logger.info(f"Executing model conversion...")
        logger.info(f"Command: {cmd}")
        result = os.system(cmd)
        if result != 0:
            logger.error("Model conversion failed!")
            exit(1)
        logger.info("Model conversion completed successfully!")

        # 移动输出文件到指定目录
        output_bin_path = os.path.join(bpu_output_dir, f"{output_model_prefix}.bin")
        final_output_path = os.path.join(opt.output_dir, f"{output_model_prefix}.bin")
        
        logger.info(f"Looking for output file: {output_bin_path}")
        if os.path.exists(output_bin_path):
            shutil.move(output_bin_path, final_output_path)
            logger.info(f"Output file moved to: {final_output_path}")
        else:
            logger.error(f"Output file not found: {output_bin_path}")
            # 列出实际生成的文件
            if os.path.exists(bpu_output_dir):
                actual_files = os.listdir(bpu_output_dir)
                logger.error(f"Files found in output directory: {actual_files}")
            exit(1)

        # 移动hb_mapper日志文件到输出目录
        mapper_log_source = os.path.join(opt.ws, "hb_mapper_makertbin.log")
        mapper_log_dest = os.path.join(opt.output_dir, "hb_mapper_makertbin.log")
        
        if os.path.exists(mapper_log_source):
            shutil.move(mapper_log_source, mapper_log_dest)
            logger.info(f"hb_mapper log moved to: {mapper_log_dest}")
        else:
            logger.warning(f"hb_mapper log not found at: {mapper_log_source}")
            
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to: {original_cwd}")

    # clean the work space
    logger.info("Cleaning up...")
    if not opt.save_cache:
        if os.path.exists(opt.ws):
            shutil.rmtree(opt.ws)
            logger.info("Temporary files cleaned up.")
    else:
        logger.info(f"Cache files preserved in: {opt.ws}")
        logger.info(f"Note: hb_mapper log is available at: {mapper_log_dest}")
    
    logger.info(f"Conversion completed successfully!")
    logger.info(f"Output file: {final_output_path}")
    logger.info(f"Mapper log: {mapper_log_dest}")

if __name__ == "__main__":
    main()