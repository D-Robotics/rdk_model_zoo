#!/user/bin/env python

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

try:
    import cv2
except ImportError:
    os.system('pip install opencv-python')
    import cv2

try:
    import numpy as np
except:
    os.system('pip install numpy')
    import numpy as np

try:
    import onnxruntime as ort
except:
    os.system('pip install onnxruntime')
    import onnxruntime as ort



# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("MZOO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal-images', type=str, default='./cal_images', help='*.jpg, *.png calibration images path, 20 ~ 50 pictures is OK.') 
    parser.add_argument('--onnx', type=str, default='./yolo11n.onnx', help='origin float onnx model path.')
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
    logger.info(opt)

    # check hb_mapper
    try:
        subprocess.run(['hb_mapper', '--version'], capture_output=True, text=True, check=True)
        logger.info("hb_mapper is available and working.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("hb_mapper is not available.")

    # check onnx file
    session = None
    width = 640
    height = 640
    try:
        logger.info(f"Loading ONNX model from: {opt.onnx}")
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
        logger.debug("NCHW check successs.")

        # get input_h, input_w
        height = input_shape[2]
        width = input_shape[3]
        assert isinstance(height, int), "input height dtype error."
        assert isinstance(width, int), "input width dtype error"

    except FileNotFoundError:
        logger.info(f"Error: Model file not found at '{opt.onnx}'.")
        return None
    except Exception as e:
        logger.info(f"Error analyzing ONNX model: {e}")
        return None
    finally:
        if session is not None:
            del session
            logger.debug("ONNX Runtime session released.")

    # check cal-images folder
    if not os.path.exists(opt.cal_images):
        logger.error(f"cal-images file: \'{opt.cal_images}\' is not exist, please check!")
        exit(1)
    if len(os.listdir(opt.cal_images))==0:
        logger.error(f"cal-images file: \'{opt.cal_images}\' is empty, please check!")
        exit(1)
    
    # check cal-images file
    img_cnt = 0
    img_names = []
    for name in os.listdir(opt.cal_images):
        if name.endswith(".jpg") or name.endswith(".png") or name.endswith(".jpeg") \
            or name.endswith(".JPG") or name.endswith(".PNG") or name.endswith(".JPEG"):
            img_cnt += 1
            img_names.append(name)
        else:
            logger.warning(f"cal-images file: \'{opt.cal_images}\' contains non-image files, skipping!")
    if img_cnt == 0:
        logger.error(f"cal-images file: \'{opt.cal_images}\' is too small, please check!")
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
    
    # calbibration data folder check 
    if os.path.exists(opt.ws) and os.path.isdir(opt.ws):
        logger.info(f"Folder '{opt.ws}' Exist, Removing...")
        try:
            shutil.rmtree(opt.ws)
            logger.info(f"Folder '{opt.ws}' Removed Successfully.")
            os.makedirs(os.path.join(opt.ws, opt.cal), exist_ok=True)
            logger.info(f"Folder '{opt.ws}' Created Successfully.")
        except Exception as e:
            logger.error(f"Remove Folder '{opt.ws}' Error: {e}")
            exit(1)
    else:
        logger.info(f"Folder '{opt.ws}' NOT Exist, Creating...")
        try:
            os.makedirs(os.path.join(opt.ws, opt.cal), exist_ok=True)
            logger.info(f"Folder '{opt.ws}' Created Successfully.")
        except Exception as e:
            logger.error(f"Create Folder '{opt.ws}' Error: {e}")
            exit(1)

    # int16
    int16_config_str = ",set_all_nodes_int16" if opt.quantized == "int16" else ""

    yaml = f'''
model_parameters:
  onnx_model: '{opt.onnx}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: '{opt.ws}/bpu_model_output'
  output_model_file_prefix: '{opt.onnx[:-5]}_bayese_640x640_nv12'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{opt.ws}/{opt.cal}'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8{int16_config_str}
compiler_parameters:
  jobs: {opt.jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{opt.optimize_level}'
'''
    with open("config.yaml", "w", encoding="utf-8") as file:
        file.write(yaml)


    # prepare calibration data
    logger.info("prepare calibration data...")
    for img_name in img_names:
        img_path = os.path.join(opt.cal_images, img_name)
        img = cv2.imread(img_path)
        # 此处的前处理以ONNX的前处理为基础，总的来说是和训练时的前处理保持一致
        # 如果yaml中有配置mean和scale, 则此处无须计算mean和scale.
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR2RGB
        input_tensor = cv2.resize(input_tensor, (width, height)) # resize
        input_tensor = np.transpose(input_tensor, (2, 0, 1))    # HWC2CHW
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)  # CHW -> NCHW
        dst_path = os.path.join(opt.ws, opt.cal, img_name + '.rgbchw') # tofile
        input_tensor.tofile(dst_path)
    logger.info("calibration data has been successfully generated.")

    # mapper
    os.system("hb_mapper makertbin --config config.yaml --model-type onnx")
    os.system(f"mv {opt.ws}/bpu_model_output/{opt.onnx[:-5]}_bayese_640x640_nv12.bin .")

    # clean the work space
    logger.info("Cleaning up...")
    
    if not opt.save_cache:
        os.system("rm config.yaml")
        os.system("rm -rf " + opt.ws)
    logger.info("Cleaning up completed.")

    
    logger.info(f"convert completed, file: {opt.onnx[:-5]}_bayese_640x640_nv12.bin")

if __name__ == "__main__":
    main()