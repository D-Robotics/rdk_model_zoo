import os
import argparse
import logging 

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



# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal-images', type=str, default='./cal_images', help='*.jpg, *.png calibration images path, 20 ~ 50 pictures is OK.') 
    parser.add_argument('--onnx', type=str, default='./yolo11n.onnx', help='origin float onnx model path.')
    # default below
    parser.add_argument('--jobs', type=int, default=16, help='model combine jobs.')
    parser.add_argument('--optimize-level', type=str, default='O3', help='O0, O1, O2, O3')
    opt = parser.parse_args()
    logger.info(opt)

    # check onnx file
    if not os.path.exists(opt.onnx):
        logger.error(f"ONNX file: \'{opt.onnx}\' is not exist, please check!")
        exit()

    # check cal-images file
    if not os.path.exists(opt.cal_images):
        logger.error(f"cal-images file: \'{opt.cal_images}\' is not exist, please check!")
        exit()
    if len(os.listdir(opt.cal_images))==0:
        logger.error(f"cal-images file: \'{opt.cal_images}\' is empty, please check!")
        exit()
    img_cnt = 0
    for name in os.listdir(opt.cal_images):
        if name.endswith(".jpg") or name.endswith(".png"):
            img_cnt += 1
    if img_cnt == 0:
        logger.error(f"cal-images file: \'{opt.cal_images}\' is too small, please check!")
        exit()
    if img_cnt < 20:
        logger.warning(f"There are {img_cnt} ( <20 ) images in the calibration dataset, which may cause the calibration to fail.")
    if img_cnt > 100:
        logger.warning(f"There are {img_cnt} ( >100 ) images in the calibration dataset, which may cost a long time to calibrate.")


    os.system("rm ")
    os.system("mkdir -p " + opt.output_dir)


    yaml = f'''
model_parameters:
  onnx_model: '{opt.onnx}'
  march: "bayes-e"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: '{opt.onnx[:-5]}_bayese_640x640_nv12'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: 'calibration_data_rgb_f32_coco_640'
  cal_data_type: 'float32'
  calibration_type: 'default'
  optimization: set_Softmax_input_int8,set_Softmax_output_int8
compiler_parameters:
  jobs: {opt.combine_jobs}
  compile_mode: 'latency'
  debug: true
  optimize_level: '{opt.optimize_level}'
'''
    with open("config.yaml", "w", encoding="utf-8") as file:
        file.write(yaml)
    
