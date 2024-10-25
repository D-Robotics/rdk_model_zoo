#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
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

# 注意: 此程序推荐在转化环境中运行
# Attention: This program is recommended to run in a transformation environment.



import os
import cv2
import numpy as np

import argparse
import logging 

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='src', help="Source images path") 
    parser.add_argument('--dist', type=str, default='calibration_data_rgb_f32_640', help="Destination images path") 
    parser.add_argument('--width', type=int, default=640, help="W in ONNX NCHW.") 
    parser.add_argument('--height', type=int, default=640, help="H in ONNX NCHW.")

    opt = parser.parse_args()
    logger.info(opt)

    # 检查源图片文件夹是否存在
    if not os.path.exists(opt.src):
        logger.error("Source images path is not exist, please check!")


    # 如果目标文件夹存在, 则报错退出
    if os.path.exists(opt.src):
        logger.error("Destination folder already exists, please check or remove it!")
    else:
        os.makedirs(opt.dist)
        logger.info("\033[1;32m" + f"Created directory Successfully: \"{opt.dist}\"" + "\033[0m")

    # 逐个转化并保存
    for img_name in os.listdir(opt.src):
        img_path = os.path.join(opt.src, img_name)
        img = cv2.imread(img_path)
        input_tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(img, (opt.width, opt.height))
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)  # NCHW
        dst_path = os.path.join(opt.dist, img_name + '.rgbchw')
        input_tensor.tofile(dst_path)
        logger.info("write:%s" % dst_path)

if __name__ == "__main__":
    main()