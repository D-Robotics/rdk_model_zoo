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

from time import time
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
    parser.add_argument('--model-path', type=str, default='models/yolo11n_detect_bayese_640x640_nv12_modified.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../resource/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='jupyter_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    opt = parser.parse_args()
    logger.info(opt)


src_root = '/open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/coco'
cal_img_num = 100  # 想要的图像个数
dst_root = '/open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/02_yolov3_darknet53/mapper/calibration_data'


## 1. 从原始图像文件夹中获取100个图像作为校准数据
num_count = 0
img_names = []
for src_name in sorted(os.listdir(src_root)):
    if num_count > cal_img_num:
        break
    img_names.append(src_name)
    num_count += 1

# 检查目标文件夹是否存在，如果不存在就创建
if not os.path.exists(dst_root):
    os.system('mkdir {0}'.format(dst_root))

## 2 为每个图像转换
# 参考了OE中/open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/01_common/python/data/下的相关代码
# 转换代码写的很棒，很智能，考虑它并不是官方python包，所以我打算换一种写法

## 2.1 定义图像缩放函数，返回为np.float32
# 图像缩放为目标尺寸(W, H)
# 值得注意的是，缩放时候，长宽等比例缩放，空白的区域填充颜色为pad_value, 默认127
def imequalresize(img, target_size, pad_value=127.):
    target_w, target_h = target_size
    image_h, image_w = img.shape[:2]
    img_channel = 3 if len(img.shape) > 2 else 1

    # 确定缩放尺度，确定最终目标尺寸
    scale = min(target_w * 1.0 / image_w, target_h * 1.0 / image_h)
    new_h, new_w = int(scale * image_h), int(scale * image_w)

    resize_image = cv2.resize(img, (new_w, new_h))

    # 准备待返回图像
    pad_image = np.full(shape=[target_h, target_w, img_channel], fill_value=pad_value)

    # 将图像resize_image放置在pad_image的中间
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    pad_image[dh:new_h + dh, dw:new_w + dw, :] = resize_image

    return pad_image

## 2.2 开始转换
for each_imgname in img_names:
    img_path = os.path.join(src_root, each_imgname)

    img = cv2.imread(img_path)  # BRG, HWC
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB, HWC
    img = imequalresize(img, (416, 416))
    img = np.transpose(img, (2, 0, 1))  # RGB, CHW

    # 将图像保存到目标文件夹下
    dst_path = os.path.join(dst_root, each_imgname + '.rgbchw')
    print("write:%s" % dst_path)
    # 图像加载默认就是uint8，但是不加这个astype的话转换模型就会出错
    # 转换模型时候，加载进来的数据竟然是float64，不清楚内部是怎么加载的。
    img.astype(np.uint8).tofile(dst_path) 

print('finish')


if __name__ == "__main__":
    main()