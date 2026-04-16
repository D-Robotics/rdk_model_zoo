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

# 注意: 此程序在RDK板端运行
# Attention: This program runs on RDK board.

from Ultralytics_YOLO_Classify_YUV420SP import *

import json

import argparse
import logging 
import os
import re

from datetime import datetime



# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='source/reference_bin_models/cls/yolo11n_cls_detect_bayese_640x640_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--image-path', type=str, default="../../../../datasets/imagenet/val_images", help='COCO2017 val source image path.')
    parser.add_argument('--json-path', type=str, default="yolo11n_cls_detect_bayese_640x640_nv12_py_coco2017_val_pridect.json", help='convert to json save path.')
    parser.add_argument('--max-num', type=int, default=100000, help='max num of images which will be precessed.')
    opt = parser.parse_args()
    logger.info(opt)
    begin_time = time()
    run(opt)
    print("\033[1;31m" + f"Total time = {(time() - begin_time)/60:.2f} min" + "\033[0m")

def run(opt):
    # 准备int的映射
    label2id = {}
    for i, key in enumerate(IMAGENET2012_CLASSES.keys()):
        label2id[key] = i
    # 实例化
    model = Ultralytics_YOLO_Calssify_Bayese_YUV420SP(model_path=opt.model_path)
    
    # 的预测结果json文件
    pridect_json = [{}]
    pridect_json[0]['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pridect_json[0]['opt.model_path'] = opt.model_path

    # 板端直接推理所有验证集，并生成标签和渲染结果 (串行程序)
    img_names = os.listdir(opt.image_path)
    img_num = len(img_names)
    control = opt.max_num
    total_cnt, top1_cnt, top5_cnt = 0, 0, 0
    for cnt, img_name in enumerate(img_names, 1):
        # 流程控制
        if control < 1:
            break
        control -= 1

        logger.info("\033[1;32m" + f"[{cnt}/{img_num}] Processing image: \"{img_name}\"" + "\033[0m")
        # 从文件名称中提取Truth
        match = re.search(r'n\d+', img_name)
        if match:
            truth = label2id[match.group(0)]
        else: 
            logger.error("Truth not found in file path.")
            continue
        # 端到端推理
        img = cv2.imread(os.path.join(opt.image_path, img_name))
        input_tensor = model.preprocess_yuv420sp(img)
        outputs = model.c2numpy(model.forward(input_tensor))
        id_, score = model.postProcess(outputs)

        total_cnt += 1
        if truth == id_[0]:
            top1_cnt += 1
            top5_cnt += 1
            logger.info("TOP 1 HIT")
        elif truth in id_:
            top5_cnt += 1
            logger.info("TOP 5 HIT")
        else:
            logger.info("MISS")

    pridect_json[0]['total_cnt'] = total_cnt
    pridect_json[0]['top1_cnt'] = top1_cnt
    pridect_json[0]['top5_cnt'] = top5_cnt
    pridect_json[0]['top1_acc'] = top1_cnt / total_cnt
    pridect_json[0]['top5_acc'] = top5_cnt / total_cnt

    # 保存标签
    with open(opt.json_path, 'w') as f:
        json.dump(pridect_json, f, ensure_ascii=False, indent=1)
    
    logger.info("\033[1;32m" + f"result label saved: \"{opt.json_path}\"" + "\033[0m")
    

if __name__ == "__main__":
    main()