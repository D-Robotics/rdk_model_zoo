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

import json

import argparse
import logging 
import os


# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt-path', type=str, default="cpp_json_result.txt", help='convert to json save path.')
    parser.add_argument('--json-path', type=str, default="yolov8n_detect_bayese_640x640_nchwrgb_coco2017_val_pridect.json", help='convert to json save path.')
    opt = parser.parse_args()

    coco_id = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

    
    # 生成pycocotools的预测结果json文件
    pridect_json = []
    id_cnt = 0
    with open(opt.txt_path, 'r') as file:
        for line in file:
            # 去除行末尾的换行符并按照空格分割行内容
            parts = line.strip().split()
            if len(parts) != 7:
                print(f"Warning: Line does not contain 7 elements. Skipping this line: {line}")
                continue

            # 解析各个字段
            img_name = parts[0]  # 去除可能存在的引号
            class_id = int(parts[1])
            score = float(parts[2])
            x1 = float(parts[3])
            y1 = float(parts[4])
            x2 = float(parts[5])
            y2 = float(parts[6])

            # 将解析后的数据添加到列表中
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            x1, y1, x2, y2, width, height = float(x1), float(y1), float(x2), float(y2), float(width), float(height)
            pridect_json.append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(coco_id[class_id]),
                    'id': id_cnt,
                    "score": float(score),
                    'image_id': int(img_name[:-4]),
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
            id_cnt += 1
    # 保存标签
    with open(opt.json_path, 'w') as f:
        json.dump(pridect_json, f, ensure_ascii=False, indent=1)
    
    logger.info("\033[1;32m" + f"result label saved: \"{opt.json_path}\"" + "\033[0m")
    

if __name__ == "__main__":
    main()