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

# 注意: 此程序在x86开发机的ultralytics环境中运行
# Attention: This program runs in the ultralytics environment of an x86 development machine


from ultralytics import YOLO

import json

import argparse
import logging 
import os
import gc


# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='pt/pt_seg/yolo11m-seg.pt', 
                        help="""Path to Detect Pytorch Model.""") 
    parser.add_argument('--type', type=str, default='seg', help='detect, seg') 
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--nms-thres', type=float, default=0.7, help='IoU threshold.')
    parser.add_argument('--score-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--image-path', type=str, default="./val2017", help='COCO2017 val source image path.')
    parser.add_argument('--result-image-dump', type=bool, default=False, help='dump image result or not')
    parser.add_argument('--result-image-path', type=str, default="coco2017_image_result", help='COCO2017 val image result saving path.')
    parser.add_argument('--json-path', type=str, default="eval_seg/yolo11m-seg_pt_coco2017_val_pridect.json", help='convert to json save path.')
    parser.add_argument('--max-num', type=int, default=500000, help='max num of images which will be precessed.')
    opt = parser.parse_args()
    logger.info(opt)

    # id -> coco_id
    coco_id = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,51,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]

    # 实例化
    model = YOLO(opt.model_path)

    # 创建 result* 目录存储结果
    if opt.result_image_dump:
        for i in range(9999):
            result_image_path = f"{opt.result_image_path}_{i}"
            if not os.path.exists(result_image_path):
                os.makedirs(result_image_path)
                logger.info("\033[1;32m" + f"Created directory Successfully: \"{result_image_path}\"" + "\033[0m")
                break
    
    # 生成pycocotools的预测结果json文件
    pridect_json = []

    # 板端直接推理所有验证集，并生成标签和渲染结果 (串行程序)
    img_names = os.listdir(opt.image_path)
    img_num = len(img_names)
    control = opt.max_num
    for cnt, img_name in enumerate(img_names, 1):
        # 流程控制
        if control < 1:
            break
        control -= 1

        logger.info("\033[1;32m" + f"[{cnt}/{img_num}] Processing image: \"{img_name}\"" + "\033[0m")
        # 端到端推理
        results = model([os.path.join(opt.image_path, img_name)], conf=opt.score_thres, iou=opt.nms_thres)
        img = results[0].orig_img
        # 保存到JSON
        id_cnt = 0
        for i in range(len(results[0].boxes.cls)):
            if opt.type == "detect":
                class_id, score, xyxy = results[0].boxes.cls[i], results[0].boxes.conf[i], results[0].boxes.xyxy[i]
                class_id = int(class_id)
                score = float(score)
                x1, x2, y1, y2 = int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3])
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
            elif opt.type == "seg":
                # detect
                class_id, score, xyxy = results[0].boxes.cls[i], results[0].boxes.conf[i], results[0].boxes.xyxy[i]
                class_id = int(class_id)
                score = float(score)
                x1, x2, y1, y2 = int(xyxy[0]), int(xyxy[2]), int(xyxy[1]), int(xyxy[3])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                x1, y1, x2, y2, width, height = float(x1), float(y1), float(x2), float(y2), float(width), float(height)
                # seg
                if len(results[0].masks[i].xy[0])<=2:
                    seg_points = [x1, y1, x2, y1, x2, y2, x1, y2] 
                else:
                    seg_points = []
                    for x, y in results[0].masks[i].xy[0]:
                        seg_points.append(float(int(x)))
                        seg_points.append(float(int(y)))
                # debug
                # print("* "*20)
                # print({
                #         'area': width * height,
                #         'bbox': [x1, y1, width, height],
                #         'category_id': int(coco_id[class_id]),
                #         'id': id_cnt,
                #         "score": float(score),
                #         'image_id': int(img_name[:-4]),
                #         'iscrowd': 0,
                #         'segmentation': [seg_points]
                #     })
                # if int(img_name[:-4])==525083:
                #     results[0].save("seg_result.jpg")
                # result
                pridect_json.append({
                        'area': width * height,
                        'bbox': [x1, y1, width, height],
                        'category_id': int(coco_id[class_id]),
                        'id': id_cnt,
                        "score": float(score),
                        'image_id': int(img_name[:-4]),
                        'iscrowd': 0,
                        'segmentation': [seg_points]
                    })
                id_cnt += 1
            else:
                logger.error(f"Bad --type {opt.type}")
                exit(0)

    # 保存标签
    with open(opt.json_path, 'w') as f:
        json.dump(pridect_json, f, ensure_ascii=False, indent=1)
    
    logger.info("\033[1;32m" + f"result label saved: \"{opt.json_path}\"" + "\033[0m")
    

if __name__ == "__main__":
    main()