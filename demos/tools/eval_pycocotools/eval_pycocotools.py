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

# 注意: 此程序推荐在BPU工具链Docker运行
# Attention: This program runs on ToolChain Docker recommended.

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    # 真值
    parser.add_argument('--truth', type=str, default='eval/instances_val2017.json')
    # 如果是目录的话，则遍历目录下的所有json文件
    # 如果是文件，则检测文件
    parser.add_argument('--json', type=str, default='eval/yolo11s_detect_bayese_640x640_nv12_coco2017_val_pridect.json')
    opt = parser.parse_args()

    if os.path.isdir(opt.json):
        for name in os.listdir(opt.json):
            if not name.endswith("json"):
                continue
            try:
                coco_true = COCO(annotation_file='eval/instances_val2017.json')  # 标准数据集（真值）
                coco_pre = coco_true.loadRes(os.path.join(opt.json, name))  # 预测数据集（预测值）
                coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")    #计算bbox值
                coco_evaluator.evaluate()
                coco_evaluator.accumulate()
                print("\n\n")
                print("\033[1;32m" + name + "\033[0m")
                coco_evaluator.summarize()
            except:
                print("\n\n")
                print("\033[1;32m" + name + "\033[0m")
                print("\033[1;31m" + "Failed!" + "\033[0m")
            print("\n\n")
    else:
        coco_true = COCO(annotation_file=opt.truth)  # 标准数据集（真值）
        coco_pre = coco_true.loadRes(opt.json)  # 预测数据集（预测值）
        coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")    #计算bbox值
        coco_evaluator.evaluate()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()


if __name__ == "__main__":
    main()