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

import logging 
# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")
def main():
    coco_true = COCO(annotation_file='instances_val2017.json')  # 标准数据集（真值）
    coco_pre = coco_true.loadRes('predict_results.json')  # 预测数据集（预测值）

    coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType="bbox")    #计算bbox值
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


if __name__ == "__main__":
    main()