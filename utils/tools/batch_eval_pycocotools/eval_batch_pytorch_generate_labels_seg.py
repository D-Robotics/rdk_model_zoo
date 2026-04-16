#!/user/bin/env python

# Copyright (c) 2025, Cauchy - WuChao in D-Robotics.
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


import os
import argparse
import logging 
from time import time

# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-script', type=str, default='eval_seg_pytorch_generate_labels.py', help='') 
    parser.add_argument('--pt-paths', type=str, default='pt/pt_seg', help='') 
    parser.add_argument('--json-paths', type=str, default='eval_seg', help='') 
    parser.add_argument('--str', type=str, default='pt_coco2017_val_pridect', help='')
    opt = parser.parse_args()
    logger.info(opt)

    logger.info(f"--str: {opt.str}")
    logger.info(f"Detected pt models:")
    for i, pt_name in enumerate(os.listdir(opt.pt_paths)):
        if pt_name.endswith(".pt"):
            logger.info(f"[{i}]:")
            logger.info(os.path.join(opt.pt_paths, pt_name))
            logger.info(os.path.join(opt.json_paths, pt_name[:-3] + "_" + opt.str + ".json"))
    # 开始
    begin_time = time()
    if "y" == input("[test] continue? (y/n) "):
        for i, pt_name in enumerate(os.listdir(opt.pt_paths)):
            if pt_name.endswith(".pt"):
                pt_path = os.path.join(opt.pt_paths, pt_name)
                json_path = os.path.join(opt.json_paths, pt_name[:-3] + "_" + opt.str + ".json")

                # 编译运行
                cmd = f"python3 {opt.eval_script} --model-path {pt_path} --json-path {json_path}"
                logger.info("[CMD]" + cmd)
                os.system(cmd)
    # 结束
    logger.info("\033[1;31m" + f"Total time = {((time() - begin_time)/60):.2f} minute(s)" + "\033[0m")
    logger.info("[END] end batch python eval.")



if __name__ == '__main__':
    main()