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
    parser.add_argument('--eval-script', type=str, default='py/eval_Ultralytics_YOLO_Pose_YUV420SP.py', help='') 
    parser.add_argument('--bin-paths', type=str, default='source/reference_bin_models/seg', help='') 
    parser.add_argument('--str', type=str, default='py_coco2017_val_pridect', help='')
    opt = parser.parse_args()
    logger.info(opt)

    logger.info(f"--str: {opt.str}")
    logger.info(f"Detected bin models:")
    for i, bin_name in enumerate(os.listdir(opt.bin_paths)):
        if bin_name.endswith(".bin"):
            logger.info(f"[{i}]:")
            logger.info(os.path.join(opt.bin_paths, bin_name))
            logger.info(bin_name[:-4] + "_" + opt.str + ".json")
    # 开始
    begin_time = time()
    if "y" == input("[test] continue? (y/n) "):
        for i, bin_name in enumerate(os.listdir(opt.bin_paths)):
            if bin_name.endswith(".bin"):
                bin_path = os.path.join(opt.bin_paths, bin_name)
                json_path = bin_name[:-4] + "_" + opt.str + ".json"

                # 编译运行
                cmd = f"python3 {opt.eval_script} --model-path {bin_path} --json-path {json_path}"
                logger.info("[CMD]" + cmd)
                os.system(cmd)
    # 结束
    logger.info("\033[1;31m" + f"Total time = {((time() - begin_time)/60):.2f} minute(s)" + "\033[0m")
    logger.info("[END] end batch python eval.")



if __name__ == '__main__':
    main()