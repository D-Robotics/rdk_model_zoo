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

# 注意: 此程序在OpenExplore发布的Docker中运行
# Attention: This program runs on RDK board.

# 复制一份cpp_eval目录, 进入cpp_eval目录运行此脚本

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
    parser.add_argument('--bin-paths', type=str, default='../ptq_models', help='') 
    parser.add_argument('--str', type=str, default='cpp_coco2017_val_pridect', help='')
    parser.add_argument('--types', type=str, help='nv12 / nchwrgb')
    opt = parser.parse_args()
    logger.info(opt)

    logger.info(f"--types: {opt.types}")
    logger.info(f"--str: {opt.str}")
    logger.info(f"Detected bin models:")
    for i, bin_name in enumerate(os.listdir(opt.bin_paths)):
        if bin_name.endswith(".bin"):
            logger.info(f"[{i}]:")
            logger.info(os.path.join("../", opt.bin_paths, bin_name))
            logger.info(os.path.join("../", bin_name[:-4] + "_" + opt.str + "_" + opt.types + ".json"))
    # 开始
    begin_time = time()
    if "y" == input("[test] continue? (y/n) "):
        for i, bin_name in enumerate(os.listdir(opt.bin_paths)):
            if bin_name.endswith(".bin"):
                bin_path = os.path.join("../", opt.bin_paths, bin_name)
                json_path = os.path.join("../", bin_name[:-4] + "_" + opt.str + "_" + opt.types + ".json")
                with open("main.cc", 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                cpp_source_code = ""
                is_model_path = True
                is_json_path = True
                for line in lines:
                    if "#define MODEL_PATH" in line and is_model_path:
                        cpp_source_code += f"#define MODEL_PATH \"{bin_path}\"\n"
                        is_model_path = False
                        continue
                    if "#define JSON_RESULT_PATH" in line and is_json_path:
                        cpp_source_code += f"#define JSON_RESULT_PATH \"{json_path}\"\n"
                        is_json_path = False
                        continue
                    cpp_source_code += line

                # 保存为main.cc文件
                with open("main.cc", 'w', encoding='utf-8') as f:
                    f.write(cpp_source_code)
                del(cpp_source_code)
                # 编译运行
                cmd = f"rm -rf build"
                logger.info("[CMD]" + cmd)
                os.system(cmd)
                cmd = f"mkdir -p build && cd build && cmake .. && make && ./main"
                logger.info("[CMD]" + cmd)
                os.system(cmd)
    # 结束
    logger.info("\033[1;31m" + f"Total time = {((time() - begin_time)/60):.2f} minute(s)" + "\033[0m")
    logger.info("[END] end batch cpp eval.")



if __name__ == '__main__':
    main()