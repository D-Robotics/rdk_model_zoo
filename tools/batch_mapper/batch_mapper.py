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

# 注意: 此程序在OpenExplore发布的Docker中运行
# Attention: This program runs on OpenExplore Docker.

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
    parser.add_argument('--onnxs-path', type=str, default='pt/pt_seg', help='') 
    parser.add_argument('--ws-path', type=str, default='mapper_ws_yolo11seg_nv12', help='')
    parser.add_argument('--cal-data-path', type=str, default='calibration_data_rgb_f32_640', help='校准数据集的目录')
    parser.add_argument('--march', type=str, default='bayes-e', help='目标处理器的架构')
    parser.add_argument('--info-str', type=str, default='_bayese_640x640_', help='')
    parser.add_argument('--types', type=str, default='nv12', help='nv12 / nchwrgb')
    parser.add_argument('--cp-bin-file', type=str, default='ptq_models_yolo11seg_nv12', help='最终bin模型的发布文件夹')
    parser.add_argument('--jobs', type=int, default=4, help='编译时的线程数')
    parser.add_argument('--test', type=bool, default=False, help='True: 快速运行, False: 正常运行')
    opt = parser.parse_args()
    logger.info(opt)
    # modifieds
    yolov8_modified=['/model.22/cv2.0/cv2.0.2/Conv_output_0_quantized','/model.22/cv2.1/cv2.1.2/Conv_output_0_quantized','/model.22/cv2.2/cv2.2.2/Conv_output_0_quantized']
    yolov10_modified=['/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv_output_0_HzDequantize','/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv_output_0_HzDequantize','/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv_output_0_HzDequantize']
    yolo11_modified=['/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize','/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize','/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize']
    yolo11_seg_modified=['/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize','/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize','/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize','/model.23/cv4.0/cv4.0.2/Conv_output_0_HzDequantize','/model.23/cv4.1/cv4.1.2/Conv_output_0_HzDequantize','/model.23/cv4.2/cv4.2.2/Conv_output_0_HzDequantize','585_HzDequantize','720_HzDequantize','1082_HzDequantize']
    yolo12_modified=['/model.21/cv2.0/cv2.0.2/Conv_output_0_HzDequantize','/model.21/cv2.1/cv2.1.2/Conv_output_0_HzDequantize','/model.21/cv2.2/cv2.2.2/Conv_output_0_HzDequantize']
    # verify
    onnx_paths = opt.onnxs_path
    logger.info("\033[1;32m" + f"configs:" + "\033[0m")
    logger.info(f"--onnxs-path: {opt.onnxs_path}")
    logger.info(f"--ws-path: {opt.ws_path}")
    logger.info(f"--cal-data-path: {opt.cal_data_path}")
    logger.info(f"--march: {opt.march}")
    logger.info(f"--info_str: {opt.info_str}")
    logger.info(f"--types: {opt.types}")
    logger.info(f"--cp-bin-file: {opt.cp_bin_file}")
    logger.info(f"--jobs: {opt.jobs}")
    logger.info(f"--test: {opt.test}")
    logger.info("\033[1;32m" + f"Detected bin models:" + "\033[0m")
    for i, onnx_name in enumerate(os.listdir(onnx_paths)):
        if onnx_name.endswith("onnx"):
            logger.info(f"[{i}]: {onnx_name}")
    if not "y" == input("[test] continue? (y/n) "):
        exit()
    # mappers
    begin_time = time()
    for onnx_name in os.listdir(onnx_paths):
        if onnx_name.endswith("onnx"):
            mapper_once(ws_path=opt.ws_path,
                        onnx_path=os.path.join(onnx_paths, onnx_name), 
                        info_str = opt.info_str, 
                        cal_data_dir=opt.cal_data_path,
                        march=opt.march,
                        types=opt.types,
                        cp_bin_file=opt.cp_bin_file,
                        bin_node_modified=yolo11_seg_modified,
                        jobs=opt.jobs,
                        test=opt.test
                        )
    # 结束
    logger.info("\033[1;31m" + f"Total time = {((time() - begin_time)/60):.2f} minute(s)" + "\033[0m")
    # 删除test的产物
    if opt.test:
        cnt = 0
        while True:
            option = input(f"\"{opt.ws_path}\", \"{opt.cp_bin_file}\" will be removed [y/n]:")
            if option == "y":
                cmd = f"rm -rf {opt.ws_path}/"
                logger.info("[CMD]" + cmd)
                os.system(cmd)
                cmd = f"rm -rf {opt.cp_bin_file}/"
                logger.info("[CMD]" + cmd)
                os.system(cmd)
                logger.info(f"[test] \"{opt.ws_path}\", \"{opt.cp_bin_file}\" has been removed.")
                break
            if option == "n":
                logger.info(f"[test] not remove")
                break
            cnt += 1
            print(f"[test] [{cnt}] Choose again.")
            if cnt > 3:
                logger.info(f"[test] not remove")
                break
    

def mapper_once(ws_path="mapper_ws",
                onnx_path="pt/yolo11n_transposeSoftmax.onnx",
                info_str="_detect_bayese_640x640_",
                cal_data_dir="calibration_data_rgb_f32_640",
                march="bayes-e",
                types="nv12",
                cp_bin_file="ptq_models",
                bin_node_modified=['/model.23/cv2.0/cv2.0.2/Conv_output_0_HzDequantize',
                              '/model.23/cv2.1/cv2.1.2/Conv_output_0_HzDequantize',
                              '/model.23/cv2.2/cv2.2.2/Conv_output_0_HzDequantize'],
                jobs=8,
                test=True):
    logger.info(f"[START] {onnx_path}")
    # basic paths
    onnx_name = os.path.basename(onnx_path)
    output_model_file_prefix = onnx_name[:-5].replace("-", "_") + info_str + types
    working_dir = os.path.join(ws_path, output_model_file_prefix)
    yaml_path = output_model_file_prefix + ".yaml"
    bin_model_path = os.path.join(ws_path, output_model_file_prefix, output_model_file_prefix+".bin")
    bin_modified_model_path = os.path.join(ws_path, output_model_file_prefix, output_model_file_prefix+"_modified.bin")
    
    # 准备yaml文件
    ## model_parameters
    model_parameters = f'''
model_parameters:
    onnx_model: '{onnx_path}'
    march: "{march}"
    layer_out_dump: False
    working_dir: '{working_dir}'
    output_model_file_prefix: '{output_model_file_prefix}'
'''
    ## input_parameters
    scale_value = 0.003921568627451
    if types == "nv12":
        input_parameters = f'''
input_parameters:
    input_name: ""
    input_type_rt: 'nv12'
    input_type_train: 'rgb'
    input_layout_train: 'NCHW'
    norm_type: 'data_scale'
    scale_value: {scale_value}
        '''
    elif types == "nchwrgb":
        input_parameters = f'''
input_parameters:
    input_name: ""
    input_type_rt: 'rgb'
    input_layout_rt: 'NCHW'
    input_type_train: 'rgb'
    input_layout_train: 'NCHW'
    norm_type: 'data_scale'
    scale_value: {scale_value}
        '''
    else:
        logger.error("Wrong types, optioned in [nv12/rgb]")
        exit(-1)
    ## calibration_parameters 校准参数组
    if test:
        calibration_parameters = f'''
calibration_parameters:
    calibration_type: skip
'''
    else:
        calibration_parameters = f'''
calibration_parameters:
    cal_data_dir: '{cal_data_dir}'
    cal_data_type: 'float32'
''' 
    ## compiler_parameters
    if test:
        compiler_parameters = f'''
compiler_parameters:
    jobs: {jobs}
    compile_mode: 'latency'
    debug: True
    advice: 1
    optimize_level: 'O0'
'''
    else:
        compiler_parameters = f'''
compiler_parameters:
    jobs: {jobs}
    compile_mode: 'latency'
    debug: True
    advice: 1
    optimize_level: 'O3'
'''
    yaml = model_parameters + input_parameters + calibration_parameters + compiler_parameters
    with open(yaml_path, 'w', encoding='utf-8') as file:
        file.write(yaml)
    logger.info(f"[YAML] {yaml_path}")
    # 开始编译
    cmd = f"hb_mapper makertbin --config {yaml_path}  --model-type onnx"
    logger.info("[CMD]" + cmd)
    os.system(cmd)
    cmd = f"cp hb_mapper_makertbin.log {working_dir}/"
    logger.info("[CMD]" + cmd)
    os.system(cmd)
    # 移除反量化节点
    bin_model_path_final = bin_model_path
    if bin_node_modified is not None:
        hb_model_modifier_cmd = f"hb_model_modifier {bin_model_path} "
        for node_name in bin_node_modified:
            hb_model_modifier_cmd += f"-r {node_name} "
        logger.info("[CMD]" + hb_model_modifier_cmd) 
        os.system(hb_model_modifier_cmd)

        cmd = f"cp hb_model_modifier.log {working_dir}/"
        logger.info("[CMD]" + cmd)
        os.system(cmd)
        bin_model_path_final = bin_modified_model_path
    # 移动产物到交付目录
    os.makedirs(cp_bin_file, exist_ok=True)
    cmd = f"cp {bin_model_path_final} {cp_bin_file}"
    logger.info("[CMD]" + cmd)
    os.system(cmd)
    # 删除yaml文件
    os.remove(yaml_path)
    logger.info(f"[YAML] \"{yaml_path}\" has been removed.")
    logger.info(f"[END] {onnx_path}")

if __name__ == "__main__":
    main()
