#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2021-2024 D-Robotics Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

一个通用的模型推理脚本，支持ONNX,HBIR(.bc)和HBM格式。
该脚本会加载原始图像，执行用户定义的预处理，使用地平线的HBRuntime执行推理，
并以后处理函数定义的格式输出分类结果到控制台。
"""

import numpy as np
import argparse
import os
from typing import Iterable
import logging # 导入logging模块

# 地平线AI工具链库
IMAGENET_VAL_CLASSES = None # 初始化为None

try:
    from horizon_tc_ui import HB_ONNXRuntime, HBRuntime, __version__
    from horizon_tc_ui.data.dataloader import SingleImageDataLoader
    from horizon_tc_ui.data.transformer import (
        PaddedCenterCropTransformer,
        HWC2CHWTransformer,
        MeanTransformer,
        ScaleTransformer,
        ResizeTransformer,
        RGB2NV12Transformer,
        RGB2BGRTransformer,
        BGR2NV12Transformer
    )
    # 直接导入ImageNet验证集的类别名称列表
    from horizon_tc_ui.data.imagenet_val import imagenet_val as IMAGENET_VAL_CLASSES

except ImportError:
    logging.warning("警告：无法完整导入 horizon_tc_ui 相关模块。")
    logging.warning("确保已正确安装地平线AI工具链，并且PYTHONPATH已正确设置。")
    logging.warning("如果 imagenet_val 不可用，相关功能将受限。")


def onnx_transformer():
    """
    定义ONNX指定的图像预处理转换器列表。
    """
    transformers = [
        PaddedCenterCropTransformer(256),
        ResizeTransformer(
            target_size=(256, 256),
            mode='skimage',
            method=3
        ),
        HWC2CHWTransformer(),
        RGB2BGRTransformer(data_format="CHW"),
        ScaleTransformer(scale_value=255.0),
        MeanTransformer(means=np.array([103.94, 116.78, 123.68])),
        ScaleTransformer(scale_value=0.017)
    ]
    return transformers

def quantied_transformers():
    """
    定义.bc指定的图像预处理转换器列表。
    """
    transformers = [
        PaddedCenterCropTransformer(256),
        ResizeTransformer(target_size=(256, 256),
                          mode='skimage',
                          method=3),
        RGB2BGRTransformer(data_format="HWC"),
        ScaleTransformer(scale_value=255),
        BGR2NV12Transformer(data_format="HWC")
    ]
    return transformers

def postprocess_classification_output(model_output: list, top_k: int = 5) -> list:
    """
    对分类模型的输出进行后处理，提取Top-K结果。
    直接使用从 horizon_tc_ui.data.imagenet_val 导入的 imagenet_val。
    """
    global IMAGENET_VAL_CLASSES # 使用全局的类别列表

    if not model_output or not isinstance(model_output[0], np.ndarray):
        logging.error("后处理错误：无效的模型输出格式。")
        return []

    scores = np.squeeze(model_output[0])

    if scores.ndim == 0:
        logging.error(f"后处理错误：模型输出在squeeze后变为标量，原始形状: {model_output[0].shape}")
        return []
    if scores.ndim != 1:
        logging.warning(f"后处理警告：期望squeeze后得到1D分数数组，实际得到 {scores.ndim}D。尝试flatten。")
        if scores.size == model_output[0].shape[-1] or (IMAGENET_VAL_CLASSES and scores.size == len(IMAGENET_VAL_CLASSES)):
            scores = scores.flatten()
        else:
            logging.error(f"后处理错误：无法将scores (shape: {scores.shape}) 处理为1D数组。")
            return []

    # 应用Softmax将logits转换为概率
    exp_logits = np.exp(scores - np.max(scores)) # 减去max是为了数值稳定性
    scores = exp_logits / np.sum(exp_logits)

    idx = np.argsort(-scores)

    top_k_results = []
    num_classes_available = len(scores)
    for i in range(min(top_k, num_classes_available)):
        label_index = idx[i]
        probability = scores[label_index]
        try:
            class_name = IMAGENET_VAL_CLASSES[label_index]
        except (TypeError, IndexError) as e_label:
            class_name = f"未知标签索引_{label_index}"
            logging.warning(f"无法获取类别名称 for index {label_index}: {e_label}")
        top_k_results.append((label_index, probability, class_name))

    return top_k_results

def nv12_split_yuv(target_size: Iterable, input_shapes: list,
                   input_data: np.ndarray) -> list:
    width, height = target_size
    image = input_data.flatten()
    y_data = image[:width * height].reshape(input_shapes[0])
    uv_data = image[width * height:].reshape(input_shapes[1])
    return [y_data, uv_data]

def run_model_inference(model_path: str, raw_image_path: str):
    """
    加载模型和原始图像，执行预处理、推理和后处理，并打印输出信息。
    """
    if not os.path.exists(model_path):
        logging.error(f"错误：模型文件未找到: {model_path}")
        return

    if model_path.endswith('.onnx'):
        active_transformers = onnx_transformer()
    else :
        active_transformers = quantied_transformers()

    try:
        data = SingleImageDataLoader(active_transformers,
                                    raw_image_path,
                                    imread_mode='skimage')
    except Exception:
        return

    logging.info(f"[*] 正在加载模型: {model_path}")
    try:
        sess = HBRuntime(model_path)
        logging.info("    模型加载成功。")
    except Exception as e:
        logging.error(f"错误：加载模型失败: {e}")
        return

    try:
        input_names = sess.input_names
        output_names = sess.output_names

        if not input_names:
            logging.error("错误：未能从模型获取输入节点名称。")
            return
        logging.info(f"    模型输入节点名称: {input_names}")

        if isinstance(sess.sess, HB_ONNXRuntime):
            feed_dict = {input_names[0]: data}
        else:
            image_data_processed = nv12_split_yuv(target_size=[256,256],
                                                  #   target_size=sess.sess.get_hw(),
                                                  input_shapes=sess.input_shapes,
                                                  input_data=data)
            feed_dict = dict(zip(input_names, image_data_processed))

        logging.info(f"[*] 正在使用输入节点 '{input_names[0]}' 进行模型推理...")

        outputs = sess.run(output_names, feed_dict)

        if outputs:
            logging.info("    模型推理完成。正在进行后处理...")
            top_results = postprocess_classification_output(outputs, top_k=5)

            if top_results:
                logging.info("The input picture is classified to be:")
                for label_item, prob_item, class_item in top_results:
                    logging.info(
                        f"    label {label_item:3d}, prob {prob_item:.5f}, class {class_item}")
            else:
                logging.warning("后处理未能生成有效分类结果。")
        else:
            logging.error("模型推理没有返回任何输出。")

    except Exception as e:
        logging.error(f"错误：模型推理或输出处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RDK模型推理脚本 (支持 ONNX, HBIR/.bc)，可直接输入原始图像进行预处理和推理。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--model_file",
        type=str,
        required=True,
        help="模型文件的路径 (例如: model.onnx, model.bc, model.hbm)"
    )
    parser.add_argument(
        "-i", "--image_file",
        type=str,
        required=True,
        help="原始输入图像文件的路径 (例如: image.jpg, image.png)"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("使用了Python标准日志进行初始化 。")

    run_model_inference(args.model_file, args.image_file)
