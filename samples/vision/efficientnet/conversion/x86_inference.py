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
        HWC2CHWTransformer,
        MeanTransformer,
        ScaleTransformer,
        RGB2NV12Transformer,
        ShortSideResizeTransformer,
        CenterCropTransformer
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
        ShortSideResizeTransformer(short_size=224),
        CenterCropTransformer(crop_size=224),
        HWC2CHWTransformer(),
        ScaleTransformer(scale_value=255.0),
        MeanTransformer(means=np.array([127.0, 127.0, 127.0])),
        ScaleTransformer(scale_value=np.array([0.007843, 0.007843, 0.007843]))
    ]
    return transformers

def quantied_transformers():
    """
    定义.bc指定的图像预处理转换器列表。
    """
    transformers = [
        ShortSideResizeTransformer(short_size=224),
        CenterCropTransformer(crop_size=224),
        ScaleTransformer(scale_value=255),
        RGB2NV12Transformer(data_format="HWC"),
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

def run_model_inference(image_path: str, sess: HBRuntime, top_k_for_output: int = 5):
    """
    执行单张图像的推理，并返回Top-K预测的类别索引和概率列表。
    """
    if not os.path.exists(image_path):
        logging.error(f"错误：图像文件未找到: {image_path}")
        return None

    # 从 sess 获取模型类型来决定使用哪种 transformer
    if isinstance(sess.sess, HB_ONNXRuntime):
        active_transformers = onnx_transformer()
    else:
        active_transformers = quantied_transformers()

    try:
        data = SingleImageDataLoader(active_transformers,
                                     image_path,
                                     imread_mode='skimage')
    except Exception as e:
        logging.error(f"错误：加载图像并预处理失败: {e}")
        return None

    try:
        input_names = sess.input_names
        output_names = sess.output_names

        if not input_names:
            logging.error("错误：未能从模型获取输入节点名称。")
            return None

        if isinstance(sess.sess, HB_ONNXRuntime):
            feed_dict = {input_names[0]: data}
        else:
            model_input_h, model_input_w = 224, 224
            if hasattr(sess.sess, 'get_hw'):
                 model_input_w, model_input_h = sess.sess.get_hw()

            image_data_processed = nv12_split_yuv(target_size=[model_input_w, model_input_h],
                                                  input_shapes=sess.input_shapes,
                                                  input_data=data)
            feed_dict = dict(zip(input_names, image_data_processed))

        outputs = sess.run(output_names, feed_dict)

        if outputs:
            top_results = postprocess_classification_output(outputs, top_k=top_k_for_output)
            return top_results # 返回 (label_index, probability) 对的列表
        else:
            logging.error(f"图像 {os.path.basename(image_path)} 模型推理没有返回任何输出。")
            return None

    except Exception as e:
        logging.error(f"错误：图像 {os.path.basename(image_path)} 推理或输出处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None

# 导入 tqdm 库用于进度条
try:
    from tqdm import tqdm
except ImportError:
    print("警告: 未安装 tqdm 库。进度条将不可用。请运行 'pip install tqdm' 安装。")
    tqdm = lambda x: x # 定义一个空函数，使代码在没有 tqdm 时也能运行

def validate_accuracy(model_path: str, dataset_root: str, val_list_file: str):
    """
    在ImageNet-1K验证集上同时计算Top-1和Top-5精度，并显示进度条。
    """

    if not os.path.exists(model_path):
        logging.error(f"错误：模型文件未找到: {model_path}")
        return

    if not os.path.exists(val_list_file):
        logging.error(f"错误：val.txt 文件未找到: {val_list_file}")
        return

    logging.info(f"[*] 正在加载模型: {model_path}")
    try:
        sess = HBRuntime(model_path)
        logging.info("    模型加载成功。")
    except Exception as e:
        logging.error(f"错误：加载模型失败: {e}")
        return

    correct_top1_predictions = 0
    correct_top5_predictions = 0
    total_images = 0

    logging.info(f"[*] 正在读取验证列表文件: {val_list_file}")
    with open(val_list_file, 'r') as f:
        lines = f.readlines()

    logging.info(f"[*] 开始在 {len(lines)} 张图片上进行精度验证 (计算 Top-1 和 Top-5 精度)...")

    # 使用 tqdm 包装 lines 迭代器以显示进度条
    # desc 是进度条前缀，unit 是单位
    for line_num, line in enumerate(tqdm(lines, desc="验证进度", unit="张图片")):
        line = line.strip()
        if not line:
            continue

        try:
            image_name, true_label = line.split(' ')
            true_label = int(true_label)
        except ValueError:
            logging.warning(f"警告：跳过 val.txt 中格式不正确的行: {line}")
            continue

        image_path = os.path.join(dataset_root, image_name)
        if not os.path.exists(image_path):
            logging.warning(f"警告：图像文件未找到，跳过: {image_path}")
            continue

        # 一次性获取 Top-5 的预测结果
        predicted_results_top5 = run_model_inference(image_path, sess, top_k_for_output=5)
        total_images += 1

        if predicted_results_top5 is not None and len(predicted_results_top5) > 0:
            # 提取预测标签索引
            predicted_labels = [res[0] for res in predicted_results_top5]

            # 检查 Top-1 精度
            if predicted_labels[0] == true_label:
                correct_top1_predictions += 1
                # logging.debug(f"图像 {image_name}: Top-1 预测 {predicted_labels[0]} == 真实 {true_label} (Top-1 正确)")
            # else:
                # logging.debug(f"图像 {image_name}: Top-1 预测 {predicted_labels[0]} != 真实 {true_label} (Top-1 错误)")

            # 检查 Top-5 精度
            if true_label in predicted_labels:
                correct_top5_predictions += 1
                # logging.debug(f"图像 {image_name}: Top-5 预测 {predicted_labels} 包含真实 {true_label} (Top-5 正确)")
            # else:
                # logging.debug(f"图像 {image_name}: Top-5 预测 {predicted_labels} 不包含真实 {true_label} (Top-5 错误)")
        # else:
            # logging.warning(f"图像 {image_name}: 未能获得有效预测结果，跳过。")

        if (line_num + 1) % 100 == 0:
            current_top1_accuracy = correct_top1_predictions / total_images if total_images > 0 else 0
            current_top5_accuracy = correct_top5_predictions / total_images if total_images > 0 else 0
            logging.info(f"已处理 {line_num + 1}/{len(lines)} 张图片. 当前 Top-1 准确率: {current_top1_accuracy:.4f}, Top-5 准确率: {current_top5_accuracy:.4f}")

    if total_images > 0:
        final_top1_accuracy = correct_top1_predictions / total_images
        final_top5_accuracy = correct_top5_predictions / total_images
        logging.info("---")
        logging.info(f"精度验证完成！")
        logging.info(f"总图像数: {total_images}")
        logging.info(f"Top-1 正确预测数: {correct_top1_predictions}")
        logging.info(f"Top-1 准确率: {final_top1_accuracy:.4f}")
        logging.info(f"Top-5 正确预测数: {correct_top5_predictions}")
        logging.info(f"Top-5 准确率: {final_top5_accuracy:.4f}")
        logging.info("---")
    else:
        logging.warning("没有处理任何图像，无法计算准确率。请检查数据集路径和 val.txt 文件。")


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
        help="原始输入图像文件的路径 (例如: image.jpg, image.png)。在精度验证模式下不需要。"
    )
    parser.add_argument(
        "-d", "--dataset_root",
        type=str,
        help="ImageNet-1K val 数据集的根目录路径 (例如: /path/to/imagenet/val)。"
    )
    parser.add_argument(
        "-l", "--val_list_file",
        type=str,
        help="包含图像名称和标签的 val.txt 文件路径 (例如: /path/to/val.txt)。"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="如果设置，则进行精度验证模式。"
    )

    args = parser.parse_args()

    # 将日志级别设置为 INFO，以便看到进度条。如果需要详细的 DEBUG 信息，可以设置为 DEBUG。
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("使用了Python标准日志进行初始化 。")

    if args.validate:
        if not args.dataset_root or not args.val_list_file:
            parser.error("--validate 模式需要 --dataset_root 和 --val_list_file 参数。")
        validate_accuracy(args.model_file, args.dataset_root, args.val_list_file)
    else:
        if not args.image_file:
            parser.error("非 --validate 模式需要 --image_file 参数。")
        logging.warning("在非验证模式下，image_file 参数是必需的。")
        logging.info("正在执行单张图像推理...")
        if not os.path.exists(args.model_file):
            logging.error(f"错误：模型文件未找到: {args.model_file}")
        else:
            try:
                sess_single = HBRuntime(args.model_file)
                top_results = run_model_inference(args.image_file, sess_single, top_k_for_output=5)
                if top_results:
                    logging.info("The input picture is classified to be:")
                    for label_item, prob_item in top_results:
                        try:
                            class_name = IMAGENET_VAL_CLASSES[label_item]
                        except (TypeError, IndexError):
                            class_name = f"未知标签索引_{label_item}"
                        logging.info(
                            f"      label {label_item:3d}, prob {prob_item:.5f}, class {class_name}")
                else:
                    logging.warning("单张图像推理未能生成有效分类结果。")
            except Exception as e:
                logging.error(f"错误：单张图像推理加载模型失败: {e}")
