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

一个通用的模型校准数据生成脚本。
该脚本会加载原始图像，执行用户定义的预处理并将校准数据保存为.npy格式。
"""

# 本示例使用skimage
import skimage
import skimage.io
import numpy as np
import os
import glob

from horizon_tc_ui.data.transformer import (PaddedCenterCropTransformer,
                                            HWC2CHWTransformer,
                                            MeanTransformer,
                                            RGB2BGRTransformer,
                                            ScaleTransformer,
                                            ResizeTransformer)

def data_transformer():
    """
    定义了用于校准数据预处理的转换器列表。
    根据工具链要求，这些转换器将包含所有与模型推理前一致的预处理步骤，
    包括均值和缩放归一化。
    """
    transformers = [
        PaddedCenterCropTransformer(224),
        ResizeTransformer(
            target_size=(224, 224),
            mode='skimage',
            method=3
        ),
        HWC2CHWTransformer(),
        RGB2BGRTransformer(),
        ScaleTransformer(scale_value=255.0),
        MeanTransformer(means=np.array([103.94, 116.78, 123.68])),
        ScaleTransformer(scale_value=0.017)
    ]
    return transformers

def convert_image(src_image_path, dst_file_path, transformers):
    """
    读取单个源图像，应用转换器，并将结果保存为.npy文件。
    保存数据类型 float32。
    """
    try:
        image_data = skimage.img_as_float(
            skimage.io.imread(src_image_path)).astype(np.float32)

        processed_image_list = [image_data]
        for trans in transformers:
            processed_image_list = trans(processed_image_list)

        final_image_data = processed_image_list[0].astype(np.float32)

        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        np.save(dst_file_path, final_image_data)
        print(f"Processed: {src_image_path} -> {dst_file_path} (saved as float32)")
    except Exception as e:
        print(f"Error processing {src_image_path}: {e}")


if __name__ == '__main__':

    src_image_dir = '../../../open_explorer/samples/ai_toolchain/horizon_model_convert_sample/01_common/calibration_data/imagenet/'
    output_calib_dir = './calibration_data_bgr/'

    os.makedirs(output_calib_dir, exist_ok=True)

    src_images_paths = sorted(glob.glob(os.path.join(src_image_dir, 'ILSVRC2012_val_*.JPEG')))

    if not src_images_paths:
        print(f"错误：在文件夹 '{src_image_dir}' 中没有找到 'ILSVRC2012_val_*.JPEG' 格式的图片。")
        exit()

    num_expected_images = 100
    if len(src_images_paths) < num_expected_images:
        print(f"警告：期望找到 {num_expected_images} 张图片，但在 '{src_image_dir}' 中只找到 {len(src_images_paths)} 张。")
    elif len(src_images_paths) > num_expected_images:
        print(f"提示：在 '{src_image_dir}' 中找到 {len(src_images_paths)} 张图片，将使用前 {num_expected_images} 张。")
        src_images_paths = src_images_paths[:num_expected_images]

    dst_files_paths = []
    for src_path in src_images_paths:
        base_name = os.path.basename(src_path)
        name_without_ext = os.path.splitext(base_name)[0]
        dst_files_paths.append(os.path.join(output_calib_dir, f"{name_without_ext}.npy"))

    active_transformers = data_transformer()

    print(f"开始处理 {len(src_images_paths)} 张图片...")
    for src_path, dst_path in zip(src_images_paths, dst_files_paths):
        convert_image(src_path, dst_path, active_transformers)

    print(f"\n成功生成 {len(dst_files_paths)} 个校准数据文件到文件夹: {output_calib_dir}")
