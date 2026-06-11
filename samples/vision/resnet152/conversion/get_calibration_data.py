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

A generic script to generate model calibration data.
This script loads raw images, performs user-defined preprocessing, and saves the calibration data as .npy format.
"""

# This example uses skimage
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
    Defines the list of transformers used for calibration data preprocessing.
    According to the toolchain requirements, these transformers will include all preprocessing steps
    consistent with those before model inference, including mean subtraction and scale normalization.
    """
    transformers = [
        PaddedCenterCropTransformer(224),
        ResizeTransformer(
            target_size=(224, 224),
            mode='skimage',
            method=3
        ),
        HWC2CHWTransformer(),
        ScaleTransformer(scale_value=255.0),
        MeanTransformer(means=np.array([123.675, 116.28, 103.53])),
        ScaleTransformer(scale_value=0.017)
    ]
    return transformers

def convert_image(src_image_path, dst_file_path, transformers):
    """
    Reads a single source image, applies transformers, and saves the result as a .npy file.
    Saves the data type as float32.
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
    output_calib_dir = './calibration_data_rgb/'

    os.makedirs(output_calib_dir, exist_ok=True)

    src_images_paths = sorted(glob.glob(os.path.join(src_image_dir, 'ILSVRC2012_val_*.JPEG')))

    if not src_images_paths:
        print(f"Error: No images matching 'ILSVRC2012_val_*.JPEG' found in folder '{src_image_dir}'.")
        exit()

    num_expected_images = 100
    if len(src_images_paths) < num_expected_images:
        print(f"Warning: Expected {num_expected_images} images, but only found {len(src_images_paths)} in '{src_image_dir}'.")
    elif len(src_images_paths) > num_expected_images:
        print(f"Note: Found {len(src_images_paths)} images in '{src_image_dir}', the first {num_expected_images} will be used.")
        src_images_paths = src_images_paths[:num_expected_images]

    dst_files_paths = []
    for src_path in src_images_paths:
        base_name = os.path.basename(src_path)
        name_without_ext = os.path.splitext(base_name)[0]
        dst_files_paths.append(os.path.join(output_calib_dir, f"{name_without_ext}.npy"))

    active_transformers = data_transformer()

    print(f"Start processing {len(src_images_paths)} images...")
    for src_path, dst_path in zip(src_images_paths, dst_files_paths):
        convert_image(src_path, dst_path, active_transformers)

    print(f"\nSuccessfully generated {len(dst_files_paths)} calibration data files in folder: {output_calib_dir}")
