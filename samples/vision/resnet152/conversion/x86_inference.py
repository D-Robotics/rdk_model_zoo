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

A generic model inference script, supporting ONNX, HBIR (.bc), and HBM formats.
This script loads raw images, performs user-defined preprocessing, executes inference
using Horizon's HBRuntime, and outputs the classification results to the console based on the
post-processing function.
"""

import numpy as np
import argparse
import os
from typing import Iterable
import logging

# Horizon AI toolchain library
IMAGENET_VAL_CLASSES = None

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
    # Directly import the ImageNet validation set class name list
    from horizon_tc_ui.data.imagenet_val import imagenet_val as IMAGENET_VAL_CLASSES

except ImportError:
    logging.warning("Warning: Unable to fully import horizon_tc_ui related modules.")
    logging.warning("Ensure the Horizon AI toolchain is installed correctly, and PYTHONPATH is set.")
    logging.warning("If imagenet_val is not available, related functionality will be limited.")


def onnx_transformer():
    """
    Defines the list of transformers used for ONNX image preprocessing.
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
        MeanTransformer(means=np.array([123.68, 116.78, 103.94])),
        ScaleTransformer(scale_value=0.017)
    ]
    return transformers

def quantied_transformers():
    """
    Defines the list of transformers used for .bc/.hbm image preprocessing.
    """
    transformers = [
        PaddedCenterCropTransformer(224),
        ResizeTransformer(target_size=(224, 224),
                          mode='skimage',
                          method=3),
        ScaleTransformer(scale_value=255),
        RGB2NV12Transformer(data_format="HWC")
    ]
    return transformers

def postprocess_classification_output(model_output: list, top_k: int = 5) -> list:
    """
    Postprocesses the classification model outputs to extract Top-K results.
    Directly uses imagenet_val imported from horizon_tc_ui.data.imagenet_val.
    """
    global IMAGENET_VAL_CLASSES

    if not model_output or not isinstance(model_output[0], np.ndarray):
        logging.error("Postprocess error: Invalid model output format.")
        return []

    scores = np.squeeze(model_output[0])

    if scores.ndim == 0:
        logging.error(f"Postprocess error: Model output becomes scalar after squeeze, original shape: {model_output[0].shape}")
        return []
    if scores.ndim != 1:
        logging.warning(f"Postprocess warning: Expected 1D scores array after squeeze, but got {scores.ndim}D. Trying to flatten.")
        if scores.size == model_output[0].shape[-1] or (IMAGENET_VAL_CLASSES and scores.size == len(IMAGENET_VAL_CLASSES)):
            scores = scores.flatten()
        else:
            logging.error(f"Postprocess error: Cannot process scores (shape: {scores.shape}) into a 1D array.")
            return []

    # Apply Softmax to convert logits to probabilities
    exp_logits = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
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
            class_name = f"unknown_label_{label_index}"
            logging.warning(f"Unable to get class name for index {label_index}: {e_label}")
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
    Runs inference on a single image and returns a list of Top-K predicted class indices, probabilities, and class names.
    """
    if not os.path.exists(image_path):
        logging.error(f"Error: Image file not found: {image_path}")
        return None

    # Get model type from sess to determine which transformer to use
    if isinstance(sess.sess, HB_ONNXRuntime):
        active_transformers = onnx_transformer()
    else:
        active_transformers = quantied_transformers()

    try:
        data = SingleImageDataLoader(active_transformers,
                                     image_path,
                                     imread_mode='skimage')
    except Exception as e:
        logging.error(f"Error: Failed to load image and preprocess: {e}")
        return None

    try:
        input_names = sess.input_names
        output_names = sess.output_names

        if not input_names:
            logging.error("Error: Failed to get input node names from the model.")
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
            return top_results
        else:
            logging.error(f"Image {os.path.basename(image_path)} model inference returned no output.")
            return None

    except Exception as e:
        logging.error(f"Error: An error occurred during inference or output processing for image {os.path.basename(image_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Import tqdm library for progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm library is not installed. Progress bar will be unavailable. Please run 'pip install tqdm' to install.")
    tqdm = lambda x: x  # Define a dummy function so the code runs without tqdm

def validate_accuracy(model_path: str, dataset_root: str, val_list_file: str):
    """
    Calculates Top-1 and Top-5 accuracy on the ImageNet-1K validation set, displaying a progress bar.
    """

    if not os.path.exists(model_path):
        logging.error(f"Error: Model file not found: {model_path}")
        return

    if not os.path.exists(val_list_file):
        logging.error(f"Error: val.txt file not found: {val_list_file}")
        return

    logging.info(f"[*] Loading model: {model_path}")
    try:
        sess = HBRuntime(model_path)
        logging.info("    Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error: Failed to load model: {e}")
        return

    correct_top1_predictions = 0
    correct_top5_predictions = 0
    total_images = 0

    logging.info(f"[*] Reading validation list file: {val_list_file}")
    with open(val_list_file, 'r') as f:
        lines = f.readlines()

    logging.info(f"[*] Starting accuracy validation on {len(lines)} images (calculating Top-1 and Top-5 accuracy)...")

    # Wrap lines iterator with tqdm to show progress bar
    for line_num, line in enumerate(tqdm(lines, desc="Validation Progress", unit="images")):
        line = line.strip()
        if not line:
            continue

        try:
            image_name, true_label = line.split(' ')
            true_label = int(true_label)
        except ValueError:
            logging.warning(f"Warning: Skipping malformed line in val.txt: {line}")
            continue

        image_path = os.path.join(dataset_root, image_name)
        if not os.path.exists(image_path):
            logging.warning(f"Warning: Image file not found, skipping: {image_path}")
            continue

        # Get Top-5 prediction results
        predicted_results_top5 = run_model_inference(image_path, sess, top_k_for_output=5)
        total_images += 1

        if predicted_results_top5 is not None and len(predicted_results_top5) > 0:
            # Extract predicted label indices
            predicted_labels = [res[0] for res in predicted_results_top5]

            # Check Top-1 accuracy
            if predicted_labels[0] == true_label:
                correct_top1_predictions += 1

            # Check Top-5 accuracy
            if true_label in predicted_labels:
                correct_top5_predictions += 1

        if (line_num + 1) % 100 == 0:
            current_top1_accuracy = correct_top1_predictions / total_images if total_images > 0 else 0
            current_top5_accuracy = correct_top5_predictions / total_images if total_images > 0 else 0
            logging.info(f"Processed {line_num + 1}/{len(lines)} images. Current Top-1 accuracy: {current_top1_accuracy:.4f}, Top-5 accuracy: {current_top5_accuracy:.4f}")

    if total_images > 0:
        final_top1_accuracy = correct_top1_predictions / total_images
        final_top5_accuracy = correct_top5_predictions / total_images
        logging.info("---")
        logging.info(f"Accuracy validation completed!")
        logging.info(f"Total images: {total_images}")
        logging.info(f"Top-1 correct predictions: {correct_top1_predictions}")
        logging.info(f"Top-1 accuracy: {final_top1_accuracy:.4f}")
        logging.info(f"Top-5 correct predictions: {correct_top5_predictions}")
        logging.info(f"Top-5 accuracy: {final_top5_accuracy:.4f}")
        logging.info("---")
    else:
        logging.warning("No images were processed, unable to calculate accuracy. Please check dataset path and val.txt file.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="RDK model inference script (supporting ONNX, HBIR/.bc), taking raw images directly for preprocessing and inference.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--model_file",
        type=str,
        required=True,
        help="Path to the model file (e.g., model.onnx, model.bc, model.hbm)"
    )
    parser.add_argument(
        "-i", "--image_file",
        type=str,
        help="Path to the raw input image file (e.g., image.jpg, image.png). Not required in accuracy validation mode."
    )
    parser.add_argument(
        "-d", "--dataset_root",
        type=str,
        help="Path to the ImageNet-1K val dataset root directory (e.g., /path/to/imagenet/val)."
    )
    parser.add_argument(
        "-l", "--val_list_file",
        type=str,
        help="Path to the val.txt file containing image names and labels (e.g., /path/to/val.txt)."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="If set, perform accuracy validation mode."
    )

    args = parser.parse_args()

    # Set log level to INFO to see the progress bar.
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Initialized with standard Python logging.")

    if args.validate:
        if not args.dataset_root or not args.val_list_file:
            parser.error("--validate mode requires --dataset_root and --val_list_file arguments.")
        validate_accuracy(args.model_file, args.dataset_root, args.val_list_file)
    else:
        if not args.image_file:
            parser.error("Non--validate mode requires --image_file argument.")
        logging.warning("The image_file argument is required in non-validation mode.")
        logging.info("Running single image inference...")
        if not os.path.exists(args.model_file):
            logging.error(f"Error: Model file not found: {args.model_file}")
        else:
            try:
                sess_single = HBRuntime(args.model_file)
                top_results = run_model_inference(args.image_file, sess_single, top_k_for_output=5)
                if top_results:
                    logging.info("The input picture is classified to be:")
                    for label_item, prob_item, class_name in top_results:
                        logging.info(
                            f"      label {label_item:3d}, prob {prob_item:.5f}, class {class_name}")
                else:
                    logging.warning("Single image inference failed to generate valid classification results.")
            except Exception as e:
                logging.error(f"Error: failed to load model for single image inference: {e}")
