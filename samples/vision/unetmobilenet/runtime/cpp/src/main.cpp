/*
 * Copyright (c) 2025 D-Robotics Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file main.cpp
 * @brief Provide a command-line demo application for UnetMobileNet semantic segmentation.
 *
 * This file implements the executable entry point that wires together the full
 * UnetMobileNet sample workflow. It parses command-line flags (model path,
 * input image, alpha blending factor), initializes the UnetMobileNet runtime,
 * and runs the standard pipeline: preprocess -> inference -> postprocess.
 *
 * The program loads a BGR test image, prepares model input tensors (direct resize
 * + BGR->NV12), executes BPU inference, decodes the segmentation mask (argmax),
 * colorizes the mask, alpha-blends it with the original image, and saves the result.
 *
 * @see unetmobilenet.hpp
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "unetmobilenet.hpp"

// ---------------------------------------------------------------------------
// Platform-dependent default model path
// ---------------------------------------------------------------------------
#ifdef SOC_S600
#define DEFAULT_MODEL_PATH "/opt/hobot/model/s600/basic/unet_mobilenet_1024x2048_nv12.hbm"
#else
#define DEFAULT_MODEL_PATH "/opt/hobot/model/s100/basic/unet_mobilenet_1024x2048_nv12.hbm"
#endif

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------
DEFINE_string(model_path, DEFAULT_MODEL_PATH,
              "Path to BPU Quantized *.hbm model file.");
DEFINE_string(test_img, "../../../test_data/segmentation.png",
              "Path to load the test image.");
DEFINE_double(alpha_f, 0.75,
              "Alpha blending factor. 0.0 = only mask, 1.0 = only original image.");

/**
 * @brief Main entry for running UnetMobileNet semantic segmentation on a single image.
 *
 * Pipeline:
 * 1) Parse CLI flags (model / image / alpha).
 * 2) Initialize UnetMobileNet runtime with quantized *.hbm model.
 * 3) Load input image (BGR) and record original resolution.
 * 4) Preprocess -> Inference -> Postprocess (argmax mask).
 * 5) Colorize mask, alpha-blend with original, and save result image.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Array of command-line argument strings.
 * @return int Process exit code (0 on success).
 */
int main(int argc, char** argv)
{
    // Parse command-line flags
    int32_t ret;
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    // Initialize UnetMobileNet runtime
    UnetMobileNet unet;
    ret = unet.init(FLAGS_model_path.c_str());
    if (ret != 0) {
        fprintf(stderr, "UnetMobileNet init failed (ret=%d)\n", ret);
        return ret;
    }

    // Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);
    const int orig_w = image.cols;
    const int orig_h = image.rows;

    // 1) Preprocess: direct resize + BGR->NV12
    ret = pre_process(unet.input_tensors, image, unet.input_w, unet.input_h);
    if (ret != 0) {
        fprintf(stderr, "pre_process failed (ret=%d)\n", ret);
        return ret;
    }

    // 2) Inference: BPU execution
    ret = infer(unet.output_tensors, unet.input_tensors, unet.dnn_handle);
    if (ret != 0) {
        fprintf(stderr, "infer failed (ret=%d)\n", ret);
        return ret;
    }

    // 3) Postprocess: argmax -> class ID mask (CV_32S)
    SegmentationMask seg_result = post_process(unet.output_tensors, orig_w, orig_h, unet.input_w, unet.input_h);

    // Colorize class IDs to BGR visualization image
    cv::Mat parsing_img = colorize_mask(seg_result.class_ids, rdk_colors);

    // Alpha blend segmentation with original image
    cv::Mat blended_img;
    cv::addWeighted(image, FLAGS_alpha_f, parsing_img, 1.0 - FLAGS_alpha_f, 0.0, blended_img);

    // Save result
    cv::imwrite("result.jpg", blended_img);
    std::cout << "[Saved] Result saved to: result.jpg" << std::endl;

    return 0;
}
