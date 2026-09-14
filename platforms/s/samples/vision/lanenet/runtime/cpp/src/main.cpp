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
 * @brief Provide a command-line demo application for LaneNet lane detection.
 *
 * This file implements the executable entry point for the LaneNet sample.
 * It parses command-line flags (model path, input image, output paths),
 * initializes the LaneNet runtime, and runs the standard pipeline:
 * preprocess -> inference -> postprocess.
 *
 * The program loads a BGR test image, prepares float32 NCHW model input tensors
 * (direct resize + BGR->RGB + ImageNet normalization), executes BPU inference,
 * decodes instance segmentation and binary segmentation outputs, and saves
 * both result images to disk.
 *
 * @note This model only supports RDK S100 platform.
 *
 * @see lanenet.hpp
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "lanenet.hpp"

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------
DEFINE_string(model_path, "/opt/hobot/model/s100/basic/lanenet256x512.hbm",
              "Path to BPU quantized *.hbm model file. Only S100 model is available.");
DEFINE_string(test_img, "../../../test_data/lane.jpg",
              "Path to load the test image.");
DEFINE_string(instance_save_path, "instance_pred.png",
              "Path to save the instance segmentation result image.");
DEFINE_string(binary_save_path, "binary_pred.png",
              "Path to save the binary segmentation result image.");

/**
 * @brief Main entry for running LaneNet lane detection on a single image.
 *
 * Pipeline:
 * 1) Parse CLI flags (model / image / output paths).
 * 2) Initialize LaneNet runtime with quantized *.hbm model.
 * 3) Load input image (BGR) from disk.
 * 4) Preprocess -> Inference -> Postprocess.
 * 5) Save instance and binary segmentation result images.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Array of command-line argument strings.
 * @return int Process exit code (0 on success).
 */
int main(int argc, char** argv)
{
    int32_t ret;
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    // Initialize LaneNet runtime
    LaneNet lanenet;
    ret = lanenet.init(FLAGS_model_path.c_str());
    if (ret != 0) {
        fprintf(stderr, "LaneNet init failed (ret=%d)\n", ret);
        return ret;
    }

    // Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);

    // 1) Preprocess: direct resize + BGR->RGB + ImageNet normalization -> float32 NCHW tensor
    ret = pre_process(lanenet.input_tensors, image, lanenet.input_w, lanenet.input_h);
    if (ret != 0) {
        fprintf(stderr, "pre_process failed (ret=%d)\n", ret);
        return ret;
    }

    // 2) Inference: BPU execution
    ret = infer(lanenet.output_tensors, lanenet.input_tensors, lanenet.dnn_handle);
    if (ret != 0) {
        fprintf(stderr, "infer failed (ret=%d)\n", ret);
        return ret;
    }

    // 3) Postprocess: decode instance and binary segmentation outputs
    cv::Mat instance_pred;
    cv::Mat binary_pred;
    post_process(instance_pred, binary_pred, lanenet.output_tensors,
                 lanenet.input_w, lanenet.input_h);

    // Save results
    cv::imwrite(FLAGS_instance_save_path, instance_pred);
    cv::imwrite(FLAGS_binary_save_path, binary_pred);
    std::cout << "[Saved] Instance segmentation result: " << FLAGS_instance_save_path << std::endl;
    std::cout << "[Saved] Binary segmentation result:   " << FLAGS_binary_save_path   << std::endl;

    return 0;
}
