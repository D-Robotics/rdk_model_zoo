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
 * @brief MobileNetV2 classification sample entry.
 *
 * This file implements a standalone sample program that demonstrates how to
 * run MobileNetV2 classification using the D-Robotics BPU runtime. It parses
 * command-line flags, loads a quantized HBM model and a test image, performs
 * preprocessing and inference, and prints top-K classification results.
 *
 * @note This is a sample application intended for demonstration and evaluation.
 */

#include <iostream>
#include <map>
#include <vector>

#include "gflags/gflags.h"
#include "file_io.hpp"
#include "visualize.hpp"
#include "mobilenetv2.hpp"

#ifdef SOC_S600
DEFINE_string(model_path, "/opt/hobot/model/s600/basic/mobilenetv2_224x224_nv12.hbm",
              "Path to BPU Quantized *.hbm model file");
#else
DEFINE_string(model_path, "/opt/hobot/model/s100/basic/mobilenetv2_224x224_nv12.hbm",
              "Path to BPU Quantized *.hbm model file");
#endif

DEFINE_string(test_img, "../../../test_data/zebra_cls.jpg",
              "Path to load the test image.");
DEFINE_string(label_file, "../../../test_data/imagenet1000_labels.txt",
              "Path to load ImageNet label mapping file.");
DEFINE_int32(top_k, 5, "Top k classes, 5 by default");

/**
 * @brief Run MobileNetV2 classification sample.
 *
 * The program parses command-line flags, loads the model and test image,
 * runs preprocessing and inference, then prints the top-K classification results.
 *
 * @param[in] argc Argument count.
 * @param[in] argv Argument vector.
 * @retval 0   Success.
 * @retval <0  Failure.
 */
int main(int argc, char **argv)
{
    int32_t ret = 0;

    // Parse command-line flags
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;  // Dump parsed argv for debugging

    // Load model and initialize runtime
    MobileNetV2 mobilenetv2;
    mobilenetv2.init(FLAGS_model_path.c_str());

    // Load test image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);

    // Preprocess image into model input tensors
    ret = pre_process(mobilenetv2.input_tensors,
                      mobilenetv2.input_w, mobilenetv2.input_h,
                      image);
    if (ret != 0) {
        printf("pre process failed!");
        return ret;
    }

    // Run inference (blocking)
    ret = infer(mobilenetv2.output_tensors,
                mobilenetv2.input_tensors,
                mobilenetv2.dnn_handle);
    if (ret != 0) {
        printf("Inference failed!");
        return ret;
    }

    // Postprocess logits to top-K classification results
    MobileNetV2Config cfg;
    std::vector<Classification> top_k_cls;
    post_process(top_k_cls, mobilenetv2.output_tensors, cfg, FLAGS_top_k);

    // Load labels and print results
    auto labels = load_linewise_labels(FLAGS_label_file);
    print_topk_results(top_k_cls, labels);

    return 0;
}
