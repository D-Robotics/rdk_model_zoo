/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2024-2025, WuChao && MaChao D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK板端运行
// Attention: This program runs on RDK board.

// ============================================================================
// Configuration Parameters
// ============================================================================

// D-Robotics *.bin 模型路径
// Path to D-Robotics *.bin model
#define MODEL_PATH "source/reference_bin_models/cls/yolo11n_cls_bayese_224x224_nv12.bin"

// 测试图片路径
// Path to test image
#define TEST_IMG_PATH "../../../../../datasets/imagenet/asset/zebra_cls.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
// Preprocessing method: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// Top K 结果数量
// Number of top K results to display
#define TOP_K 5

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>

// OpenCV
#include <opencv2/opencv.hpp>

// RDK BPU libDNN API (stack-portable layer; pulls in the X5 hbSys or the
// S-series UCP headers itself)
#include "common/dnn_io.h"
#include "common/nv12_geometry.h"
#include "common/classification_binding.h"
#include "common/dnn_resources.h"
#include "common/imagenet_labels.h"

// ============================================================================
// Macros
// ============================================================================

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

/**
 * @brief Preprocess image with letterbox or resize
 */
cv::Mat preprocess_image(const cv::Mat& img, int input_h, int input_w,
                         float& x_scale, float& y_scale,
                         int& x_shift, int& y_shift) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;

    if (PREPROCESS_TYPE == LETTERBOX_TYPE) {
        // Letterbox preprocessing
        x_scale = std::min(1.0f * input_h / img.rows, 1.0f * input_w / img.cols);
        y_scale = x_scale;

        if (x_scale <= 0 || y_scale <= 0) {
            throw std::runtime_error("Invalid scale factor");
        }

        int new_w = static_cast<int>(img.cols * x_scale);
        int new_h = static_cast<int>(img.rows * y_scale);

        x_shift = (input_w - new_w) / 2;
        y_shift = (input_h - new_h) / 2;
        int x_other = input_w - new_w - x_shift;
        int y_other = input_h - new_h - y_shift;

        cv::resize(img, result, cv::Size(new_w, new_h));
        cv::copyMakeBorder(result, result, y_shift, y_other, x_shift, x_other,
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (LetterBox) time", duration);

    } else if (PREPROCESS_TYPE == RESIZE_TYPE) {
        // Resize preprocessing
        cv::resize(img, result, cv::Size(input_w, input_h));

        x_scale = 1.0f * input_w / img.cols;
        y_scale = 1.0f * input_h / img.rows;
        x_shift = 0;
        y_shift = 0;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        LOG_TIME("Preprocess (Resize) time", duration);
    }

    LOG_INFO("Scale: x=" << x_scale << ", y=" << y_scale);
    LOG_INFO("Shift: x=" << x_shift << ", y=" << y_shift);

    return result;
}

// ============================================================================
// Main Function
// ============================================================================

int run(int argc, char** argv) {
    LOG_INFO("=== Ultralytics YOLO Classify Demo (C++) ===");
    LOG_INFO("OpenCV Version: " << CV_VERSION);

    // ========================================================================
    // 0. Parse command line arguments
    // ========================================================================

    std::string model_path = MODEL_PATH;
    std::string test_img_path = TEST_IMG_PATH;

    if (argc >= 2) model_path = argv[1];
    if (argc >= 3) test_img_path = argv[2];

    // ========================================================================
    // 1. Load BPU model
    // ========================================================================

    LOG_INFO("Loading model: " << model_path);
    auto start_time = std::chrono::high_resolution_clock::now();

    yolo::PackedModelOwner packed_model;
    auto& packed_dnn_handle=packed_model.handle;
    const char* model_file_name = model_path.c_str();
    CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "Failed to initialize model from file");

    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Load model time", load_duration);

    // ========================================================================
    // 2. Get model handle
    // ========================================================================

    const char** model_name_list=nullptr;
    int model_count = 0;
    CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "Failed to get model name list");

    if (model_count != 1 || model_name_list == nullptr || model_name_list[0] == nullptr) {
        LOG_ERROR("Expected exactly one named model");
        return -1;
    }
    const char* model_name = model_name_list[0];
    LOG_INFO("Model name: " << model_name);

    hbDNNHandle_t dnn_handle;
    CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "Failed to get model handle");

    // ========================================================================
    // 3. Check model input
    // ========================================================================

    int32_t input_h = 0;
    int32_t input_w = 0;
    yolo::InputPlan input_plan;
    {
        std::string protocol_error;
        input_plan = yolo::probe_input_protocol(dnn_handle, &protocol_error);
        if (input_plan.protocol == yolo::InputProtocol::kUnknown) {
            LOG_ERROR("Unsupported model input: " << protocol_error);
            return -1;
        }
        input_h = input_plan.input_h;
        input_w = input_plan.input_w;
        LOG_INFO("Input: "
                 << (input_plan.protocol == yolo::InputProtocol::kPackedNv12
                         ? "packed NV12 "
                         : "split Y/UV NV12 ")
                 << input_w << "x" << input_h);
    }

    // ========================================================================
    // 4. Check model outputs
    // ========================================================================

    int32_t output_count = 0;
    CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "Failed to get output count");

    if (output_count != 1) {
        LOG_ERROR("Classification model should have exactly 1 output, but has " << output_count);
        return -1;
    }

    hbDNNTensorProperties output_properties{};
    CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, 0),
        "Failed to get output tensor properties");

    const auto output_plan=yolo::bind_classification(output_properties);
    LOG_INFO("Output: 1000 FLOAT32 logits, class byte stride=" << output_plan.class_stride);
    if (IMAGENET_CLASSES.size()!=1000) throw std::runtime_error("ImageNet labels must contain 1000 entries");

    // ========================================================================
    // 5. Load and preprocess image
    // ========================================================================

    LOG_INFO("Loading image: " << test_img_path);
    cv::Mat img = cv::imread(test_img_path);
    if (img.empty()) {
        LOG_ERROR("Failed to load image: " << test_img_path);
        return -1;
    }
    LOG_INFO("Image size: " << img.cols << "x" << img.rows);

    // Preprocess image
    float x_scale, y_scale;
    int x_shift, y_shift;
    cv::Mat preprocessed = preprocess_image(img, input_h, input_w,
                                           x_scale, y_scale, x_shift, y_shift);

    // Convert to I420 (shared source for both input protocols)
    cv::Mat yuv_mat;
    cv::cvtColor(preprocessed, yuv_mat, cv::COLOR_BGR2YUV_I420);
    const uint8_t* i420 = yuv_mat.ptr<uint8_t>();

    // ========================================================================
    // 6. Prepare input tensor(s)
    // ========================================================================

    yolo::Nv12Input inputs;
    if (!inputs.allocate(dnn_handle, input_plan)) {
        LOG_ERROR("Failed to allocate model input tensors");
        return -1;
    }
    if (!inputs.upload(input_plan, i420)) {
        LOG_ERROR("Failed to upload the preprocessed frame");
        return -1;
    }

    // ========================================================================
    // 7. Prepare output tensor
    // ========================================================================

    yolo::OutputTensorOwner output_owner;
    CHECK_SUCCESS(output_owner.allocate(output_properties), "Failed to allocate output tensor");
    hbDNNTensor* output=&output_owner.tensor;

    // ========================================================================
    // 8. Run inference
    // ========================================================================

    LOG_INFO("Running inference...");
    start_time = std::chrono::high_resolution_clock::now();

    CHECK_SUCCESS(
        yolo::infer_sync(output, inputs.tensors(), inputs.input_count(), dnn_handle),
        "Inference failed");

    auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("BPU inference time", infer_duration);

    // ========================================================================
    // 9. Post-process
    // ========================================================================

    LOG_INFO("Post-processing...");
    start_time = std::chrono::high_resolution_clock::now();

    CHECK_SUCCESS(YOLO_SYS_FLUSH(YOLO_SYS_MEM(output[0]), HB_SYS_MEM_CACHE_INVALIDATE),
                  "Failed to invalidate output cache");
    const auto results=yolo::classification_topk(YOLO_SYS_MEM(output[0])->virAddr,
        static_cast<size_t>(output_properties.alignedByteSize),output_plan,TOP_K);

    auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start_time).count() / 1000.0;
    LOG_TIME("Post-processing time", post_duration);

    // ========================================================================
    // 10. Display results
    // ========================================================================

    LOG_INFO("Classification results:");
    LOG_INFO("Image: " << test_img_path);
    std::cout << std::endl;

    for (size_t i = 0; i < results.size(); i++) {
        const auto& res = results[i];
        std::cout << "\033[1;32m"
                  << "TOP" << (i + 1) << " -> "
                  << "id: " << res.id << ", "
                  << "score: " << std::fixed << std::setprecision(3) << res.probability << ", "
                  << "name: " << IMAGENET_CLASSES.at(res.id)
                  << "\033[0m" << std::endl;
    }

    // ========================================================================
    // 11. Cleanup
    // ========================================================================

    // Tensor/input/model owners release resources on every return or exception.

    LOG_INFO("=== Demo completed successfully ===");
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc,argv); }
    catch (const std::exception& error) { LOG_ERROR(error.what()); return 1; }
}
