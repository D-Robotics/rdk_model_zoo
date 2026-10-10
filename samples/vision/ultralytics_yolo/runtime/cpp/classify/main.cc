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
#include <memory>
#include <string>
#include <numeric>

// OpenCV
#include <opencv2/opencv.hpp>

// RDK BPU libDNN API (stack-portable layer; pulls in the X5 hbSys or the
// S-series UCP headers itself)
#include "common/classification_binding.h"
#include "common/dnn_resources.h"
#include "common/imagenet_labels.h"
#include "common/task_benchmark.h"
#include "common/task_session.h"

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

// ============================================================================
// Classification Runtime
// ============================================================================

// One model context, input tensors and the logits output. Each benchmark
// stream owns its own runtime; `run` leaves the Top-K result for reporting.
class ClassifyRuntime {
public:
    explicit ClassifyRuntime(const std::string& model_path) {
        session_.initialize(model_path);
        int32_t output_count = 0;
        if (hbDNNGetOutputCount(&output_count, session_.model()) != 0)
            throw std::runtime_error("Failed to get output count");
        if (output_count != 1)
            throw std::runtime_error("Classification model should have exactly 1 output, but has " +
                                     std::to_string(output_count));
        hbDNNTensorProperties properties{};
        if (hbDNNGetOutputTensorProperties(&properties, session_.model(), 0) != 0)
            throw std::runtime_error("Failed to get output tensor properties");
        plan_ = yolo::bind_classification(properties);
        if (output_.allocate(properties) != 0)
            throw std::runtime_error("Failed to allocate output tensor");
    }

    int input_h() const { return session_.input_h(); }
    int input_w() const { return session_.input_w(); }
    const char* implementation() const { return "native_cpp_yolo_classify"; }
    size_t class_stride() const { return plan_.class_stride; }
    const std::vector<yolo::ClassificationScore>& results() const { return results_; }

    // Timing starts with the in-memory BGR image and ends with Top-K results.
    size_t run(const cv::Mat& image, int resize_type, yolo::StageTiming* timing) {
        const auto start = std::chrono::steady_clock::now();
        yolo::ImageTransform transform;
        const cv::Mat resized = yolo::preprocess_image(image, input_h(), input_w(),
                                                       resize_type, &transform);
        cv::Mat i420;
        cv::cvtColor(resized, i420, cv::COLOR_BGR2YUV_I420);
        session_.upload(i420.ptr<uint8_t>());
        const auto preprocessed = std::chrono::steady_clock::now();
        session_.infer(&output_.tensor);
        const auto inferred = std::chrono::steady_clock::now();
        if (YOLO_SYS_FLUSH(YOLO_SYS_MEM(output_.tensor), HB_SYS_MEM_CACHE_INVALIDATE) != 0)
            throw std::runtime_error("Failed to invalidate output cache");
        results_ = yolo::classification_topk(
            YOLO_SYS_MEM(output_.tensor)->virAddr,
            static_cast<size_t>(output_.tensor.properties.alignedByteSize), plan_, TOP_K);
        const auto finished = std::chrono::steady_clock::now();
        timing->preprocess_ms = yolo::elapsed_ms(start, preprocessed);
        timing->runtime_ms = yolo::elapsed_ms(preprocessed, inferred);
        timing->postprocess_ms = yolo::elapsed_ms(inferred, finished);
        timing->end_to_end_ms = yolo::elapsed_ms(start, finished);
        return results_.size();
    }

private:
    // Declared first so the output is released before the model context.
    yolo::TaskSession session_;
    yolo::OutputTensorOwner output_;
    yolo::ClassificationPlan plan_{};
    std::vector<yolo::ClassificationScore> results_;
};

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
    try {
        if (IMAGENET_CLASSES.size() != 1000)
            throw std::runtime_error("ImageNet labels must contain 1000 entries");
        const yolo::TaskCommand command = yolo::parse_task_command(
            argc, argv, {MODEL_PATH, TEST_IMG_PATH}, yolo::BenchmarkOptions());
        if (command.help) {
            yolo::print_task_usage(argv[0], "MODEL IMAGE", false);
            return 0;
        }
        LOG_INFO("=== Ultralytics YOLO Classify Demo (C++) ===");
        LOG_INFO("Loading model: " << command.paths[0]);

        yolo::BenchmarkMeta meta;
        meta.output_kind = "topk_predictions";
        meta.timing_scope = "in_memory_bgr_to_topk_predictions";
        return yolo::run_task<ClassifyRuntime>(
            command, PREPROCESS_TYPE, meta,
            [&command]() {
                return std::unique_ptr<ClassifyRuntime>(new ClassifyRuntime(command.paths[0]));
            },
            [&command](ClassifyRuntime& runtime, const cv::Mat&, int) {
                LOG_INFO("Output: 1000 FLOAT32 logits, class byte stride=" << runtime.class_stride());
                LOG_INFO("Classification results:");
                LOG_INFO("Image: " << command.paths[1]);
                std::cout << std::endl;
                const auto& results = runtime.results();
                for (size_t i = 0; i < results.size(); i++) {
                    const auto& res = results[i];
                    std::cout << "\033[1;32m"
                              << "TOP" << (i + 1) << " -> "
                              << "id: " << res.id << ", "
                              << "score: " << std::fixed << std::setprecision(3) << res.probability << ", "
                              << "name: " << IMAGENET_CLASSES.at(res.id)
                              << "\033[0m" << std::endl;
                }
            });
    } catch (const std::exception& error) {
        LOG_ERROR(error.what());
        return 1;
    }
}
