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
 * @brief Provide a command-line demo application for YOLO11 object detection.
 *
 * This file implements the executable entry point that wires together the
 * full YOLO11 sample workflow. It parses command-line flags (model path,
 * input image, label file, and thresholds), initializes the YOLO11 runtime,
 * and runs the standard pipeline: preprocess -> inference -> postprocess.
 *
 * The program loads a BGR test image, prepares model input tensors (letterbox
 * resize + BGR->NV12), executes BPU inference, decodes and filters detections
 * (DFL + score + NMS), draws visualization overlays, and saves the result image
 * to disk.
 *
 * @see yolo11.hpp
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "yolo11.hpp"

// ---------------------------------------------------------------------------
// Platform-dependent default model path
// ---------------------------------------------------------------------------
#ifdef SOC_S600
#define DEFAULT_MODEL_PATH "/opt/hobot/model/s600/basic/yolo11n_detect_nashe_640x640_nv12.hbm"
#else
#define DEFAULT_MODEL_PATH "/opt/hobot/model/s100/basic/yolo11n_detect_nashe_640x640_nv12.hbm"
#endif

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------
DEFINE_string(model_path, DEFAULT_MODEL_PATH,
              "Path to BPU Quantized *.hbm model file.");
DEFINE_string(test_img, "../../../test_data/kite.jpg",
              "Path to load the test image.");
DEFINE_string(label_file, "../../../test_data/coco_classes.names",
              "Path to load class name list (one label per line, e.g., COCO).");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.45,   "IoU threshold for Non-Maximum Suppression.");

/**
 * @brief Main entry for running YOLO11 object detection on a single image.
 *
 * Pipeline:
 * 1) Parse CLI flags (model / image / labels / thresholds).
 * 2) Build runtime config (score/NMS thresholds).
 * 3) Initialize YOLO11 runtime with quantized *.hbm model.
 * 4) Load input image (BGR) and record original resolution.
 * 5) Preprocess -> Inference -> Postprocess.
 * 6) Draw detections and save result image.
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

    // Build inference configuration
    Yolo11Config cfg;
    cfg.score_thresh = static_cast<float>(FLAGS_score_thres);
    cfg.nms_thresh   = static_cast<float>(FLAGS_nms_thres);

    // Initialize YOLO11 runtime
    YOLO11 yolo11;
    ret = yolo11.init(FLAGS_model_path.c_str());
    if (ret != 0) {
        fprintf(stderr, "YOLO11 init failed (ret=%d)\n", ret);
        return ret;
    }

    // Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);
    const int orig_w = image.cols;
    const int orig_h = image.rows;

    // 1) Preprocess: letterbox resize + BGR->NV12
    ret = pre_process(yolo11.input_tensors, image,
                      yolo11.input_w, yolo11.input_h);
    if (ret != 0) {
        fprintf(stderr, "pre_process failed (ret=%d)\n", ret);
        return ret;
    }

    // 2) Inference: BPU execution
    ret = infer(yolo11.output_tensors, yolo11.input_tensors, yolo11.dnn_handle);
    if (ret != 0) {
        fprintf(stderr, "infer failed (ret=%d)\n", ret);
        return ret;
    }

    // 3) Postprocess: DFL decode + NMS + rescale
    std::vector<Detection> results;
    post_process(results, yolo11.output_tensors, cfg,
                 orig_w, orig_h, yolo11.input_w, yolo11.input_h);

    // Draw detections and save result image
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    draw_boxes(image, results, class_names, rdk_colors);

    cv::imwrite("result.jpg", image);
    std::cout << "[Saved] Result saved to: result.jpg" << std::endl;

    return 0;
}
