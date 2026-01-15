/*
 * Copyright (c) 2025, XiangshunZhao D-Robotics.
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
 * @file main.cc
 * @brief Provide a command-line demo application for running YOLOv5x object detection.
 *
 *        This file implements the executable entry point that wires together the
 *        full YOLOv5x sample workflow. It parses command-line flags (model path,
 *        input image, label file, and thresholds), initializes the YOLOv5x runtime,
 *        and runs the standard pipeline: preprocess -> inference -> postprocess.
 *
 *        The program loads a BGR test image, prepares model input tensors (including
 *        resize/format conversion as required by the compiled model), executes BPU
 *        inference via the runtime APIs, decodes and filters detections (score/NMS),
 *        draws visualization overlays, and saves the final result image to disk.
 *
 *        This sample is intended as a minimal reference for integrating the YOLOv5x
 *        wrapper and related utilities into an application with clear, end-to-end
 *        execution flow and configurable runtime parameters.
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "yolov5.hpp"

DEFINE_string(model_path, "/opt/hobot/model/s100/basic/yolov5x_672x672_nv12.hbm",
              "Path to BPU Quantized *.hbm model file");
DEFINE_string(test_img, "/app/res/assets/kite.jpg",
              "Path to load the test image.");
DEFINE_string(label_file, "/app/res/labels/coco_classes.names",
              "Path to load class name list (one label per line, e.g., COCO).");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres, 0.45, "IoU threshold for Non-Maximum Suppression.");

/**
 * @brief Main entry for running YOLOv5x object detection.
 *
 * Pipeline:
 * 1) Parse CLI flags (model / image / labels / thresholds)
 * 2) Build runtime config (score/NMS thresholds)
 * 3) Initialize YOLOv5x runtime with quantized *.hbm model
 * 4) Load input image (BGR) and record original resolution
 * 5) Preprocess -> Inference -> Postprocess
 * 6) Draw detections and save result image
 *
 * @param argc [in] Number of command-line arguments.
 * @param argv [in] Array of command-line argument strings.
 * @return int [out] Process exit code (0 on success).
 */
int main(int argc, char **argv)
{
    // Parse command line arguments
    int32_t ret;
    gflags::SetUsageMessage(argv[0]);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << gflags::GetArgv() << std::endl;

    Yolov5Config cfg;
    cfg.score_thresh = FLAGS_score_thres;
    cfg.nms_thresh   = FLAGS_nms_thres;

    // Create runtime wrapper and initialize model
    YOLOv5x yolov5x = YOLOv5x();

    yolov5x.init(FLAGS_model_path.c_str());

    // Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);

    // original image size
    const int orig_w = image.cols;
    const int orig_h = image.rows;

    // 1) Preprocess
    ret = pre_process(yolov5x.input_tensors, image,
                      yolov5x.input_w, yolov5x.input_h);

    if (ret != 0) {
        printf("pre process failed!");
        return ret;
    }

    // 2) Inference
    ret = infer(yolov5x.output_tensors,
                yolov5x.input_tensors,
                yolov5x.dnn_handle);
    if (ret != 0) {
        printf("Inference failed!");
        return ret;
    }

    // 3) Postprocess
    std::vector<Detection> results;

    post_process(results,
                 yolov5x.output_tensors, cfg, orig_w, orig_h,
                 yolov5x.input_w, yolov5x.input_h);

    // Load class names and render detections
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    draw_boxes(image, results, class_names, rdk_colors);

    // Persist output
    cv::imwrite("result.jpg", image);
    std::cout << "[Saved] Result saved to: result.jpg" << std::endl;

    return 0;
}
