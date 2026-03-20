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
 * @brief Provide a command-line demo application for YOLO11-Seg instance segmentation.
 *
 * This file implements the executable entry point that wires together the full
 * YOLO11-Seg sample workflow. It parses command-line flags, initializes the
 * YOLO11Seg runtime, and runs the standard pipeline:
 *   preprocess -> inference -> postprocess -> visualize -> save.
 *
 * @see yolo11seg.hpp
 */

#include <iostream>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "yolo11seg.hpp"

// ---------------------------------------------------------------------------
// Platform-dependent default model path
// ---------------------------------------------------------------------------
#ifdef SOC_S600
#define DEFAULT_MODEL_PATH "/opt/hobot/model/s600/basic/yolo11n_seg_nashe_640x640_nv12.hbm"
#else
#define DEFAULT_MODEL_PATH "/opt/hobot/model/s100/basic/yolo11n_seg_nashe_640x640_nv12.hbm"
#endif

// ---------------------------------------------------------------------------
// Command-line flags
// ---------------------------------------------------------------------------
DEFINE_string(model_path, DEFAULT_MODEL_PATH,
              "Path to BPU Quantized *.hbm model file.");
DEFINE_string(test_img, "../../../test_data/office_desk.jpg",
              "Path to load the test image.");
DEFINE_string(label_file, "../../../test_data/coco_classes.names",
              "Path to load class name list (one label per line, e.g., COCO).");
DEFINE_double(score_thres, 0.25, "Confidence score threshold for filtering detections.");
DEFINE_double(nms_thres,   0.7,  "IoU threshold for Non-Maximum Suppression.");
DEFINE_bool(no_morph,      false, "Disable morphological opening on mask edges.");

/**
 * @brief Main entry for running YOLO11-Seg instance segmentation on a single image.
 *
 * Pipeline:
 * 1) Parse CLI flags.
 * 2) Build inference config and initialize YOLO11Seg runtime.
 * 3) Load input image (BGR) and record original resolution.
 * 4) Preprocess -> Inference -> Postprocess.
 * 5) Draw boxes, masks, and contours, then save the result image.
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

    // Build inference config
    Yolo11SegConfig cfg;
    cfg.score_thresh = static_cast<float>(FLAGS_score_thres);
    cfg.nms_thresh   = static_cast<float>(FLAGS_nms_thres);
    cfg.do_morph     = !FLAGS_no_morph;

    // Initialize YOLO11Seg runtime
    YOLO11Seg yolo11_seg;
    ret = yolo11_seg.init(FLAGS_model_path.c_str());
    if (ret != 0) {
        fprintf(stderr, "YOLO11Seg init failed (ret=%d)\n", ret);
        return ret;
    }

    // Load input image (BGR)
    auto image = load_bgr_image(FLAGS_test_img);
    const int orig_w = image.cols;
    const int orig_h = image.rows;

    // 1) Preprocess: letterbox + BGR->NV12
    ret = pre_process(yolo11_seg.input_tensors, image,
                      yolo11_seg.input_w, yolo11_seg.input_h);
    if (ret != 0) { fprintf(stderr, "pre_process failed\n"); return ret; }

    // 2) Inference: BPU execution
    ret = infer(yolo11_seg.output_tensors, yolo11_seg.input_tensors, yolo11_seg.dnn_handle);
    if (ret != 0) { fprintf(stderr, "infer failed\n"); return ret; }

    // 3) Postprocess: decode + NMS + mask generation
    InstanceSegResult seg_result = post_process(yolo11_seg.output_tensors, cfg,
                                                orig_w, orig_h, yolo11_seg.input_w, yolo11_seg.input_h);

    // Visualize: boxes + masks + contours
    std::vector<std::string> class_names = load_linewise_labels(FLAGS_label_file);
    draw_boxes(image, seg_result.detections, class_names, rdk_colors);
    draw_masks(image, seg_result.detections, seg_result.masks, rdk_colors, 0.4f);
    draw_contours(image, seg_result.detections, seg_result.masks, rdk_colors, 1);

    // Save result
    cv::imwrite("result.jpg", image);
    std::cout << "[Saved] Result saved to: result.jpg" << std::endl;

    return 0;
}
