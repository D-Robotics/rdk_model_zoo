#ifndef RDK_MODEL_ZOO_YOLOV5_VISUALIZE_HPP_
#define RDK_MODEL_ZOO_YOLOV5_VISUALIZE_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_decode.hpp"

#include <string>
#include <vector>

namespace cv { class Mat; }

namespace yolov5 {

void render_detections(cv::Mat& image, const std::vector<Detection>& detections,
                       int model_size, bool letterbox, const std::string& output_path,
                       const std::vector<std::string>& labels);

std::vector<std::string> load_labels(const std::string& label_path);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_VISUALIZE_HPP_
