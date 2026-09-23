// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_visualize.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <string>

namespace yolov5 {

void render_detections(cv::Mat& image, const std::vector<Detection>& detections,
                       int model_size, bool letterbox, const std::string& output_path,
                       const std::vector<std::string>& labels) {
  if (image.empty() || model_size <= 0 || output_path.empty())
    throw std::invalid_argument("Cannot render YOLOv5 detections");
  const double scale = letterbox
      ? std::min(static_cast<double>(model_size) / image.cols,
                 static_cast<double>(model_size) / image.rows)
      : 1.0;
  const double pad_x = letterbox ? (model_size - image.cols * scale) / 2.0 : 0.0;
  const double pad_y = letterbox ? (model_size - image.rows * scale) / 2.0 : 0.0;
  for (const auto& detection : detections) {
    const int x1 = std::max(0, std::min(image.cols - 1,
        static_cast<int>((detection.x1 - pad_x) / scale)));
    const int y1 = std::max(0, std::min(image.rows - 1,
        static_cast<int>((detection.y1 - pad_y) / scale)));
    const int x2 = std::max(0, std::min(image.cols - 1,
        static_cast<int>((detection.x2 - pad_x) / scale)));
    const int y2 = std::max(0, std::min(image.rows - 1,
        static_cast<int>((detection.y2 - pad_y) / scale)));
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    const std::string name = detection.class_id >= 0 &&
            detection.class_id < static_cast<int>(labels.size())
        ? labels[static_cast<std::size_t>(detection.class_id)]
        : std::to_string(detection.class_id);
    cv::putText(image, name + ":" +
                std::to_string(detection.score), cv::Point(x1, std::max(12, y1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
  }
  if (!cv::imwrite(output_path, image)) throw std::runtime_error("Cannot save YOLOv5 output image");
}

std::vector<std::string> load_labels(const std::string& label_path) {
  if (label_path.empty()) return {};
  std::ifstream input(label_path);
  if (!input) throw std::runtime_error("Cannot open YOLOv5 label file: " + label_path);
  std::vector<std::string> labels;
  for (std::string line; std::getline(input, line);) labels.push_back(line);
  return labels;
}

}  // namespace yolov5
