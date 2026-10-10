// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "depth.hpp"

namespace yolo26_depth {
struct RuntimeOptions {
  std::string target = "x5", model_path, image_path, output_directory;
  int warmup = 0;
  bool help = false;
};
RuntimeOptions parse_options(const std::vector<std::string> &arguments);
void print_help();
cv::Mat load_image(const std::string &path);
cv::Mat colorize_depth(const cv::Mat &depth);
void save_results(const RuntimeOptions &options, const cv::Mat &image,
                  const DepthResult &result, const std::string &model_name);
std::string json_quote(const std::string &value);
void write_npy(const std::string &path, const std::vector<float> &values,
               int height, int width);
void write_f32(const std::string &path, const std::vector<float> &values);
} // namespace yolo26_depth
