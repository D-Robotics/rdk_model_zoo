// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <vector>
namespace yolo26_depth {
struct RuntimeOptions {
  std::string target = "x5", model_path, image_path, output_directory;
  int warmup = 0;
  bool help = false;
};
RuntimeOptions parse_options(const std::vector<std::string> &arguments);
std::string json_quote(const std::string &value);
void write_npy(const std::string &path, const std::vector<float> &values,
               int height, int width);
void write_f32(const std::string &path, const std::vector<float> &values);
} // namespace yolo26_depth
