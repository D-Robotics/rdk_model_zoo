// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace yolo26_depth {
constexpr int kInputSize = 768;
constexpr int kOutputSize = 192;
struct ImageContext {
  int original_height = 0, original_width = 0;
  int top = 0, bottom = 0, left = 0, right = 0;
};
struct TensorLayout {
  std::array<std::size_t, 4> valid{};
  std::array<std::size_t, 4> aligned{};
  std::array<std::size_t, 4>
      strides{}; // byte strides; all zero means derive from aligned shape
  std::size_t capacity = 0;
};
ImageContext letterbox_geometry(int height, int width);
void validate_context(const ImageContext &context);
std::array<std::size_t, 4> validated_strides(const TensorLayout &layout);
std::vector<float> read_log_depth(const void *bytes,
                                  const TensorLayout &layout);
double percentile(std::vector<float> values, double fraction);
bool match_x5_identity(std::string boardinfo, std::string socinfo,
                       std::string device_tree);
void require_x5_board();
} // namespace yolo26_depth
