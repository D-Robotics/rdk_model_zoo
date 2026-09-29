// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#ifndef YOLO_COMMON_TASK_OUTPUTS_H_
#define YOLO_COMMON_TASK_OUTPUTS_H_
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/tensor_view.h"
namespace yolo {
struct FloatOutputPlan {
  OutputShape shape;
  size_t row_bytes, cell_bytes, required_bytes;
};
inline size_t checked_span(size_t step, size_t count, size_t tail,
                           size_t limit) {
  if (tail > limit || (count && step > (limit - tail) / count))
    throw std::invalid_argument("Output strides exceed physical allocation.");
  return step * count + tail;
}
inline FloatOutputPlan nhwc_float_plan(const std::vector<int>& shape,
                                       const std::vector<size_t>& strides,
                                       size_t bytes, bool float32, bool none) {
  if (!float32 || !none || shape.size() != 4 || strides.size() != 4 ||
      shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 ||
      strides[3] != sizeof(float))
    throw std::invalid_argument(
        "Expected batch-one unquantized FLOAT32 NHWC output.");
  size_t cell = checked_span(sizeof(float), shape[3], 0, bytes);
  if (strides[2] < cell || strides[2] % sizeof(float) ||
      strides[1] % sizeof(float))
    throw std::invalid_argument("Invalid output cell/row stride.");
  size_t row = checked_span(strides[2], shape[2] - 1, cell, bytes);
  if (strides[1] < row) throw std::invalid_argument("Overlapping output rows.");
  size_t required = checked_span(strides[1], shape[1] - 1, row, bytes);
  return {{shape[1], shape[2], shape[3]}, strides[1], strides[2], required};
}
inline std::vector<float> copy_float_output(const void* data, size_t bytes,
                                            const FloatOutputPlan& plan) {
  // Revalidate even when called with a caller-constructed plan.
  auto validated =
      nhwc_float_plan({1, plan.shape.h, plan.shape.w, plan.shape.c},
                      {bytes, plan.row_bytes, plan.cell_bytes, sizeof(float)},
                      bytes, true, true);
  if (!data || validated.required_bytes != plan.required_bytes)
    throw std::invalid_argument("Invalid output buffer or plan.");
  std::vector<float> result;
  result.reserve(static_cast<size_t>(plan.shape.h) * plan.shape.w *
                 plan.shape.c);
  const auto* memory = static_cast<const unsigned char*>(data);
  for (int y = 0; y < plan.shape.h; ++y)
    for (int x = 0; x < plan.shape.w; ++x)
      for (int c = 0; c < plan.shape.c; ++c) {
        float value;
        std::memcpy(&value,
                    memory + y * plan.row_bytes + x * plan.cell_bytes +
                        c * sizeof(float),
                    sizeof(float));
        if (!std::isfinite(value))
          throw std::invalid_argument("Output contains nonfinite values.");
        result.push_back(value);
      }
  return result;
}
struct TaskHeadPlan {
  std::array<int, 3> cls, box, extra;
  int prototype = -1;
  bool direct_ltrb = false;
};
inline TaskHeadPlan bind_task_heads(const std::vector<OutputShape>& shapes,
                                    int h, int w, bool segment) {
  if (h <= 0 || h != w || h % 32 || shapes.size() != (segment ? 10u : 9u))
    throw std::invalid_argument(
        "Pose/segment require square stride-32 geometry and exactly 9/10 "
        "heads.");
  TaskHeadPlan plan;
  int box_channels = 0;
  for (int i = 0; i < 3; ++i) {
    const int grid = h / (8 << i);
    plan.cls[i] = find_output_by_shape(shapes, grid, grid, segment ? 80 : 1);
    plan.extra[i] = find_output_by_shape(shapes, grid, grid, segment ? 32 : 51);
    int direct = find_output_by_shape(shapes, grid, grid, 4);
    int dfl = find_output_by_shape(shapes, grid, grid, 64);
    if (plan.cls[i] < 0 || plan.extra[i] < 0 || (direct >= 0) == (dfl >= 0))
      throw std::invalid_argument(
          "Missing, ambiguous or incompatible task output roles.");
    int channels = direct >= 0 ? 4 : 64;
    if (box_channels && box_channels != channels)
      throw std::invalid_argument("Mixed DFL/LTRB output scales.");
    box_channels = channels;
    plan.box[i] = direct >= 0 ? direct : dfl;
  }
  if (segment) {
    plan.prototype = find_output_by_shape(shapes, h / 4, w / 4, 32);
    if (plan.prototype < 0)
      throw std::invalid_argument("Expected a unique stride-4 NHWC prototype.");
  }
  plan.direct_ltrb = box_channels == 4;
  return plan;
}
}  // namespace yolo
#endif
