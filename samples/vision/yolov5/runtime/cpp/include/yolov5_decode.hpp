#ifndef RDK_MODEL_ZOO_YOLOV5_DECODE_HPP_
#define RDK_MODEL_ZOO_YOLOV5_DECODE_HPP_
// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <vector>

namespace yolov5 {

struct HeadShape {
  int height;
  int width;
  int channels;
};

struct Detection {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int class_id;
};

bool validate_head_shapes(const std::vector<HeadShape>& heads,
                         int input_size, int classes);

std::vector<int> order_heads_by_shape(const std::vector<HeadShape>& heads,
                                      int input_size, int classes);

std::vector<Detection> decode_heads(
    const std::vector<std::vector<float>>& raw_heads,
    const std::vector<HeadShape>& heads, int input_size, int classes,
    float score_threshold, float nms_threshold,
    const std::array<float, 18>& anchors);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_DECODE_HPP_
