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

// Per-target decode policy. The two fixed sources really differ here, so the
// difference is an explicit parameter rather than a silently unified rule:
//   * X5 runs cv::dnn::NMSBoxes(..., score_threshold, nms_threshold, top_k=300)
//     per class, which keeps only scores strictly greater than the threshold and
//     caps each class at 300 boxes.
//   * S uses yolov5_decode_all_layers (conf < threshold discards, i.e. equal
//     scores are kept) followed by nms_bboxes with no per-class cap.
struct DecodePolicy {
  float score_threshold = 0.25F;
  float nms_threshold = 0.45F;
  // -1 keeps every surviving box; a positive value caps the boxes kept per
  // class (source X5 NMS_TOP_K is 300).
  int top_k_per_class = -1;
  // true keeps score > threshold (source X5 OpenCV boundary); false keeps
  // score >= threshold (source S boundary).
  bool strict_score_boundary = false;
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
    const DecodePolicy& policy,
    const std::array<float, 18>& anchors);

}  // namespace yolov5

#endif  // RDK_MODEL_ZOO_YOLOV5_DECODE_HPP_
