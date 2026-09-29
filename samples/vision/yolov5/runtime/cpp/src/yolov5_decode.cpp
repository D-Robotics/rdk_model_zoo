// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_decode.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace yolov5 {
namespace {

float sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

float iou(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float inter = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
  const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
  const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
  return inter / std::max(area_a + area_b - inter, 1.0e-12F);
}

}  // namespace

bool validate_head_shapes(const std::vector<HeadShape>& heads,
                          int input_size, int classes) {
  if (input_size <= 0 || classes <= 0 || heads.size() != 3) return false;
  const int channels = 3 * (5 + classes);
  std::set<int> strides;
  for (const auto& head : heads) {
    if (head.height != head.width || head.height <= 0 ||
        head.channels != channels || input_size % head.height != 0) {
      return false;
    }
    const int stride = input_size / head.height;
    if (stride != 8 && stride != 16 && stride != 32) return false;
    strides.insert(stride);
  }
  return strides.size() == 3;
}

std::vector<int> order_heads_by_shape(const std::vector<HeadShape>& heads,
                                      int input_size, int classes) {
  if (!validate_head_shapes(heads, input_size, classes))
    throw std::invalid_argument("YOLOv5 output heads are not a unique 8/16/32 contract");
  std::vector<int> ordered;
  for (int stride : {8, 16, 32}) {
    auto it = std::find_if(heads.begin(), heads.end(), [&](const HeadShape& h) {
      return input_size / h.height == stride;
    });
    ordered.push_back(static_cast<int>(std::distance(heads.begin(), it)));
  }
  return ordered;
}

std::vector<Detection> decode_heads(
    const std::vector<std::vector<float>>& raw_heads,
    const std::vector<HeadShape>& heads, int input_size, int classes,
    const DecodePolicy& policy,
    const std::array<float, 18>& anchors) {
  if (raw_heads.size() != heads.size() || !std::isfinite(policy.score_threshold) ||
      !std::isfinite(policy.nms_threshold) || policy.score_threshold < 0.0F ||
      policy.score_threshold > 1.0F || policy.nms_threshold < 0.0F ||
      policy.nms_threshold > 1.0F || policy.top_k_per_class == 0 ||
      policy.top_k_per_class < -1)
    throw std::invalid_argument("Invalid YOLOv5 decode arguments");
  const auto ordered = order_heads_by_shape(heads, input_size, classes);
  const int channels = 3 * (5 + classes);
  std::vector<Detection> candidates;
  for (std::size_t level = 0; level < ordered.size(); ++level) {
    const int source = ordered[level];
    const auto& shape = heads[source];
    const auto& raw = raw_heads[source];
    const std::size_t expected = static_cast<std::size_t>(shape.height) *
                                 static_cast<std::size_t>(shape.width) * channels;
    if (raw.size() != expected)
      throw std::invalid_argument("YOLOv5 output buffer does not match metadata");
    const int stride = input_size / shape.height;
    for (int y = 0; y < shape.height; ++y) {
      for (int x = 0; x < shape.width; ++x) {
        for (int a = 0; a < 3; ++a) {
          const std::size_t base = (static_cast<std::size_t>(y) * shape.width + x) * channels +
                                   static_cast<std::size_t>(a) * (5 + classes);
          const float objectness = sigmoid(raw[base + 4]);
          if (!std::isfinite(objectness)) continue;
          // Both preserved C++ sources select one maximum class per anchor before
          // confidence filtering; emitting every class changes the source contract.
          int cls = 0;
          for (int candidate = 1; candidate < classes; ++candidate)
            if (raw[base + 5 + candidate] > raw[base + 5 + cls]) cls = candidate;
          const float score = objectness * sigmoid(raw[base + 5 + cls]);
          if (!std::isfinite(score)) continue;
          if (policy.strict_score_boundary ? !(score > policy.score_threshold)
                                           : score < policy.score_threshold)
            continue;
          const float cx = (2.0F * sigmoid(raw[base]) - 0.5F + x) * stride;
          const float cy = (2.0F * sigmoid(raw[base + 1]) - 0.5F + y) * stride;
          const float w = std::pow(2.0F * sigmoid(raw[base + 2]), 2.0F) * anchors[level * 6 + a * 2];
          const float h = std::pow(2.0F * sigmoid(raw[base + 3]), 2.0F) * anchors[level * 6 + a * 2 + 1];
          candidates.push_back({cx - w / 2.0F, cy - h / 2.0F, cx + w / 2.0F,
                                cy + h / 2.0F, score, cls});
        }
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
    return a.score > b.score;
  });
  std::vector<Detection> result;
  std::vector<bool> suppressed(candidates.size(), false);
  std::vector<int> kept_per_class(static_cast<std::size_t>(classes), 0);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) continue;
    const int cls = candidates[i].class_id;
    // Candidates are sorted by descending score, so once a class reaches its
    // cap every remaining candidate of that class is dropped as well; this is
    // the source X5 NMSBoxes top_k break.
    if (policy.top_k_per_class > 0 && kept_per_class[static_cast<std::size_t>(cls)] >=
                                          policy.top_k_per_class)
      continue;
    result.push_back(candidates[i]);
    ++kept_per_class[static_cast<std::size_t>(cls)];
    for (std::size_t j = i + 1; j < candidates.size(); ++j) {
      if (!suppressed[j] && candidates[i].class_id == candidates[j].class_id &&
          iou(candidates[i], candidates[j]) > policy.nms_threshold)
        suppressed[j] = true;
    }
  }
  return result;
}

}  // namespace yolov5
