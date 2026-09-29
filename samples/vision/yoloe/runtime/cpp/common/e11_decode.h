// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Canonical float-only E11 candidate math, derived from the pinned S native
// implementation. Reuses shared DFL16/sigmoid primitives; owns no SDK memory.
#include "candidate.h"
#include "common/decode.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>
namespace yoloe {
namespace detail {
inline float candidate_iou(const RawDetection &a, const RawDetection &b) {
  float w = std::max(0.f, std::min(a.box[2], b.box[2]) -
                              std::max(a.box[0], b.box[0]));
  float h = std::max(0.f, std::min(a.box[3], b.box[3]) -
                              std::max(a.box[1], b.box[1]));
  float intersection = w * h;
  float area_a = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]);
  float area_b = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
  return intersection / (area_a + area_b - intersection + 1e-9f);
}
// Internal: candidates have already passed finite DFL decoding. Unlike the
// source unordered_map/OpenMP merge, ties are deterministic by input index.
inline std::vector<RawDetection>
nms_e11(const std::vector<RawDetection> &candidates, float threshold) {
  std::map<int, std::vector<size_t>> classes;
  for (size_t i = 0; i < candidates.size(); ++i)
    classes[candidates[i].label].push_back(i);
  std::vector<RawDetection> kept;
  for (auto &entry : classes) {
    auto &ids = entry.second;
    std::stable_sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
      return candidates[a].score > candidates[b].score;
    });
    std::vector<bool> suppressed(ids.size(), false);
    for (size_t i = 0; i < ids.size(); ++i) {
      if (suppressed[i])
        continue;
      kept.push_back(candidates[ids[i]]);
      for (size_t j = i + 1; j < ids.size(); ++j)
        if (!suppressed[j] &&
            candidate_iou(candidates[ids[i]], candidates[ids[j]]) > threshold)
          suppressed[j] = true;
    }
  }
  return kept;
}
} // namespace detail
// Ten compact float NHWC vectors, ordered cls/DFL64/coeff32 at strides 8/16/32
// then prototype. Source C++ semantics: score >= threshold, suppress IoU > NMS.
inline std::vector<RawDetection>
decode_e11(const std::array<std::vector<float>, 10> &outputs,
           float score_threshold = 0.25f, float nms_threshold = 0.7f) {
  if (!std::isfinite(score_threshold) || score_threshold <= 0 ||
      score_threshold >= 1 || !std::isfinite(nms_threshold) ||
      nms_threshold < 0 || nms_threshold > 1)
    throw std::invalid_argument("E11 requires score in (0,1), NMS in [0,1]");
  for (int i = 0; i < 10; ++i) {
    int grid = i == 9 ? 160 : 80 >> (i / 3),
        channels = i == 9 ? 32 : (i % 3 == 0 ? 4585 : (i % 3 == 1 ? 64 : 32));
    if (outputs[i].size() != static_cast<size_t>(grid) * grid * channels)
      throw std::invalid_argument("Wrong compact YOLOE11 tensor size");
    for (float value : outputs[i])
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite YOLOE11 output");
  }
  float threshold = yolo::raw_logit_threshold(score_threshold);
  std::vector<RawDetection> candidates;
  for (int scale = 0; scale < 3; ++scale) {
    int grid = 80 >> scale, stride = 8 << scale;
    for (int anchor = 0; anchor < grid * grid; ++anchor) {
      const float *cls =
          outputs[3 * scale].data() + static_cast<size_t>(anchor) * 4585;
      int label = static_cast<int>(std::max_element(cls, cls + 4585) - cls);
      if (cls[label] < threshold)
        continue;
      RawDetection detection;
      detection.label = label;
      detection.score = yolo::sigmoid(cls[label]);
      float distances[4];
      yolo::decode_box_dfl(outputs[3 * scale + 1].data() +
                               static_cast<size_t>(anchor) * 64,
                           distances);
      float x = anchor % grid + 0.5f, y = anchor / grid + 0.5f;
      yolo::box_from_distances(x, y, distances, static_cast<float>(stride),
                               &detection.box[0], &detection.box[1],
                               &detection.box[2], &detection.box[3]);
      for (float value : detection.box)
        if (!std::isfinite(value))
          throw std::invalid_argument("Nonfinite decoded E11 box");
      std::copy_n(outputs[3 * scale + 2].data() +
                      static_cast<size_t>(anchor) * 32,
                  32, detection.coefficients.begin());
      candidates.push_back(detection);
    }
  }
  return detail::nms_e11(candidates, nms_threshold);
}
} // namespace yoloe
