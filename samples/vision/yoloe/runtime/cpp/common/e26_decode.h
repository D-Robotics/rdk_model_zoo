// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Source: rdk_s 380e1a2bf42041af54be6f34935e50197cfadff9, YOLOE26 raw-v1.
// Finite compact float tensors only; no SDK, image I/O, NMS or dequantization.
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
namespace yoloe {
struct RawDetection {
  std::array<float, 4> box{};
  float score = 0;
  int label = 0;
  std::array<float, 32> coefficients{};
};
namespace detail {
constexpr int kModelWidth = 640, kClasses = 4585, kMaskChannels = 32,
              kMaskSize = 160;
constexpr std::array<int, 3> kStrides{8, 16, 32};
struct Tensor {
  const float *data;
  int h, w, channels;
  float at(int anchor, int channel) const {
    return data[static_cast<size_t>(anchor) * channels + channel];
  }
};
inline std::vector<RawDetection> decode(const std::vector<Tensor> &tensors,
                                        float threshold, int max_det,
                                        bool single_label) {
  if (!(threshold > 0.0f && threshold < 1.0f) || max_det < 1 ||
      max_det > 8400 || tensors.size() != 10) {
    throw std::invalid_argument("Invalid threshold, max_det, or output count");
  }
  for (int i = 0; i < 10; ++i) {
    const auto &tensor = tensors[i];
    const int hw = i == 9 ? kMaskSize : kModelWidth / kStrides[i / 3];
    const int channels =
        i == 9 ? kMaskChannels
               : (i % 3 == 0 ? kClasses : (i % 3 == 1 ? 4 : kMaskChannels));
    if (!tensor.data || tensor.h != hw || tensor.w != hw ||
        tensor.channels != channels) {
      throw std::invalid_argument("Invalid raw-v1 output shape");
    }
    const size_t count = static_cast<size_t>(hw) * hw * channels;
    for (size_t j = 0; j < count; ++j) {
      if (!std::isfinite(tensor.data[j])) {
        throw std::invalid_argument("Non-finite model output");
      }
    }
  }

  struct Candidate {
    float value;
    int scale;
    int anchor;
    int label;
    int rank{0};
  };

  std::vector<Candidate> anchors;
  anchors.reserve(8400);
  for (int scale = 0; scale < 3; ++scale) {
    const auto &cls = tensors[scale * 3];
    for (int anchor = 0; anchor < cls.h * cls.w; ++anchor) {
      int label = 0;
      for (int c = 1; c < kClasses; ++c) {
        if (cls.at(anchor, c) > cls.at(anchor, label))
          label = c;
      }
      anchors.push_back({cls.at(anchor, label), scale, anchor, label});
    }
  }

  auto order = [](const Candidate &a, const Candidate &b) {
    if (a.value != b.value)
      return a.value > b.value;
    if (a.scale != b.scale)
      return a.scale < b.scale;
    if (a.anchor != b.anchor)
      return a.anchor < b.anchor;
    return a.label < b.label;
  };
  const size_t anchor_count = std::min<size_t>(max_det, anchors.size());
  std::partial_sort(anchors.begin(), anchors.begin() + anchor_count,
                    anchors.end(), order);
  anchors.resize(anchor_count);

  if (!single_label) {
    std::vector<Candidate> classes;
    classes.reserve(anchor_count * kClasses);
    for (int rank = 0; rank < static_cast<int>(anchor_count); ++rank) {
      const auto &anchor = anchors[rank];
      for (int c = 0; c < kClasses; ++c) {
        classes.push_back({tensors[3 * anchor.scale].at(anchor.anchor, c),
                           anchor.scale, anchor.anchor, c, rank});
      }
    }
    const size_t class_count = std::min<size_t>(max_det, classes.size());
    std::partial_sort(classes.begin(), classes.begin() + class_count,
                      classes.end(),
                      [](const Candidate &a, const Candidate &b) {
                        if (a.value != b.value)
                          return a.value > b.value;
                        if (a.rank != b.rank)
                          return a.rank < b.rank;
                        return a.label < b.label;
                      });
    classes.resize(class_count);
    anchors = std::move(classes);
  }

  const float raw_threshold = std::log(threshold / (1.0f - threshold));
  std::vector<RawDetection> result;
  result.reserve(anchors.size());
  for (const auto &candidate : anchors) {
    if (candidate.value <= raw_threshold)
      continue;
    const int stride = kStrides[candidate.scale];
    const int grid = kModelWidth / stride;
    const float x = candidate.anchor % grid + 0.5f;
    const float y = candidate.anchor / grid + 0.5f;
    const auto &box = tensors[3 * candidate.scale + 1];
    RawDetection detection;
    detection.box = {(x - box.at(candidate.anchor, 0)) * stride,
                     (y - box.at(candidate.anchor, 1)) * stride,
                     (x + box.at(candidate.anchor, 2)) * stride,
                     (y + box.at(candidate.anchor, 3)) * stride};
    detection.score =
        1.0f / (1.0f + std::exp(-std::clamp(candidate.value, -80.0f, 80.0f)));
    detection.label = candidate.label;
    for (int c = 0; c < kMaskChannels; ++c) {
      detection.coefficients[c] =
          tensors[3 * candidate.scale + 2].at(candidate.anchor, c);
    }
    for (float value : detection.box)
      if (!std::isfinite(value))
        throw std::invalid_argument("Decoded box overflow");
    result.push_back(detection);
  }
  return result;
}

} // namespace detail
// Semantic order: (classes, direct LTRB, mask coefficients) for each stride,
// then prototype. Exact vector sizes are checked before any dereference.
inline std::vector<RawDetection>
decode_e26(const std::array<std::vector<float>, 10> &outputs,
           float threshold = 0.25f, int max_det = 300,
           bool single_label = true) {
  std::vector<detail::Tensor> tensors;
  for (int i = 0; i < 10; ++i) {
    int grid = i == 9 ? 160 : (80 >> (i / 3));
    int channels = i == 9 ? 32 : (i % 3 == 0 ? 4585 : (i % 3 == 1 ? 4 : 32));
    if (outputs[i].size() != static_cast<size_t>(grid) * grid * channels)
      throw std::invalid_argument("Wrong compact YOLOE26 tensor size");
    tensors.push_back({outputs[i].data(), grid, grid, channels});
  }
  return detail::decode(tensors, threshold, max_det, single_label);
}
} // namespace yoloe
