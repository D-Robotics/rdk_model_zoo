/*
 * Copyright (c) 2026, D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared box-decode primitives for the two YOLO head contracts published by
// this sample:
//   * direct-LTRB heads (YOLO26): box maps carry 4 float distances per cell.
//   * DFL heads (YOLOv5u/v8/v9/v10/yolo11/yolo12/yolov13): box maps carry
//     4 x 16 raw logits per cell that softmax into distance distributions.
// The math mirrors runtime/python/rdk_yolo_utils/postprocess.py so the C++
// and Python runtimes stay contract-compatible.

#ifndef RUNTIME_CPP_COMMON_DECODE_H_
#define RUNTIME_CPP_COMMON_DECODE_H_

#include <algorithm>
#include <cmath>
#include <limits>

namespace yolo {

// Box-decode protocols selectable per model.
enum class BoxDecode { kUnknown, kDirectLtrb, kDfl };

// DFL heads distribute each side distance over this many bins.
const int kDflBins = 16;

// Channels carried by one box-map cell under each protocol.
inline int box_channels(BoxDecode decode) {
  if (decode == BoxDecode::kDirectLtrb) return 4;
  if (decode == BoxDecode::kDfl) return 4 * kDflBins;
  return 0;
}

// Maps an observed box-map channel count to a decode protocol.
inline BoxDecode box_decode_from_channels(int channels) {
  if (channels == 4) return BoxDecode::kDirectLtrb;
  if (channels == 4 * kDflBins) return BoxDecode::kDfl;
  return BoxDecode::kUnknown;
}

inline float sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
  const float exp_value = std::exp(value);
  return exp_value / (1.0f + exp_value);
}

// Raw-logit threshold equivalent to `sigmoid(raw) >= score_threshold`.
// Thresholds outside (0, 1) degenerate to +/- infinity, matching the
// behaviour of the released detect sample.
inline float raw_logit_threshold(float score_threshold) {
  if (score_threshold <= 0.0f) return -std::numeric_limits<float>::infinity();
  if (score_threshold >= 1.0f) return std::numeric_limits<float>::infinity();
  return -std::log(1.0f / score_threshold - 1.0f);
}

// Letterbox/resize transform mapping model-input coordinates back to the
// source image.
struct ImageTransform {
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  int shift_x = 0;
  int shift_y = 0;
};

// Direct-LTRB heads: the four distances are stored as-is.
inline void decode_box_ltrb(const float* box4, float* ltrb4) {
  ltrb4[0] = box4[0];
  ltrb4[1] = box4[1];
  ltrb4[2] = box4[2];
  ltrb4[3] = box4[3];
}

// DFL heads: softmax over 16 bins per side, then the bin-index expectation
// is the distance (numerically stabilised like the Python softmax).
inline void decode_box_dfl(const float* box64, float* ltrb4) {
  for (int side = 0; side < 4; ++side) {
    const float* bins = box64 + side * kDflBins;
    float max_value = bins[0];
    for (int j = 1; j < kDflBins; ++j) {
      if (bins[j] > max_value) max_value = bins[j];
    }
    float sum = 0.0f;
    float expectation = 0.0f;
    for (int j = 0; j < kDflBins; ++j) {
      const float weight = std::exp(bins[j] - max_value);
      sum += weight;
      expectation += weight * j;
    }
    ltrb4[side] = expectation / sum;
  }
}

// Converts per-cell distances around a grid centre to input-image corners.
inline void box_from_distances(float grid_center_x, float grid_center_y,
                               const float* ltrb4, float stride, float* x1,
                               float* y1, float* x2, float* y2) {
  *x1 = (grid_center_x - ltrb4[0]) * stride;
  *y1 = (grid_center_y - ltrb4[1]) * stride;
  *x2 = (grid_center_x + ltrb4[2]) * stride;
  *y2 = (grid_center_y + ltrb4[3]) * stride;
}

// Maps input-image coordinates back to source-image coordinates and clamps
// them to the image bounds. Returns false for degenerate boxes.
inline bool map_to_source(float* x1, float* y1, float* x2, float* y2,
                          const ImageTransform& transform, int image_w,
                          int image_h) {
  *x1 = (*x1 - transform.shift_x) / transform.scale_x;
  *y1 = (*y1 - transform.shift_y) / transform.scale_y;
  *x2 = (*x2 - transform.shift_x) / transform.scale_x;
  *y2 = (*y2 - transform.shift_y) / transform.scale_y;
  *x1 = std::max(0.0f, std::min(*x1, static_cast<float>(image_w)));
  *y1 = std::max(0.0f, std::min(*y1, static_cast<float>(image_h)));
  *x2 = std::max(0.0f, std::min(*x2, static_cast<float>(image_w)));
  *y2 = std::max(0.0f, std::min(*y2, static_cast<float>(image_h)));
  return *x2 > *x1 && *y2 > *y1;
}

}  // namespace yolo

#endif  // RUNTIME_CPP_COMMON_DECODE_H_
