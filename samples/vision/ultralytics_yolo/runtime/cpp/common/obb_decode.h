/*
 * Copyright (c) 2026, D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Host-testable geometry for the YOLO26 oriented-box head.

#ifndef RUNTIME_CPP_COMMON_OBB_DECODE_H_
#define RUNTIME_CPP_COMMON_OBB_DECODE_H_

#include <algorithm>
#include <cmath>

#include "decode.h"

namespace yolo {

struct RotatedBox {
  float cx = 0.0f;
  float cy = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
  float angle_rad = 0.0f;
};

inline bool decode_obb_cell(const float* ltrb_raw, float angle_raw,
                            float grid_x, float grid_y, float stride,
                            float angle_sign, float angle_offset_rad,
                            RotatedBox* box) {
  if (ltrb_raw == nullptr || box == nullptr || !std::isfinite(angle_raw) ||
      !std::isfinite(stride) || stride <= 0.0f) {
    return false;
  }
  const float left = std::fabs(ltrb_raw[0]);
  const float top = std::fabs(ltrb_raw[1]);
  const float right = std::fabs(ltrb_raw[2]);
  const float bottom = std::fabs(ltrb_raw[3]);
  if (!std::isfinite(left) || !std::isfinite(top) ||
      !std::isfinite(right) || !std::isfinite(bottom)) {
    return false;
  }
  const float angle = angle_raw * angle_sign + angle_offset_rad;
  if (!std::isfinite(angle)) return false;
  const float half_w = (right - left) * 0.5f;
  const float half_h = (bottom - top) * 0.5f;
  const float cosine = std::cos(angle);
  const float sine = std::sin(angle);
  box->cx = (grid_x + half_w * cosine - half_h * sine) * stride;
  box->cy = (grid_y + half_w * sine + half_h * cosine) * stride;
  box->width = (left + right) * stride;
  box->height = (top + bottom) * stride;
  box->angle_rad = angle;
  return std::isfinite(box->cx) && std::isfinite(box->cy) &&
         std::isfinite(box->width) && std::isfinite(box->height) &&
         box->width > 0.0f && box->height > 0.0f;
}

inline void regularize_obb(RotatedBox* box, bool regularize,
                           bool wrap_half_turn) {
  if (box == nullptr) return;
  const float half_pi = static_cast<float>(std::acos(-1.0) * 0.5);
  const float pi = half_pi * 2.0f;
  if (regularize && box->width < box->height) {
    std::swap(box->width, box->height);
    box->angle_rad += half_pi;
  }
  if (wrap_half_turn) {
    box->angle_rad = std::fmod(box->angle_rad + half_pi, pi);
    if (box->angle_rad < 0.0f) box->angle_rad += pi;
    box->angle_rad -= half_pi;
  }
}

// Restores a model-input box to source pixels. `clip` bounds the centre and
// size to the image, matching the X5 policy of runtime/python/obb_decode.py;
// the S-series policy keeps the unclipped geometry.
inline bool map_obb_to_source(RotatedBox* box,
                              const ImageTransform& transform, int image_w,
                              int image_h, bool clip) {
  if (box == nullptr || image_w <= 0 || image_h <= 0 ||
      !std::isfinite(transform.scale_x) ||
      !std::isfinite(transform.scale_y) || transform.scale_x <= 0.0f ||
      transform.scale_y <= 0.0f) {
    return false;
  }
  box->cx = (box->cx - transform.shift_x) / transform.scale_x;
  box->cy = (box->cy - transform.shift_y) / transform.scale_y;
  box->width /= transform.scale_x;
  box->height /= transform.scale_y;
  if (!clip) return std::isfinite(box->angle_rad);
  box->cx = std::max(0.0f, std::min(box->cx, static_cast<float>(image_w)));
  box->cy = std::max(0.0f, std::min(box->cy, static_cast<float>(image_h)));
  box->width = std::max(0.0f, std::min(box->width, static_cast<float>(image_w)));
  box->height = std::max(0.0f, std::min(box->height, static_cast<float>(image_h)));
  return std::isfinite(box->angle_rad);
}

}  // namespace yolo

#endif  // RUNTIME_CPP_COMMON_OBB_DECODE_H_
