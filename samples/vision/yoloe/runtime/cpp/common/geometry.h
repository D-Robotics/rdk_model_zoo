// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
namespace yoloe {
enum class Protocol { E11, E26 };
struct Geometry {
  int width, height, resized_w, resized_h, left, top, right, bottom,
      resize_type;
  Protocol protocol;
};
inline int round_even(double x) {
  int floor = static_cast<int>(std::floor(x));
  double fraction = x - floor;
  return floor + (fraction > 0.5 || (fraction == 0.5 && floor % 2));
}
inline Geometry make_geometry(int width, int height, Protocol protocol,
                              int resize_type = 1) {
  if (width <= 0 || height <= 0 ||
      (protocol != Protocol::E11 && protocol != Protocol::E26) ||
      (resize_type != 0 && resize_type != 1) ||
      (protocol == Protocol::E26 && resize_type != 1))
    throw std::invalid_argument("Invalid YOLOE image geometry/protocol");
  int rw = 640, rh = 640;
  if (resize_type == 1) {
    double gain = std::min(640.0 / width, 640.0 / height);
    rw = protocol == Protocol::E26 ? round_even(width * gain)
                                   : static_cast<int>(width * gain);
    rh = protocol == Protocol::E26 ? round_even(height * gain)
                                   : static_cast<int>(height * gain);
    rw = std::clamp(rw, 1, 640);
    rh = std::clamp(rh, 1, 640);
  }
  int left = (640 - rw) / 2, top = (640 - rh) / 2;
  return {width,           height,         rw,          rh,      left, top,
          640 - rw - left, 640 - rh - top, resize_type, protocol};
}
inline void validate_geometry(const Geometry &g) {
  auto expected = make_geometry(g.width, g.height, g.protocol, g.resize_type);
  if (g.resized_w != expected.resized_w || g.resized_h != expected.resized_h ||
      g.left != expected.left || g.top != expected.top ||
      g.right != expected.right || g.bottom != expected.bottom)
    throw std::invalid_argument("Geometry differs from preprocessing protocol");
}
inline std::array<float, 4> restore_box(const std::array<float, 4> &box,
                                        const Geometry &g) {
  validate_geometry(g);
  for (float v : box)
    if (!std::isfinite(v))
      throw std::invalid_argument("Nonfinite box");
  if (box[2] < box[0] || box[3] < box[1])
    throw std::invalid_argument("Reversed box");
  float sx = static_cast<float>(static_cast<double>(g.resized_w) / g.width);
  float sy = static_cast<float>(static_cast<double>(g.resized_h) / g.height);
  return {std::clamp((box[0] - g.left) / sx, 0.f, static_cast<float>(g.width)),
          std::clamp((box[1] - g.top) / sy, 0.f, static_cast<float>(g.height)),
          std::clamp((box[2] - g.left) / sx, 0.f, static_cast<float>(g.width)),
          std::clamp((box[3] - g.top) / sy, 0.f, static_cast<float>(g.height))};
}
} // namespace yoloe
