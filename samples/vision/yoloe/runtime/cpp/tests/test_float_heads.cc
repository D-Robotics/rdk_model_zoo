// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "float_heads.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
template <class F> void rejects(F fn) {
  bool failed = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    failed = true;
  }
  EXPECT(failed);
}
int main() {
  for (int channels : {4, 64}) {
    std::vector<yolo::OutputShape> shapes;
    for (int grid : {80, 40, 20})
      for (int c : {4585, channels, 32})
        shapes.emplace_back(grid, grid, c);
    shapes.emplace_back(160, 160, 32);
    std::reverse(shapes.begin(), shapes.end());
    auto roles = yoloe::bind_heads(shapes, channels);
    EXPECT(roles[0] == 9);
    EXPECT(roles[1] == 8);
    EXPECT(roles[9] == 0);
    rejects([&] { yoloe::bind_heads(shapes, channels == 4 ? 64 : 4); });
    auto invalid = shapes;
    invalid[9].c = 80;
    rejects([&] { yoloe::bind_heads(invalid, channels); });
    invalid = shapes;
    invalid[8] = invalid[9];
    rejects([&] { yoloe::bind_heads(invalid, channels); });
    invalid = shapes;
    invalid.push_back(shapes.back());
    rejects([&] { yoloe::bind_heads(invalid, channels); });
  }
  // Padded physical NHWC reads use the existing shared float boundary.
  auto plan =
      yolo::nhwc_float_plan({1, 2, 2, 2}, {64, 32, 12, 4}, 64, true, true);
  std::vector<float> raw(16, -999);
  raw[0] = 1;
  raw[1] = 2;
  raw[3] = 3;
  raw[4] = 4;
  raw[8] = 5;
  raw[9] = 6;
  raw[11] = 7;
  raw[12] = 8;
  EXPECT(yolo::copy_float_output(raw.data(), 64, plan) ==
         std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8}));
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 2, 2}, {64, 32, 12, 4}, 64, false, true);
  });
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 2, 2}, {64, 32, 12, 4}, 64, true, false);
  });
  rejects([&] { yolo::copy_float_output(raw.data(), 48, plan); });
  raw[3] = std::numeric_limits<float>::infinity();
  rejects([&] { yolo::copy_float_output(raw.data(), 64, plan); });
}
