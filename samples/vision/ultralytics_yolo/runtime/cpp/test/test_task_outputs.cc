// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "common/task_outputs.h"
#define expect(v)                           \
  do {                                      \
    if (!(v)) throw std::runtime_error(#v); \
  } while (0)
template <class F>
void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::invalid_argument&) {
    bad = true;
  }
  expect(bad);
}
int main() {
  for (bool segment : {false, true})
    for (int box_channels : {4, 64}) {
      std::vector<yolo::OutputShape> shapes;
      for (int stride : {8, 16, 32}) {
        shapes.push_back({64 / stride, 64 / stride, segment ? 80 : 1});
        shapes.push_back({64 / stride, 64 / stride, box_channels});
        shapes.push_back({64 / stride, 64 / stride, segment ? 32 : 51});
      }
      if (segment) shapes.push_back({16, 16, 32});
      std::reverse(shapes.begin(), shapes.end());
      auto plan = yolo::bind_task_heads(shapes, 64, 64, segment);
      expect(plan.direct_ltrb == (box_channels == 4));
      expect(shapes[plan.cls[0]].c == (segment ? 80 : 1));
      expect(shapes[plan.box[2]].h == 2);
      if (segment) expect(shapes[plan.prototype].h == 16);
      auto bad = shapes;
      bad[plan.box[1]].c = box_channels == 4 ? 64 : 4;
      rejects([&] { yolo::bind_task_heads(bad, 64, 64, segment); });
      bad = shapes;
      bad[plan.extra[0]] = bad[plan.cls[0]];
      rejects([&] { yolo::bind_task_heads(bad, 64, 64, segment); });
      rejects([&] { yolo::bind_task_heads(shapes, 63, 64, segment); });
      rejects([&] { yolo::bind_task_heads(shapes, 64, 128, segment); });
    }
  // Two rows, three cells, two channels, with unused cell/row padding.
  std::vector<float> raw(40, std::numeric_limits<float>::quiet_NaN());
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 2; ++c) raw[y * 20 + x * 4 + c] = y * 6 + x * 2 + c;
  auto plan =
      yolo::nhwc_float_plan({1, 2, 3, 2}, {160, 80, 16, 4}, 160, true, true);
  auto copied = yolo::copy_float_output(raw.data(), 160, plan);
  expect(copied.size() == 12);
  for (int i = 0; i < 12; ++i) expect(copied[i] == i);
  raw[0] = 99;
  expect(copied[0] == 0);
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 3, 2}, {160, 80, 4, 4}, 160, true, true);
  });
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 3, 2}, {160, 8, 16, 4}, 160, true, true);
  });
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 3, 2}, {160, 80, 16, 4}, 119, true, true);
  });
  rejects([&] {
    yolo::nhwc_float_plan({2, 2, 3, 2}, {160, 80, 16, 4}, 320, true, true);
  });
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 3, 2}, {160, 80, 16, 4}, 160, false, true);
  });
  rejects([&] {
    yolo::nhwc_float_plan({1, 2, 3, 2}, {160, 80, 16, 4}, 160, true, false);
  });
  rejects([&] { yolo::copy_float_output(raw.data(), 119, plan); });
  raw[4] = std::numeric_limits<float>::infinity();
  rejects([&] { yolo::copy_float_output(raw.data(), 160, plan); });
}
