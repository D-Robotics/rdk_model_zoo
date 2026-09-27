// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "geometry.h"
#include <cfenv>
#include <stdexcept>
#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
int main() {
  auto rounded = yoloe::make_geometry(1000, 333, yoloe::Protocol::E26);
  EXPECT(rounded.resized_w == 640 && rounded.resized_h == 213);
  EXPECT(rounded.top == 213 && rounded.bottom == 214);
  auto truncated = yoloe::make_geometry(1000, 333, yoloe::Protocol::E11);
  EXPECT(truncated.resized_h == 213);
  auto difference = yoloe::make_geometry(1000, 334, yoloe::Protocol::E26);
  EXPECT(difference.resized_h == 214);
  difference = yoloe::make_geometry(1000, 334, yoloe::Protocol::E11);
  EXPECT(difference.resized_h == 213);
  auto stretch = yoloe::make_geometry(1000, 334, yoloe::Protocol::E11, 0);
  EXPECT(stretch.resized_h == 640 && stretch.top == 0);
  auto narrow = yoloe::make_geometry(1000000, 1, yoloe::Protocol::E26);
  EXPECT(narrow.resized_h == 1);
  auto box = yoloe::restore_box({0, 213, 640, 426}, rounded);
  EXPECT((box == std::array<float, 4>{0, 0, 1000, 333}));
  bool rejected = false;
  try {
    yoloe::make_geometry(0, 1, yoloe::Protocol::E26);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  EXPECT(rejected);
  rejected = false;
  try {
    yoloe::make_geometry(10, 10, yoloe::Protocol::E26, 0);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  EXPECT(rejected);
  rounded.resized_h = 212;
  rejected = false;
  try {
    yoloe::validate_geometry(rounded);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  EXPECT(rejected);
  // Python round uses ties-to-even, independent of the C floating rounding
  // mode.
  EXPECT(yoloe::round_even(212.5) == 212);
  EXPECT(yoloe::round_even(213.5) == 214);
  std::fesetround(FE_UPWARD);
  EXPECT(yoloe::round_even(212.5) == 212);
  std::fesetround(FE_TONEAREST);
}
