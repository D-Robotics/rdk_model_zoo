/*
 * Copyright (c) 2026, D-Robotics.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host unit tests for the oriented-box geometry (inline in inc/yolo.hpp).

#include <cmath>
#include <cstdio>

#include "yolo.hpp"

namespace {

int failures = 0;

void expect_near(const char* what, float actual, float expected, float tol) {
  if (!std::isfinite(actual) || std::fabs(actual - expected) > tol) {
    std::printf("FAIL %s: actual=%f expected=%f\n", what, actual, expected);
    ++failures;
  }
}

}  // namespace

int main() {
  const float ltrb[4] = {2.0f, 1.0f, 4.0f, 3.0f};
  yolo::RotatedBox box;
  if (!yolo::decode_obb_cell(ltrb, 0.0f, 0.5f, 0.5f, 8.0f, 1.0f, 0.0f,
                             &box)) {
    std::printf("FAIL decode_obb_cell rejected valid input\n");
    ++failures;
  }
  expect_near("center x", box.cx, 12.0f, 1e-6f);
  expect_near("center y", box.cy, 12.0f, 1e-6f);
  expect_near("width", box.width, 48.0f, 1e-6f);
  expect_near("height", box.height, 32.0f, 1e-6f);

  box.width = 2.0f;
  box.height = 4.0f;
  box.angle_rad = 0.2f;
  yolo::regularize_obb(&box, true, true);
  expect_near("regularized width", box.width, 4.0f, 1e-6f);
  expect_near("regularized height", box.height, 2.0f, 1e-6f);
  const float half_pi = static_cast<float>(std::acos(-1.0) * 0.5);
  if (box.angle_rad < -half_pi || box.angle_rad >= half_pi) {
    std::printf("FAIL wrapped angle outside [-pi/2, pi/2)\n");
    ++failures;
  }

  box.cx = 40.0f;
  box.cy = 30.0f;
  box.width = 20.0f;
  box.height = 10.0f;
  yolo::ImageTransform transform;
  transform.scale_x = 2.0f;
  transform.scale_y = 2.0f;
  transform.shift_x = 4;
  transform.shift_y = 2;
  if (!yolo::map_obb_to_source(&box, transform, 100, 80, true)) {
    std::printf("FAIL map_obb_to_source rejected valid transform\n");
    ++failures;
  }
  expect_near("mapped center x", box.cx, 18.0f, 1e-6f);
  expect_near("mapped center y", box.cy, 14.0f, 1e-6f);
  expect_near("mapped width", box.width, 10.0f, 1e-6f);
  expect_near("mapped height", box.height, 5.0f, 1e-6f);

  // The S-series policy keeps geometry outside the image instead of clipping.
  yolo::RotatedBox outside;
  outside.cx = 250.0f;
  outside.cy = 10.0f;
  outside.width = 40.0f;
  outside.height = 20.0f;
  if (!yolo::map_obb_to_source(&outside, transform, 100, 80, false)) {
    std::printf("FAIL map_obb_to_source rejected unclipped mapping\n");
    ++failures;
  }
  expect_near("unclipped center x", outside.cx, 123.0f, 1e-6f);
  expect_near("unclipped width", outside.width, 20.0f, 1e-6f);
  yolo::RotatedBox clipped;
  clipped.cx = 250.0f;
  clipped.cy = 10.0f;
  clipped.width = 40.0f;
  clipped.height = 20.0f;
  yolo::map_obb_to_source(&clipped, transform, 100, 80, true);
  expect_near("clipped center x", clipped.cx, 100.0f, 1e-6f);

  const float invalid[4] = {NAN, 1.0f, 2.0f, 3.0f};
  if (yolo::decode_obb_cell(invalid, 0.0f, 0.5f, 0.5f, 8.0f, 1.0f, 0.0f,
                            &box)) {
    std::printf("FAIL decode_obb_cell accepted non-finite input\n");
    ++failures;
  }
  if (failures == 0) {
    std::printf("test_obb_decode: OK\n");
    return 0;
  }
  std::printf("test_obb_decode: %d failure(s)\n", failures);
  return 1;
}
