/*
 * Copyright (c) 2026, D-Robotics.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host unit tests for output discovery by shape (common/tensor_view.h).

#include <cstdio>
#include <vector>

#include "common/tensor_view.h"

namespace {

int failures = 0;

void expect_eq(const char* what, int actual, int expected) {
  if (actual != expected) {
    std::printf("FAIL %s: actual=%d expected=%d\n", what, actual, expected);
    ++failures;
  }
}

}  // namespace

int main() {
  std::vector<yolo::OutputShape> outputs = {
      {80, 80, 80}, {80, 80, 4}, {40, 40, 80}, {40, 40, 4},
      {20, 20, 80}, {20, 20, 4},
  };

  expect_eq("cls@80", yolo::find_output_by_shape(outputs, 80, 80, 80), 0);
  expect_eq("ltrb box@80", yolo::find_output_by_shape(outputs, 80, 80, 4), 1);
  expect_eq("cls@20", yolo::find_output_by_shape(outputs, 20, 20, 80), 4);
  expect_eq("missing", yolo::find_output_by_shape(outputs, 80, 80, 64), -1);

  // DFL variant of the same layout.
  outputs[1].c = 64;
  outputs[3].c = 64;
  outputs[5].c = 64;
  expect_eq("dfl box@80", yolo::find_output_by_shape(outputs, 80, 80, 64), 1);
  expect_eq("ltrb box now missing", yolo::find_output_by_shape(outputs, 80, 80, 4), -1);

  // Ambiguous layouts are rejected, mirroring the detect sample.
  outputs.push_back({80, 80, 80});
  expect_eq("ambiguous", yolo::find_output_by_shape(outputs, 80, 80, 80), -1);

  // TensorView addressing honours the aligned shape.
  float data[32] = {0.0f};  // room for 2 rows x aligned_w=4 cells of 4 floats
  for (int i = 0; i < 32; ++i) data[i] = static_cast<float>(i);
  yolo::TensorView view;
  view.data = data;
  view.h = 2;
  view.w = 2;
  view.channels = 4;
  view.row_step = 8;
  view.cell_step = 4;
  expect_eq("cell(1,0)[0]", static_cast<int>(view.cell(1, 0)[0]), 8);
  expect_eq("cell(1,1)[2]", static_cast<int>(view.cell(1, 1)[2]), 14);

  // Row padding: row_step wider than the valid width.
  view.row_step = 16;
  expect_eq("padded cell(1,0)[0]", static_cast<int>(view.cell(1, 0)[0]), 16);

  if (failures == 0) {
    std::printf("test_head_probe: OK\n");
    return 0;
  }
  std::printf("test_head_probe: %d failure(s)\n", failures);
  return 1;
}
