/*
 * Copyright (c) 2026, D-Robotics.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host unit tests for the NV12 plane packing helpers
// (common/nv12_geometry.{h,cc}).

#include <cstdio>
#include <cstring>
#include <vector>

#include "common/nv12_geometry.h"

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
  const int h = 4;
  const int w = 4;
  // I420 planes: Y 4x4, U 2x2, V 2x2.
  const unsigned char y[16] = {0,  1,  2,  3,  4,  5,  6,  7,
                               8,  9,  10, 11, 12, 13, 14, 15};
  const unsigned char u[4] = {10, 11, 12, 13};
  const unsigned char v[4] = {20, 21, 22, 23};

  // Packed NV12: Y plane then interleaved UV.
  unsigned char packed[24] = {0};
  yolo::i420_to_packed_nv12(y, u, v, h, w, packed);
  expect_eq("packed y[0]", packed[0], 0);
  expect_eq("packed y[15]", packed[15], 15);
  expect_eq("packed uv u0", packed[16], 10);
  expect_eq("packed uv v0", packed[17], 20);
  expect_eq("packed uv u3", packed[22], 13);
  expect_eq("packed uv v3", packed[23], 23);

  // Split NV12 with row padding: y_stride 8 bytes (4 valid), uv_stride 8
  // bytes (4 valid: 2 pixels x u,v).
  const int y_stride = 8;
  const int uv_stride = 8;
  std::vector<unsigned char> y_dst(y_stride * h, 0xAA);
  std::vector<unsigned char> uv_dst(uv_stride * (h / 2), 0xAA);
  yolo::i420_to_split_nv12(y, u, v, h, w, y_dst.data(), y_stride,
                           uv_dst.data(), uv_stride);

  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < w; ++col) {
      expect_eq("split y sample", y_dst[row * y_stride + col], y[row * w + col]);
    }
    for (int pad = w; pad < y_stride; ++pad) {
      expect_eq("split y padding untouched", y_dst[row * y_stride + pad], 0xAA);
    }
  }
  // UV rows: 2x2 chroma -> row 0 at bytes 0..3 (u0,v0,u1,v1), row 1 at
  // bytes uv_stride..uv_stride+3 (u2,v2,u3,v3), padding in between.
  const unsigned char expected_row0[4] = {10, 20, 11, 21};
  const unsigned char expected_row1[4] = {12, 22, 13, 23};
  for (int i = 0; i < 4; ++i) {
    expect_eq("split uv row0", uv_dst[i], expected_row0[i]);
    expect_eq("split uv row1", uv_dst[uv_stride + i], expected_row1[i]);
  }
  for (int pad = 4; pad < uv_stride; ++pad) {
    expect_eq("split uv row0 padding untouched", uv_dst[pad], 0xAA);
    expect_eq("split uv row1 padding untouched",
              uv_dst[uv_stride + pad], 0xAA);
  }

  if (failures == 0) {
    std::printf("test_nv12_geometry: OK\n");
    return 0;
  }
  std::printf("test_nv12_geometry: %d failure(s)\n", failures);
  return 1;
}
