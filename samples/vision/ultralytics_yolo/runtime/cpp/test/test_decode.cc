/*
 * Copyright (c) 2026, D-Robotics.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host unit tests for the shared box-decode primitives (common/decode.h).

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "common/decode.h"

namespace {

int failures = 0;

void expect_near(const char* what, float actual, float expected, float tol) {
  if (std::fabs(actual - expected) > tol) {
    std::printf("FAIL %s: actual=%f expected=%f\n", what, actual, expected);
    ++failures;
  }
}

void expect_true(const char* what, bool condition) {
  if (!condition) {
    std::printf("FAIL %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  // Direct-LTRB boxes are consumed verbatim.
  const float box4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float ltrb[4];
  yolo::decode_box_ltrb(box4, ltrb);
  expect_near("ltrb[0]", ltrb[0], 1.0f, 1e-6f);
  expect_near("ltrb[3]", ltrb[3], 4.0f, 1e-6f);

  // Grid-to-image box math.
  float x1, y1, x2, y2;
  yolo::box_from_distances(5.5f, 3.5f, ltrb, 16.0f, &x1, &y1, &x2, &y2);
  expect_near("x1", x1, (5.5f - 1.0f) * 16.0f, 1e-4f);
  expect_near("y1", y1, (3.5f - 2.0f) * 16.0f, 1e-4f);
  expect_near("x2", x2, (5.5f + 3.0f) * 16.0f, 1e-4f);
  expect_near("y2", y2, (3.5f + 4.0f) * 16.0f, 1e-4f);

  // DFL: a dominant bin at index 5 decodes to a distance of ~5.
  float bins[yolo::kDflBins] = {0.0f};
  bins[5] = 20.0f;
  yolo::decode_box_dfl(bins, ltrb);
  expect_near("dfl one-hot", ltrb[0], 5.0f, 1e-3f);

  // DFL: two equally-weighted bins at 2 and 6 decode to 4.
  for (int j = 0; j < yolo::kDflBins; ++j) bins[j] = -100.0f;
  bins[2] = 0.0f;
  bins[6] = 0.0f;
  yolo::decode_box_dfl(bins, ltrb);
  expect_near("dfl two-peak", ltrb[0], 4.0f, 1e-3f);

  // Threshold inversion matches sigmoid.
  expect_near("raw logit threshold", yolo::raw_logit_threshold(0.25f),
              -std::log(3.0f), 1e-5f);
  expect_true("threshold 0 is -inf",
              yolo::raw_logit_threshold(0.0f) ==
                  -std::numeric_limits<float>::infinity());
  expect_true("threshold 1 is +inf",
              yolo::raw_logit_threshold(1.0f) ==
                  std::numeric_limits<float>::infinity());
  expect_near("sigmoid(0)", yolo::sigmoid(0.0f), 0.5f, 1e-6f);
  expect_near("sigmoid(-4)", yolo::sigmoid(-4.0f), 0.0179862f, 1e-5f);

  // Channel-count dispatch.
  expect_true("4ch -> direct LTRB",
              yolo::box_decode_from_channels(4) == yolo::BoxDecode::kDirectLtrb);
  expect_true("64ch -> DFL",
              yolo::box_decode_from_channels(64) == yolo::BoxDecode::kDfl);
  expect_true("7ch -> unknown",
              yolo::box_decode_from_channels(7) == yolo::BoxDecode::kUnknown);

  // Transform mapping back to source coordinates.
  yolo::ImageTransform transform;
  transform.scale_x = 2.0f;
  transform.scale_y = 2.0f;
  transform.shift_x = 10;
  transform.shift_y = 4;
  float sx1 = 20.0f, sy1 = 8.0f, sx2 = 30.0f, sy2 = 44.0f;
  expect_true("map_to_source keeps a valid box",
              yolo::map_to_source(&sx1, &sy1, &sx2, &sy2, transform, 640, 480));
  expect_near("sx1", sx1, 5.0f, 1e-5f);
  expect_near("sy1", sy1, 2.0f, 1e-5f);
  expect_near("sx2", sx2, 10.0f, 1e-5f);
  expect_near("sy2", sy2, 20.0f, 1e-5f);

  if (failures == 0) {
    std::printf("test_decode: OK\n");
    return 0;
  }
  std::printf("test_decode: %d failure(s)\n", failures);
  return 1;
}
