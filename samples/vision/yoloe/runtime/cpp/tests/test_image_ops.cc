// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "image_ops.h"
#include <limits>
#include <stdexcept>
#define EXPECT(v)                                                              \
  do {                                                                         \
    if (!(v))                                                                  \
      throw std::runtime_error(#v);                                            \
  } while (0)
template <class F> void rejects(F fn) {
  bool bad = false;
  try {
    fn();
  } catch (const std::invalid_argument &) {
    bad = true;
  }
  EXPECT(bad);
}
int main() {
  cv::Mat image(334, 1000, CV_8UC3, cv::Scalar(1, 2, 3));
  auto prepared = yoloe::prepare_bgr(image, yoloe::Protocol::E26);
  EXPECT(prepared.geometry.resized_h == 214);
  EXPECT(prepared.pixels.rows == 640);
  EXPECT(prepared.pixels.at<cv::Vec3b>(0, 0)[0] == 114);
  auto e11 = yoloe::prepare_bgr(image, yoloe::Protocol::E11);
  EXPECT(e11.geometry.resized_h == 213);
  EXPECT(e11.pixels.at<cv::Vec3b>(0, 0)[0] == 127);
  auto stretch = yoloe::prepare_bgr(image, yoloe::Protocol::E11, 0);
  EXPECT(stretch.pixels.at<cv::Vec3b>(0, 0)[0] == 1);
  rejects([&] { yoloe::prepare_bgr(cv::Mat(), yoloe::Protocol::E26); });
  std::vector<float> proto(160 * 160 * 32, 0);
  for (size_t i = 0; i < proto.size(); i += 32)
    proto[i] = 1;
  yoloe::RawDetection candidate;
  candidate.box = {0, 213, 640, 427};
  candidate.coefficients[0] = 1;
  auto masks = yoloe::restore_e26_masks({candidate}, proto, prepared.geometry);
  EXPECT(masks.size() == 1);
  EXPECT((masks[0].box == std::array<float, 4>{0, 0, 1000, 334}));
  EXPECT(masks[0].mask.rows == 334 && masks[0].mask.cols == 1000);
  EXPECT(cv::countNonZero(masks[0].mask) == 334000);
  candidate.box = {0, 200, 64, 210};
  masks = yoloe::restore_e26_masks({candidate}, proto, prepared.geometry);
  EXPECT(masks[0].mask.rows == 0 && masks[0].mask.cols == 100);
  candidate.box = {-10, -10, -5, -5};
  masks = yoloe::restore_e26_masks({candidate}, proto, prepared.geometry);
  EXPECT(masks[0].mask.empty());
  candidate.box = {0, 213, 640, 427};
  candidate.coefficients[0] = -1;
  masks = yoloe::restore_e26_masks({candidate}, proto, prepared.geometry);
  EXPECT(cv::countNonZero(masks[0].mask) == 0);
  rejects([&] { yoloe::restore_e26_masks({candidate}, proto, e11.geometry); });
  proto.pop_back();
  rejects(
      [&] { yoloe::restore_e26_masks({candidate}, proto, prepared.geometry); });
  proto.push_back(0);
  candidate.coefficients[0] = std::numeric_limits<float>::max();
  proto[0] = 2;
  rejects(
      [&] { yoloe::restore_e26_masks({candidate}, proto, prepared.geometry); });
  candidate.coefficients[0] = 1;
  proto[0] = std::numeric_limits<float>::quiet_NaN();
  rejects(
      [&] { yoloe::restore_e26_masks({candidate}, proto, prepared.geometry); });
}
