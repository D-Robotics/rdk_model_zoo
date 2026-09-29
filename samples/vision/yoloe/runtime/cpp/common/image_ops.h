// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "candidate.h"
#include "geometry.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
namespace yoloe {
struct PreparedBGR {
  cv::Mat pixels;
  Geometry geometry;
};
inline PreparedBGR prepare_bgr(const cv::Mat &image, Protocol protocol,
                               int resize_type = 1) {
  if (image.empty() || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected nonempty BGR uint8 image");
  auto geometry = make_geometry(image.cols, image.rows, protocol, resize_type);
  cv::Mat resized, pixels;
  cv::resize(image, resized, {geometry.resized_w, geometry.resized_h}, 0, 0,
             resize_type == 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
  int value = protocol == Protocol::E26 ? 114 : 127;
  cv::copyMakeBorder(resized, pixels, geometry.top, geometry.bottom,
                     geometry.left, geometry.right, cv::BORDER_CONSTANT,
                     cv::Scalar(value, value, value));
  return {pixels, geometry};
}
struct RestoredMask {
  std::array<float, 4> box;
  cv::Mat mask;
};
inline std::vector<RestoredMask>
restore_e26_masks(const std::vector<RawDetection> &candidates,
                  const std::vector<float> &proto, const Geometry &geometry) {
  validate_geometry(geometry);
  if (geometry.protocol != Protocol::E26 || proto.size() != 160 * 160 * 32)
    throw std::invalid_argument(
        "E26 masks require matching geometry and 160x160x32 prototype");
  for (float value : proto)
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite prototype");
  std::vector<RestoredMask> result;
  result.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    auto box = restore_box(candidate.box, geometry);
    for (float value : candidate.coefficients)
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite mask coefficient");
    cv::Mat raw(160, 160, CV_32FC1);
    for (int y = 0; y < 160; ++y)
      for (int x = 0; x < 160; ++x) {
        float sum = 0;
        const float *pixel = proto.data() + (y * 160 + x) * 32;
        for (int c = 0; c < 32; ++c)
          sum += pixel[c] * candidate.coefficients[c];
        if (!std::isfinite(sum))
          throw std::invalid_argument("Mask combination overflow");
        raw.at<float>(y, x) = sum;
      }
    cv::Mat canvas, binary(640, 640, CV_8UC1, cv::Scalar(0));
    cv::resize(raw, canvas, {640, 640}, 0, 0, cv::INTER_LINEAR);
    for (int y = 0; y < 640; ++y)
      for (int x = 0; x < 640; ++x)
        binary.at<unsigned char>(y, x) =
            canvas.at<float>(y, x) > 0 && x >= candidate.box[0] &&
            x < candidate.box[2] && y >= candidate.box[1] &&
            y < candidate.box[3];
    cv::Mat content = binary(cv::Rect(geometry.left, geometry.top,
                                      geometry.resized_w, geometry.resized_h));
    cv::Mat full;
    cv::resize(content, full, {geometry.width, geometry.height}, 0, 0,
               cv::INTER_NEAREST);
    auto bound = [](float value, int limit) {
      return static_cast<int>(std::clamp(static_cast<double>(value), 0.0,
                                         static_cast<double>(limit)));
    };
    int x1 = bound(box[0], geometry.width), x2 = bound(box[2], geometry.width);
    int y1 = bound(box[1], geometry.height),
        y2 = bound(box[3], geometry.height);
    cv::Mat mask(std::max(y2 - y1, 0), std::max(x2 - x1, 0), CV_8UC1);
    if (x2 > x1 && y2 > y1)
      mask = full(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
    result.push_back({box, mask});
  }
  return result;
}
// S E11 uses a cropped prototype binary mask, unlike E26 canvas logits.
inline std::vector<RestoredMask>
restore_e11_masks(const std::vector<RawDetection> &candidates,
                  const std::vector<float> &proto, const Geometry &geometry,
                  bool do_morph = false) {
  validate_geometry(geometry);
  if (geometry.protocol != Protocol::E11 || proto.size() != 160 * 160 * 32)
    throw std::invalid_argument(
        "E11 masks require matching geometry and 160x160x32 prototype");
  for (float value : proto)
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite prototype");
  std::vector<RestoredMask> result;
  result.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    auto box = restore_box(candidate.box, geometry);
    for (float value : candidate.coefficients)
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite mask coefficient");
    // Clip to actual image content before the prototype crop, excluding
    // padding.
    int left = static_cast<int>(
        std::clamp(candidate.box[0], static_cast<float>(geometry.left),
                   static_cast<float>(640 - geometry.right)) *
        0.25f);
    int right = static_cast<int>(
        std::clamp(candidate.box[2], static_cast<float>(geometry.left),
                   static_cast<float>(640 - geometry.right)) *
        0.25f);
    int top = static_cast<int>(
        std::clamp(candidate.box[1], static_cast<float>(geometry.top),
                   static_cast<float>(640 - geometry.bottom)) *
        0.25f);
    int bottom = static_cast<int>(
        std::clamp(candidate.box[3], static_cast<float>(geometry.top),
                   static_cast<float>(640 - geometry.bottom)) *
        0.25f);
    auto bound = [](float value, int limit) {
      return static_cast<int>(std::clamp(static_cast<double>(value), 0.0,
                                         static_cast<double>(limit)));
    };
    int width = std::max(0, bound(box[2], geometry.width) -
                                bound(box[0], geometry.width));
    int height = std::max(0, bound(box[3], geometry.height) -
                                 bound(box[1], geometry.height));
    cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
    if (width && height && right > left && bottom > top) {
      cv::Mat cropped(bottom - top, right - left, CV_8UC1);
      for (int y = top; y < bottom; ++y)
        for (int x = left; x < right; ++x) {
          const float *pixel = proto.data() + (y * 160 + x) * 32;
          float sum = 0;
          for (int c = 0; c < 32; ++c)
            sum += pixel[c] * candidate.coefficients[c];
          if (!std::isfinite(sum))
            throw std::invalid_argument("Mask combination overflow");
          cropped.at<unsigned char>(y - top, x - left) = sum > 0.5f;
        }
      cv::resize(cropped, mask, {width, height}, 0, 0, cv::INTER_LANCZOS4);
      if (do_morph)
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN,
                         cv::Mat::ones(5, 5, CV_8UC1));
      // Lanczos may overshoot to 2; retain foreground support as binary 0/1.
      cv::threshold(mask, mask, 0, 1, cv::THRESH_BINARY);
    }
    result.push_back({box, mask});
  }
  return result;
}
} // namespace yoloe
