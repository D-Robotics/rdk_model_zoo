// SPDX-License-Identifier: Apache-2.0
// Separable antialiased bicubic resampling matching Pillow's uint8 RGB policy.
// Algorithm reference: python-pillow/Pillow 11.3.0, src/libImaging/Resample.c.
#pragma once
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mobilenet {
struct FilterRow { int start; std::vector<int32_t> weights; };
inline double cubic(double x) {
  x = std::abs(x);
  if (x < 1) return ((1.5 * x - 2.5) * x) * x + 1;
  if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
  return 0;
}
inline std::vector<FilterRow> filters(int input, int output) {
  std::vector<FilterRow> rows(output);
  const double ratio = double(input) / output, scale = std::max(1.0, ratio);
  for (int i = 0; i < output; ++i) {
    double center = (i + 0.5) * ratio;
    int lo = std::max(0, int(center - 2 * scale + 0.5));
    int hi = std::min(input, int(center + 2 * scale + 0.5));
    std::vector<double> values;
    double sum = 0;
    for (int j = lo; j < hi; ++j) { values.push_back(cubic((j + 0.5 - center) / scale)); sum += values.back(); }
    rows[i].start = lo;
    for (double v : values) rows[i].weights.push_back(int32_t(std::round(v / sum * (1 << 22))));
  }
  return rows;
}
/** Resize a decoded CV_8UC3 image and center-crop, preserving channel order. */
inline cv::Mat center_crop(const cv::Mat& src, int size = 224, int shorter = 256) {
  if (src.empty() || src.type() != CV_8UC3) throw std::runtime_error("Expected decoded uint8 three-channel image");
  int width = src.cols <= src.rows ? shorter : int(double(shorter) * src.cols / src.rows);
  int height = src.cols <= src.rows ? int(double(shorter) * src.rows / src.cols) : shorter;
  cv::Mat horizontal, resized;
  if (width == src.cols) horizontal = src;
  else {
    auto f = filters(src.cols, width); horizontal.create(src.rows, width, CV_8UC3);
    cv::parallel_for_(cv::Range(0, src.rows), [&](const cv::Range& r) {
      for (int y = r.start; y < r.end; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        int64_t sum = 1 << 21;
        for (size_t k = 0; k < f[x].weights.size(); ++k) sum += int64_t(src.ptr<uint8_t>(y)[3 * (f[x].start + k) + c]) * f[x].weights[k];
        horizontal.ptr<uint8_t>(y)[3 * x + c] = uint8_t(std::clamp<int64_t>(sum >> 22, 0, 255));
      }
    });
  }
  if (height == src.rows) resized = horizontal;
  else {
    auto f = filters(src.rows, height); resized.create(height, width, CV_8UC3);
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& r) {
      for (int y = r.start; y < r.end; ++y) for (int x = 0; x < width * 3; ++x) {
        int64_t sum = 1 << 21;
        for (size_t k = 0; k < f[y].weights.size(); ++k) sum += int64_t(horizontal.ptr<uint8_t>(f[y].start + k)[x]) * f[y].weights[k];
        resized.ptr<uint8_t>(y)[x] = uint8_t(std::clamp<int64_t>(sum >> 22, 0, 255));
      }
    });
  }
  // nearbyint uses round-to-even, as does Python round for center-crop offsets.
  return resized(cv::Rect(int(std::nearbyint((width - size) / 2.0)),
                         int(std::nearbyint((height - size) / 2.0)), size, size)).clone();
}
/** Convert a prepared BGR crop to packed NV12, using the shared Python policy. */
inline std::vector<uint8_t> nv12(const cv::Mat& bgr) {
  cv::Mat planar; cv::cvtColor(bgr, planar, cv::COLOR_BGR2YUV_I420);
  const size_t pixels = bgr.total(); std::vector<uint8_t> result(pixels * 3 / 2);
  std::copy(planar.data, planar.data + pixels, result.data());
  for (size_t i = 0; i < pixels / 4; ++i) {
    result[pixels + 2 * i] = planar.data[pixels + i];
    result[pixels + 2 * i + 1] = planar.data[pixels + pixels / 4 + i];
  }
  return result;
}
}  // namespace mobilenet
