// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "tensor_contract.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace yolo26_depth {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("Tensor extent overflow");
  return a * b;
}
int round_even(double value) {
  const auto floor_value = std::floor(value);
  const auto fraction = value - floor_value;
  return static_cast<int>(
      floor_value +
      (fraction > .5 || (fraction == .5 && std::fmod(floor_value, 2) != 0)));
}
std::string trim(std::string text) {
  const auto whitespace = [](unsigned char c) {
    return c == 0 || std::isspace(c);
  };
  while (!text.empty() && whitespace(text.back()))
    text.pop_back();
  const auto first = std::find_if_not(text.begin(), text.end(), whitespace);
  text.erase(text.begin(), first);
  return text;
}
std::string lower(std::string text) {
  text = trim(text);
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return text;
}
std::string read(const char *path) {
  std::ifstream stream(path);
  return std::string(std::istreambuf_iterator<char>(stream), {});
}
} // namespace
ImageContext letterbox_geometry(int height, int width) {
  if (height <= 0 || width <= 0)
    throw std::invalid_argument("Image dimensions must be positive");
  const double ratio =
      std::min(double(kInputSize) / height, double(kInputSize) / width);
  const int h = round_even(height * ratio), w = round_even(width * ratio);
  if (h <= 0 || w <= 0)
    throw std::invalid_argument("Aspect ratio collapses letterbox dimension");
  const int ph = kInputSize - h, pw = kInputSize - w;
  return {height,
          width,
          round_even(ph / 2.0 - .1),
          round_even(ph / 2.0 + .1),
          round_even(pw / 2.0 - .1),
          round_even(pw / 2.0 + .1)};
}
void validate_context(const ImageContext &c) {
  const auto expected = letterbox_geometry(c.original_height, c.original_width);
  if (c.top != expected.top || c.bottom != expected.bottom ||
      c.left != expected.left || c.right != expected.right)
    throw std::invalid_argument(
        "Context padding does not match source geometry");
}
std::array<std::size_t, 4> validated_strides(const TensorLayout &layout) {
  const std::array<std::size_t, 4> nhwc{1, 192, 192, 1}, nchw{1, 1, 192, 192};
  if (layout.valid != nhwc && layout.valid != nchw)
    throw std::invalid_argument(
        "Expected one float32 192-square depth channel");
  for (int i = 0; i < 4; ++i)
    if (layout.aligned[i] < layout.valid[i])
      throw std::invalid_argument("Aligned shape is smaller than valid shape");
  auto strides = layout.strides;
  if (std::all_of(strides.begin(), strides.end(),
                  [](auto v) { return v == 0; })) {
    strides[3] = sizeof(float);
    for (int i = 2; i >= 0; --i)
      strides[i] = multiply(strides[i + 1], layout.aligned[i + 1]);
  }
  for (int i = 0; i < 4; ++i)
    if (!strides[i] || strides[i] % sizeof(float))
      throw std::invalid_argument("Invalid or mixed-zero output byte strides");
  if (strides[3] < sizeof(float))
    throw std::invalid_argument("Output element stride too small");
  for (int i = 2; i >= 0; --i)
    if (strides[i] < multiply(strides[i + 1], layout.aligned[i + 1]))
      throw std::invalid_argument("Output byte strides overlap");
  if (multiply(strides[0], layout.aligned[0]) > layout.capacity)
    throw std::invalid_argument("Output aligned extent exceeds allocation");
  return strides;
}
std::vector<float> read_log_depth(const void *bytes,
                                  const TensorLayout &layout) {
  const auto stride = validated_strides(layout);
  if (!bytes)
    throw std::invalid_argument("Null output buffer");
  const bool nhwc = layout.valid[3] == 1;
  std::vector<float> values(kOutputSize * kOutputSize);
  const auto *source = static_cast<const unsigned char *>(bytes);
  for (std::size_t h = 0; h < kOutputSize; ++h)
    for (std::size_t w = 0; w < kOutputSize; ++w) {
      const auto offset =
          nhwc ? h * stride[1] + w * stride[2] : h * stride[2] + w * stride[3];
      float value = 0;
      std::memcpy(&value, source + offset, sizeof(value));
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite raw depth output");
      values[h * kOutputSize + w] = value;
    }
  return values;
}
double percentile(std::vector<float> values, double fraction) {
  if (values.empty() || !std::isfinite(fraction) || fraction < 0 ||
      fraction > 1 || !std::all_of(values.begin(), values.end(), [](float v) {
        return std::isfinite(v);
      }))
    throw std::invalid_argument(
        "Percentile requires finite values and fraction in [0,1]");
  std::sort(values.begin(), values.end());
  const double rank = (values.size() - 1) * fraction;
  const auto low = static_cast<std::size_t>(std::floor(rank));
  const auto high = static_cast<std::size_t>(std::ceil(rank));
  return double(values[low]) +
         (double(values[high]) - values[low]) * (rank - low);
}
bool match_x5_identity(std::string boardinfo, std::string socinfo,
                       std::string device_tree) {
  boardinfo = lower(boardinfo);
  socinfo = lower(socinfo);
  device_tree = trim(device_tree);
  if (!boardinfo.empty())
    return boardinfo == "x5";
  if (!socinfo.empty())
    return socinfo == "x5u" || socinfo == "x5h" || socinfo == "x5m";
  return device_tree == "D-Robotics RDK X5 V1.0";
}
void require_x5_board() {
  if (!match_x5_identity(read("/sys/class/boardinfo/soc_name"),
                         read("/sys/class/socinfo/soc_name"),
                         read("/proc/device-tree/model")))
    throw std::invalid_argument(
        "This native runtime requires an identified RDK X5 board");
}
} // namespace yolo26_depth
