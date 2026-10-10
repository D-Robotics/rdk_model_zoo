// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace yolo26_depth {

constexpr int kInputSize = 768;
constexpr int kOutputSize = 192;

struct ImageContext {
  int original_height = 0, original_width = 0;
  int top = 0, bottom = 0, left = 0, right = 0;
};

struct TensorLayout {
  std::array<std::size_t, 4> valid{};
  std::array<std::size_t, 4> aligned{};
  std::array<std::size_t, 4>
      strides{}; // byte strides; all zero means derive from aligned shape
  std::size_t capacity = 0;
};

// Timing provenance carried beside the measured values: latency covers one
// full forward including buffer copy, cache operations, SDK run and raw
// output copy (not BPU-only), after exactly `warmup` unmeasured forwards.
struct RunMetadata {
  double latency_ms = 0.0;
  int warmup = 0;
};

struct PreparedInput {
  std::vector<std::uint8_t> nv12;
  ImageContext context;
};

struct RawDepth {
  std::vector<float> values;
  RunMetadata run;
};

struct DepthResult {
  cv::Mat log_depth;
  cv::Mat depth_native;
  ImageContext context;
  RunMetadata run;
};

struct DepthOptions {
  int warmup = 0;
};

ImageContext letterbox_geometry(int height, int width);
void validate_context(const ImageContext &context);
std::array<std::size_t, 4> validated_strides(const TensorLayout &layout);
std::vector<float> read_log_depth(const void *bytes,
                                  const TensorLayout &layout);
double percentile(std::vector<float> values, double fraction);
std::vector<std::uint8_t> pack_nv12(const cv::Mat &image);
bool match_x5_identity(std::string boardinfo, std::string socinfo,
                       std::string device_tree);
void require_x5_board();

// YOLO26-Depth X5 model: letterbox/NV12 preprocess, warmup plus one timed
// SDK forward, calibrated log-depth restore. The constructor loads the
// runtime; SDK handles and buffers live in the private Impl in depth.cpp.
// Not thread-safe: concurrent tasks require independent instances.
class Yolo26Depth {
public:
  using ExecutionGate = std::function<void()>;
  explicit Yolo26Depth(const std::string &path,
                       const DepthOptions &options = {},
                       ExecutionGate gate = {});
  ~Yolo26Depth();
  Yolo26Depth(const Yolo26Depth &) = delete;
  Yolo26Depth &operator=(const Yolo26Depth &) = delete;

  PreparedInput preprocess(const cv::Mat &image) const;
  RawDepth infer(const PreparedInput &prepared);
  DepthResult postprocess(const RawDepth &raw,
                          const ImageContext &context) const;
  DepthResult predict(const cv::Mat &image);
  const std::string &model_name() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int warmup_ = 0;
};

} // namespace yolo26_depth
