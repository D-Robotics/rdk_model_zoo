// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// YOLO26-Depth X5 model: board identity, tensor contract, letterbox/NV12
// preprocess, warmup plus one timed forward, and the calibrated log-depth
// restore. DNN handles and buffers live in the private Impl in this file;
// the public header exposes only owned stage-data types.
#include "depth.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <opencv2/imgproc.hpp>

namespace yolo26_depth {
namespace {
void check(int code, const char *operation) {
  if (code)
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(code));
}
// Default-constructible, non-copyable task guard: releases on every
// exceptional path and never releases twice (the success path clears the
// handle after its own checked release).
struct TaskGuard {
  TaskGuard() = default;
  hbDNNTaskHandle_t handle = nullptr;
  ~TaskGuard() {
    if (handle)
      hbDNNReleaseTask(handle);
  }
  TaskGuard(const TaskGuard &) = delete;
  TaskGuard &operator=(const TaskGuard &) = delete;
};
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
TensorLayout output_layout(const hbDNNTensorProperties &p) {
  if (p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.quantiType != NONE ||
      p.validShape.numDimensions != 4 || p.alignedShape.numDimensions != 4 ||
      p.alignedByteSize <= 0)
    throw std::invalid_argument(
        "Expected float32 NONE-quantized four-dimensional depth output");
  TensorLayout layout;
  layout.capacity = static_cast<std::size_t>(p.alignedByteSize);
  for (int i = 0; i < 4; ++i) {
    const auto byte_stride = static_cast<long long>(p.stride[i]);
    if (p.validShape.dimensionSize[i] <= 0 ||
        p.alignedShape.dimensionSize[i] <= 0 || byte_stride < 0)
      throw std::invalid_argument("Invalid output dimensions/strides");
    layout.valid[i] = static_cast<std::size_t>(p.validShape.dimensionSize[i]);
    layout.aligned[i] =
        static_cast<std::size_t>(p.alignedShape.dimensionSize[i]);
    layout.strides[i] = static_cast<std::size_t>(p.stride[i]);
  }
  validated_strides(layout);
  return layout;
}
void validate_input(const hbDNNTensorProperties &p) {
  constexpr int bytes = kInputSize * kInputSize * 3 / 2;
  if (p.tensorType != HB_DNN_IMG_TYPE_NV12 || p.validShape.numDimensions != 4 ||
      p.alignedShape.numDimensions != 4 || p.alignedByteSize < bytes)
    throw std::invalid_argument(
        "Expected compact NV12 input with sufficient storage");
  std::array<int, 4> shape{};
  for (int i = 0; i < 4; ++i) {
    shape[i] = p.validShape.dimensionSize[i];
    if (p.alignedShape.dimensionSize[i] != shape[i])
      throw std::invalid_argument("Padded NV12 input geometry is not supported "
                                  "by this compact pyramid binding");
  }
  if (shape != std::array<int, 4>{1, 3, 768, 768} &&
      shape != std::array<int, 4>{1, 768, 768, 3})
    throw std::invalid_argument(
        "NV12 logical input must describe 768-square geometry");
}
} // namespace

// ===========================================================================
// Board identity
// ===========================================================================
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

// ===========================================================================
// Geometry and output tensor contract
// ===========================================================================
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

// ===========================================================================
// NV12 packing
// ===========================================================================
std::vector<std::uint8_t> pack_nv12(const cv::Mat &image) {
  if (image.empty() || image.type() != CV_8UC3 || image.rows % 2 ||
      image.cols % 2)
    throw std::invalid_argument(
        "NV12 conversion needs even BGR uint8 dimensions");
  cv::Mat i420;
  cv::cvtColor(image, i420, cv::COLOR_BGR2YUV_I420);
  if (!i420.isContinuous())
    i420 = i420.clone();
  const std::size_t area = static_cast<std::size_t>(image.rows) * image.cols;
  std::vector<std::uint8_t> packed(area * 3 / 2);
  std::memcpy(packed.data(), i420.data, area);
  const auto *u = i420.data + area;
  const auto *v = u + area / 4;
  for (std::size_t i = 0; i < area / 4; ++i) {
    packed[area + 2 * i] = u[i];
    packed[area + 2 * i + 1] = v[i];
  }
  return packed;
}

// ===========================================================================
// Yolo26Depth model
// ===========================================================================
struct Yolo26Depth::Impl {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;
  hbDNNTensor input{}, output{};
  bool input_allocated = false, output_allocated = false;
  TensorLayout layout;
  std::string name;
  ~Impl() {
    if (input_allocated)
      hbSysFreeMem(&input.sysMem[0]);
    if (output_allocated)
      hbSysFreeMem(&output.sysMem[0]);
    if (packed)
      hbDNNRelease(packed);
  }
  // One full forward: input copy + cache clean, infer, wait, output cache
  // invalidate, owned raw read.
  std::vector<float> run(const std::vector<std::uint8_t> &nv12) {
    if (nv12.size() != kInputSize * kInputSize * 3 / 2)
      throw std::invalid_argument("Incorrect compact NV12 byte count");
    std::memset(input.sysMem[0].virAddr, 0,
                input.properties.alignedByteSize);
    std::memcpy(input.sysMem[0].virAddr, nv12.data(), nv12.size());
    check(hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem input");
    TaskGuard task;
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    auto *output_tensor = &output;
    check(hbDNNInfer(&task.handle, &output_tensor, &input, model, &control),
          "hbDNNInfer");
    check(hbDNNWaitTaskDone(task.handle, 0), "hbDNNWaitTaskDone");
    check(hbSysFlushMem(&output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
          "hbSysFlushMem output");
    auto values = read_log_depth(output.sysMem[0].virAddr, layout);
    // Release the finished task and report a failed release; the guard
    // covers every exceptional path above and never releases twice.
    const int code = hbDNNReleaseTask(task.handle);
    task.handle = nullptr;
    check(code, "hbDNNReleaseTask");
    return values;
  }
};

Yolo26Depth::Yolo26Depth(const std::string &path, const DepthOptions &options,
                         ExecutionGate gate)
    : impl_(std::make_unique<Impl>()) {
  (gate ? gate : require_x5_board)();
  if (options.warmup < 0)
    throw std::invalid_argument("warmup must be a nonnegative integer");
  warmup_ = options.warmup;
  if (!std::filesystem::is_regular_file(path) ||
      !std::filesystem::file_size(path))
    throw std::invalid_argument("Missing or empty BIN model");
  const char *file = path.c_str();
  check(hbDNNInitializeFromFiles(&impl_->packed, &file, 1),
        "hbDNNInitializeFromFiles");
  const char **names = nullptr;
  int count = 0;
  check(hbDNNGetModelNameList(&names, &count, impl_->packed),
        "hbDNNGetModelNameList");
  if (count != 1 || !names || !names[0])
    throw std::invalid_argument("Expected exactly one named model");
  impl_->name = names[0];
  check(hbDNNGetModelHandle(&impl_->model, impl_->packed, names[0]),
        "hbDNNGetModelHandle");
  check(hbDNNGetInputCount(&count, impl_->model), "hbDNNGetInputCount");
  if (count != 1)
    throw std::invalid_argument("Expected exactly one NV12 input");
  check(hbDNNGetOutputCount(&count, impl_->model), "hbDNNGetOutputCount");
  if (count != 1)
    throw std::invalid_argument("Expected exactly one depth output");
  check(
      hbDNNGetInputTensorProperties(&impl_->input.properties, impl_->model, 0),
      "hbDNNGetInputTensorProperties");
  check(hbDNNGetOutputTensorProperties(&impl_->output.properties, impl_->model,
                                       0),
        "hbDNNGetOutputTensorProperties");
  validate_input(impl_->input.properties);
  impl_->layout = output_layout(impl_->output.properties);
  check(hbSysAllocCachedMem(&impl_->input.sysMem[0],
                            impl_->input.properties.alignedByteSize),
        "hbSysAllocCachedMem input");
  impl_->input_allocated = true;
  check(hbSysAllocCachedMem(&impl_->output.sysMem[0],
                            impl_->output.properties.alignedByteSize),
        "hbSysAllocCachedMem output");
  impl_->output_allocated = true;
}
Yolo26Depth::~Yolo26Depth() = default;
const std::string &Yolo26Depth::model_name() const { return impl_->name; }

PreparedInput Yolo26Depth::preprocess(const cv::Mat &image) const {
  if (image.empty() || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected a nonempty BGR uint8 HWC image");
  const auto context = letterbox_geometry(image.rows, image.cols);
  const int width = kInputSize - context.left - context.right,
            height = kInputSize - context.top - context.bottom;
  cv::Mat resized, padded;
  if (image.rows == height && image.cols == width)
    resized = image;
  else
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
  cv::copyMakeBorder(resized, padded, context.top, context.bottom, context.left,
                     context.right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  return {pack_nv12(padded), context};
}

// Exact warmup/timing contract: `warmup` unmeasured full forwards, then one
// timed full forward; the timer brackets the forward only (buffer copy,
// cache operations, SDK run, raw output copy), never the restore.
RawDepth Yolo26Depth::infer(const PreparedInput &prepared) {
  for (int i = 0; i < warmup_; ++i)
    impl_->run(prepared.nv12);
  const auto start = std::chrono::steady_clock::now();
  auto values = impl_->run(prepared.nv12);
  const auto end = std::chrono::steady_clock::now();
  return {std::move(values),
          {std::chrono::duration<double, std::milli>(end - start).count(),
           warmup_}};
}

DepthResult Yolo26Depth::postprocess(const RawDepth &raw,
                                     const ImageContext &context) const {
  validate_context(context);
  if (raw.values.size() != kOutputSize * kOutputSize ||
      !std::all_of(raw.values.begin(), raw.values.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument(
        "Expected finite raw float32 192-square calibrated log-depth");
  cv::Mat log(kOutputSize, kOutputSize, CV_32FC1);
  std::copy(raw.values.begin(), raw.values.end(), log.ptr<float>());
  cv::Mat depth, square, restored;
  cv::exp(log, depth);
  if (!cv::checkRange(depth))
    throw std::invalid_argument("Depth exponential overflow");
  cv::resize(depth, square, cv::Size(kInputSize, kInputSize), 0, 0,
             cv::INTER_LINEAR);
  const cv::Rect area(context.left, context.top,
                      kInputSize - context.left - context.right,
                      kInputSize - context.top - context.bottom);
  cv::resize(square(area), restored,
             cv::Size(context.original_width, context.original_height), 0, 0,
             cv::INTER_LINEAR);
  return {log, restored, context, raw.run};
}
DepthResult Yolo26Depth::predict(const cv::Mat &image) {
  const auto prepared = preprocess(image);
  return postprocess(infer(prepared), prepared.context);
}
} // namespace yolo26_depth
