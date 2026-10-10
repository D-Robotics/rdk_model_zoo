/**
 * @file gemma4_vision_engine.cpp
 * @brief Complete Vision path: preprocessing, ViT execution and tensor
 *        transport for Gemma4-E2B.
 *
 * The preprocessing stage prepares in-memory pixels in the layout expected
 * by the compiled vision HBM; the engine consumes prepared patches, submits
 * BPU inference, and converts the encoder output into features consumed by
 * the text runtime through the fixed-export descriptor contract below.
 *
 * @note VisionEngine instances are not thread-safe.
 */

#include "gemma4_vision_engine.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "gemma4_config.hpp"
#include "hb_utils.hpp"

namespace gemma4 {

VisionEngine::VisionEngine(const std::string &vision_hbm) {
  try {
    const char *path = vision_hbm.c_str();
    const auto start = std::chrono::steady_clock::now();
    HBDNN_CHECK(hbDNNInitializeFromFiles(&packed_, &path, 1),
                "load vision hbm");
    if (!packed_)
      throw std::runtime_error("SDK returned null packed model");
    load_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();
    HBDNN_CHECK(hbDNNGetModelHandle(&handle_, packed_, "Gemma4VisionModel"),
                "get Gemma4VisionModel");
    if (!handle_)
      throw std::runtime_error("SDK returned null model handle");
    int input_count = 0, output_count = 0;
    HBDNN_CHECK(hbDNNGetInputCount(&input_count, handle_),
                "vision input count");
    HBDNN_CHECK(hbDNNGetOutputCount(&output_count, handle_),
                "vision output count");
    if (input_count != 1 || output_count != 1)
      throw std::runtime_error(
          "Vision requires exactly one input and one output");
    hbDNNTensorProperties input{}, output{};
    HBDNN_CHECK(hbDNNGetInputTensorProperties(&input, handle_, 0),
                "get input tensor props");
    HBDNN_CHECK(hbDNNGetOutputTensorProperties(&output, handle_, 0),
                "get output tensor props");
    ValidateVisionTensor(input, true);
    ValidateVisionTensor(output, false);
    input_capacity_ = input.alignedByteSize;
    output_capacity_ = output.alignedByteSize;
    inputs_.reserve(1);
    outputs_.reserve(1);
    inputs_.push_back(AllocateTensor(input));
    outputs_.push_back(AllocateTensor(output));
  } catch (...) {
    Release();
    throw;
  }
}

VisionEngine::~VisionEngine() { Release(); }

void VisionEngine::Release() noexcept {
  FreeTensors(inputs_);
  FreeTensors(outputs_);
  if (packed_)
    hbDNNRelease(packed_);
  packed_ = nullptr;
  handle_ = nullptr;
}

std::vector<float> VisionEngine::Infer(const std::vector<float> &patches) {
  WriteVisionInput(inputs_[0], patches, input_capacity_);
  LogVisionValues("patches", patches);
  LogVisionTensor("input tensor", inputs_[0].properties);
  RunInfer(handle_, inputs_, outputs_);
  auto features = ReadVisionOutput(outputs_[0], output_capacity_);
  LogVisionTensor("output tensor", outputs_[0].properties);
  LogVisionValues("vision output", features);
  return features;
}

// ---- Image IO and vision stage composition (formerly separate fragments) ----

cv::Mat LoadImage(const std::string &path) {
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::runtime_error("failed to read image: " + path);
  return image;
}

std::vector<float> ForwardVision(const std::vector<float> &patches,
                                 const VisionRunner &runner) {
  if (!runner ||
      patches.size() != static_cast<size_t>(kVisionPatches) * kVisionPatchDim)
    throw std::invalid_argument(
        "Vision forward requires a runner and [2520,768] patches");
  for (float value : patches) {
    if (!std::isfinite(value) || value < 0.f || value > 1.f)
      throw std::invalid_argument(
          "Vision RGB patches must be finite values in [0,1]");
  }
  return runner(patches);
}

std::vector<float> PostprocessVision(const std::vector<float> &raw) {
  if (raw.size() != static_cast<size_t>(kVisionSoftTokens) * kHiddenSize)
    throw std::invalid_argument("Vision output requires [280,1536] features");
  for (float value : raw) {
    if (!std::isfinite(value))
      throw std::invalid_argument("Vision output must be finite");
  }
  return raw;
}

std::vector<float> PredictVision(const cv::Mat &bgr,
                                 const VisionRunner &runner) {
  return PostprocessVision(ForwardVision(PreprocessImage(bgr), runner));
}

// ---- Preprocessing stage: in-memory pixels to prepared patches ----

std::vector<float> PreprocessImage(const cv::Mat &bgr) {
  if (bgr.empty() || bgr.dims != 2 || bgr.type() != CV_8UC3) {
    throw std::invalid_argument(
        "Vision input requires a nonempty 2D CV_8UC3 BGR image");
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(kImageWidth, kImageHeight), 0, 0,
             cv::INTER_CUBIC);

  cv::Mat f32;
  resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

  std::vector<float> patches(static_cast<size_t>(kVisionPatches) *
                             static_cast<size_t>(kVisionPatchDim));

  const int hp = kImageHeight / kPatchSize;
  const int wp = kImageWidth / kPatchSize;
  if (hp * wp != kVisionPatches) {
    throw std::runtime_error("unexpected patch grid");
  }

  int patch_idx = 0;
  for (int y = 0; y < hp; ++y) {
    for (int x = 0; x < wp; ++x) {
      float *dst =
          patches.data() + static_cast<size_t>(patch_idx) * kVisionPatchDim;
      int out = 0;
      for (int py = 0; py < kPatchSize; ++py) {
        for (int px = 0; px < kPatchSize; ++px) {
          for (int c = 0; c < 3; ++c) {
            dst[out++] =
                f32.at<cv::Vec3f>(y * kPatchSize + py, x * kPatchSize + px)[c];
          }
        }
      }
      ++patch_idx;
    }
  }

  return patches;
}

// ---- Fixed Vision export descriptor contract and strided physical IO ----

namespace {

// Preserve the source's truncating float-to-half conversion for finite [0,1]
// pixels, rather than silently changing rounding during the refactor.
uint16_t FloatToHalf(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mant = bits & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    mant = (mant | 0x800000) >> (1 - exp);
    return static_cast<uint16_t>(sign | (mant >> 13));
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                               (mant >> 13));
}

float HalfToFloat(uint16_t value) {
  const int exp = (value >> 10) & 31;
  const int mant = value & 1023;
  if (exp == 31)
    throw std::invalid_argument("Vision output contains NaN or infinity");
  const float magnitude =
      exp == 0 ? std::ldexp(static_cast<float>(mant), -24)
               : std::ldexp(1.f + static_cast<float>(mant) / 1024.f, exp - 15);
  return (value & 0x8000) ? -magnitude : magnitude;
}

} // namespace

void ValidateVisionTensor(const hbDNNTensorProperties &p, bool input,
                          int64_t capacity) {
  const int n = p.validShape.numDimensions;
  const int max_dimensions = sizeof(p.validShape.dimensionSize) /
                             sizeof(p.validShape.dimensionSize[0]);
  if (n < 2 || n > max_dimensions || p.alignedByteSize <= 0 || capacity < 0 ||
      (capacity > 0 && p.alignedByteSize > capacity) || p.quantiType != NONE)
    throw std::invalid_argument(
        "Invalid Vision tensor rank, capacity or quantization");
  if ((input && p.tensorType != HB_DNN_TENSOR_TYPE_F16) ||
      (!input && p.tensorType != HB_DNN_TENSOR_TYPE_F16 &&
       p.tensorType != HB_DNN_TENSOR_TYPE_F32))
    throw std::invalid_argument("Vision requires F16 input and F16/F32 output");
  const int rows = input ? kVisionPatches : kVisionSoftTokens;
  const int cols = input ? kVisionPatchDim : kHiddenSize;
  for (int axis = 0; axis < n; ++axis) {
    const int expected = axis == n - 2 ? rows : (axis == n - 1 ? cols : 1);
    if (p.validShape.dimensionSize[axis] != expected)
      throw std::invalid_argument(
          "Vision tensor shape differs from the fixed semantic matrix");
  }
  const int bytes = p.tensorType == HB_DNN_TENSOR_TYPE_F16 ? 2 : 4;
  int64_t span = bytes;
  for (int axis = n - 1; axis >= 0; --axis) {
    const int64_t stride = p.stride[axis];
    const int64_t steps = p.validShape.dimensionSize[axis] - 1;
    if (stride <= 0 || stride % bytes || (steps > 0 && stride < span) ||
        span > p.alignedByteSize ||
        (steps > 0 && stride > (p.alignedByteSize - span) / steps))
      throw std::invalid_argument(
          "Vision tensor byte strides overlap or exceed its allocation");
    span +=
        steps *
        stride; // division guard above prevents overflow before multiplication
  }
}

void WriteVisionInput(hbDNNTensor &tensor, const std::vector<float> &patches,
                      int64_t capacity) {
  const auto &p = tensor.properties;
  ValidateVisionTensor(p, true, capacity);
  if (capacity <= 0 || !tensor.sysMem.virAddr ||
      patches.size() != static_cast<size_t>(kVisionPatches) * kVisionPatchDim)
    throw std::invalid_argument(
        "Vision input buffer or prepared patch count is invalid");
  for (float value : patches)
    if (!std::isfinite(value) || value < 0.f || value > 1.f)
      throw std::invalid_argument(
          "Vision patches must be finite [0,1] RGB values");
  auto *dst = static_cast<unsigned char *>(tensor.sysMem.virAddr);
  std::memset(dst, 0, static_cast<size_t>(p.alignedByteSize));
  const int n = p.validShape.numDimensions;
  for (int row = 0; row < kVisionPatches; ++row)
    for (int col = 0; col < kVisionPatchDim; ++col) {
      const uint16_t value = FloatToHalf(
          patches[static_cast<size_t>(row) * kVisionPatchDim + col]);
      std::memcpy(dst + row * p.stride[n - 2] + col * p.stride[n - 1], &value,
                  2);
    }
}

std::vector<float> ReadVisionOutput(const hbDNNTensor &tensor,
                                    int64_t capacity) {
  const auto &p = tensor.properties;
  ValidateVisionTensor(p, false, capacity);
  if (capacity <= 0 || !tensor.sysMem.virAddr)
    throw std::invalid_argument("Vision output buffer is unavailable");
  const auto *src = static_cast<const unsigned char *>(tensor.sysMem.virAddr);
  const int n = p.validShape.numDimensions;
  std::vector<float> result(static_cast<size_t>(kVisionSoftTokens) *
                            kHiddenSize);
  for (int row = 0; row < kVisionSoftTokens; ++row)
    for (int col = 0; col < kHiddenSize; ++col) {
      const auto *address = src + row * p.stride[n - 2] + col * p.stride[n - 1];
      float value;
      if (p.tensorType == HB_DNN_TENSOR_TYPE_F16) {
        uint16_t half;
        std::memcpy(&half, address, 2);
        value = HalfToFloat(half);
      } else {
        std::memcpy(&value, address, 4);
        if (!std::isfinite(value))
          throw std::invalid_argument("Vision output contains NaN or infinity");
      }
      result[static_cast<size_t>(row) * kHiddenSize + col] = value;
    }
  return result;
}

} // namespace gemma4
