// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
#include "gemma4_vision_tensor.hpp"
#include "gemma4_config.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace gemma4 {
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
