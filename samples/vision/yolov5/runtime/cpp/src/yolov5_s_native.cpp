// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_s_native.hpp"

#include <cstring>
#include <stdexcept>

namespace yolov5 {

std::vector<float> dequant_s32_nhwc(const unsigned char* base, const TensorMeta& meta,
                                    const float* scale_data, long long scale_len,
                                    const std::int32_t* zero_point_data,
                                    long long zero_point_len) {
  if (meta.quanti_type != kQuantiScale && meta.quanti_type != kQuantiNone)
    throw std::invalid_argument("dequant_s32_nhwc: unsupported quantization kind");
  if (meta.num_dimensions != 4 || meta.valid[0] != 1 || base == nullptr)
    throw std::invalid_argument("dequant_s32_nhwc: expected a batch-1 rank-4 buffer");
  if (meta.quanti_type == kQuantiScale &&
      (scale_data == nullptr || (scale_len != 1 && scale_len < meta.valid[3])))
    throw std::invalid_argument("dequant_s32_nhwc: unusable scale descriptor");
  if (meta.quanti_type == kQuantiScale && zero_point_len != 0 && zero_point_len != 1 &&
      zero_point_len < meta.valid[3])
    throw std::invalid_argument("dequant_s32_nhwc: unusable zero-point descriptor");
  // A declared zero point without a buffer must never reach the read below.
  if (meta.quanti_type == kQuantiScale && zero_point_len > 0 &&
      zero_point_data == nullptr)
    throw std::invalid_argument("dequant_s32_nhwc: zero-point length without data");

  const long long height = meta.valid[1];
  const long long width = meta.valid[2];
  const long long channels = meta.valid[3];
  std::vector<float> result(static_cast<std::size_t>(height * width * channels));
  const bool scalar_scale = scale_len == 1;
  const bool scalar_zero = zero_point_len == 1;
  std::size_t index = 0;
  for (long long h = 0; h < height; ++h) {
    for (long long w = 0; w < width; ++w) {
      const std::size_t pixel = static_cast<std::size_t>((h * width + w) * meta.stride[2]);
      for (long long c = 0; c < channels; ++c) {
        const std::size_t offset = pixel + static_cast<std::size_t>(c * meta.stride[3]);
        if (meta.quanti_type == kQuantiScale) {
          const float scale = scalar_scale ? scale_data[0] : scale_data[c];
          const std::int32_t zero =
              zero_point_len == 0 ? 0 : (scalar_zero ? zero_point_data[0] : zero_point_data[c]);
          std::int32_t quantized = 0;
          std::memcpy(&quantized, base + offset, sizeof(quantized));
          result[index] = (static_cast<float>(quantized) - static_cast<float>(zero)) * scale;
        } else {
          float value = 0.0F;
          std::memcpy(&value, base + offset, sizeof(value));
          result[index] = value;
        }
        ++index;
      }
    }
  }
  return result;
}

bool bpu_core_to_backend(long long bpu_core, unsigned long long* backend) {
  if (backend == nullptr) return false;
  if (bpu_core == -1) {
    *backend = 1ULL << 7;  // HB_UCP_BPU_CORE_ANY
    return true;
  }
  if (bpu_core < 0 || bpu_core > 3) return false;
  *backend = 1ULL << static_cast<unsigned>(bpu_core);  // HB_UCP_BPU_CORE_0..3
  return true;
}

}  // namespace yolov5
