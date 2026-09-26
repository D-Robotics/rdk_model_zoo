// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace lanenet {
enum class ScalarType { Float32, Int64, Int32, Int16, Int8, UInt8 };
struct TensorSpec {
  ScalarType type;
  std::vector<std::size_t> shape;
  std::vector<std::size_t> strides; // bytes, resolved before allocation
  std::size_t capacity;
};
struct RawTensor {
  TensorSpec spec;
  std::vector<unsigned char> bytes;
};
struct OutputRoles {
  std::size_t embedding;
  std::size_t binary;
};
struct LaneResult {
  std::vector<float> embedding;     // owned CHW 3x256x512, no clipping
  std::vector<std::uint8_t> binary; // owned HxW labels 0/1, no scaling
};
std::size_t element_bytes(ScalarType type);
std::size_t validate_layout(const TensorSpec &spec);
OutputRoles bind_roles(const std::vector<TensorSpec> &outputs);
std::vector<unsigned char> compact_bytes(const RawTensor &raw);
void write_input(const std::vector<float> &input, const TensorSpec &spec,
                 void *destination);
LaneResult decode_outputs(const std::vector<RawTensor> &raw);
std::uint8_t display_component(float value);
bool is_s100(std::string soc, std::string board);
void require_s100_board();
} // namespace lanenet
