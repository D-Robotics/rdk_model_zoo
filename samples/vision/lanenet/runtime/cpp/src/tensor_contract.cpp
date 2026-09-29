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
namespace lanenet {
namespace {
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("Tensor size overflow");
  return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::invalid_argument("Tensor extent overflow");
  return a + b;
}
std::size_t offset(std::size_t index, const TensorSpec &spec) {
  std::size_t out = 0;
  for (std::size_t i = spec.shape.size(); i-- > 0;) {
    out += index % spec.shape[i] * spec.strides[i];
    index /= spec.shape[i];
  }
  return out; // layout validation proves each term and the sum fit capacity
}
std::string normalize(std::string value) {
  const auto start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return {};
  value = value.substr(start, value.find_last_not_of(" \t\r\n") - start + 1);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
std::string read_identity(const char *path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in), {});
}
} // namespace
std::size_t element_bytes(ScalarType type) {
  switch (type) {
  case ScalarType::Float32:
  case ScalarType::Int32:
    return 4;
  case ScalarType::Int64:
    return 8;
  case ScalarType::Int16:
    return 2;
  case ScalarType::Int8:
  case ScalarType::UInt8:
    return 1;
  }
  throw std::invalid_argument("Unknown tensor scalar type");
}
std::size_t validate_layout(const TensorSpec &spec) {
  const auto item = element_bytes(spec.type);
  const auto rank = spec.shape.size();
  if (rank == 0 || rank > 8 || spec.strides.size() != rank ||
      spec.capacity == 0)
    throw std::invalid_argument("Invalid tensor rank/strides/capacity");
  std::size_t span = item, count = 1;
  for (std::size_t i = rank; i-- > 0;) {
    const auto stride = spec.strides[i], dim = spec.shape[i];
    if (dim == 0 || stride == 0 || stride % item || stride < span ||
        stride > spec.capacity)
      throw std::invalid_argument("Overlapping or misaligned tensor strides");
    span = add(span, mul(dim - 1, stride));
    count = mul(count, dim);
  }
  if (span > spec.capacity)
    throw std::invalid_argument("Tensor exceeds buffer capacity");
  return count;
}
OutputRoles bind_roles(const std::vector<TensorSpec> &outputs) {
  const auto none = std::numeric_limits<std::size_t>::max();
  OutputRoles roles{none, none};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto &s = outputs[i];
    validate_layout(s);
    if (s.type == ScalarType::Float32 &&
        s.shape == std::vector<std::size_t>{1, 3, 256, 512}) {
      if (roles.embedding != none)
        throw std::invalid_argument("Ambiguous embedding output role");
      roles.embedding = i;
    }
    if (s.type == ScalarType::Int64 &&
        (s.shape == std::vector<std::size_t>{1, 1, 256, 512} ||
         s.shape == std::vector<std::size_t>{1, 256, 512})) {
      if (roles.binary != none)
        throw std::invalid_argument("Ambiguous binary output role");
      roles.binary = i;
    }
  }
  if (roles.embedding == none || roles.binary == none)
    throw std::invalid_argument("Missing embedding/binary output role");
  return roles;
}
std::vector<unsigned char> compact_bytes(const RawTensor &raw) {
  const auto count = validate_layout(raw.spec),
             item = element_bytes(raw.spec.type);
  if (raw.bytes.size() < raw.spec.capacity)
    throw std::invalid_argument(
        "Owned output is smaller than bound allocation");
  std::vector<unsigned char> result(mul(count, item));
  for (std::size_t index = 0; index < count; ++index)
    std::memcpy(result.data() + index * item,
                raw.bytes.data() + offset(index, raw.spec), item);
  return result;
}
void write_input(const std::vector<float> &input, const TensorSpec &spec,
                 void *destination) {
  if (spec.type != ScalarType::Float32 ||
      spec.shape != std::vector<std::size_t>{1, 3, 256, 512})
    throw std::invalid_argument("Expected float32 NCHW image input");
  const auto count = validate_layout(spec);
  if (!destination || input.size() != count ||
      !std::all_of(input.begin(), input.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument("Invalid prepared input values");
  std::memset(destination, 0, spec.capacity);
  auto *base = static_cast<unsigned char *>(destination);
  for (std::size_t i = 0; i < count; ++i)
    std::memcpy(base + offset(i, spec), &input[i], sizeof(float));
}
LaneResult decode_outputs(const std::vector<RawTensor> &raw) {
  std::vector<TensorSpec> specs;
  for (const auto &value : raw)
    specs.push_back(value.spec);
  const auto roles = bind_roles(specs);
  // Validate every owned allocation, retaining auxiliaries without assigning
  // semantics.
  for (const auto &value : raw)
    if (value.bytes.size() < value.spec.capacity)
      throw std::invalid_argument("Truncated auxiliary/output allocation");
  auto embedding = compact_bytes(raw[roles.embedding]),
       binary = compact_bytes(raw[roles.binary]);
  LaneResult result;
  result.embedding.resize(3 * 256 * 512);
  result.binary.resize(256 * 512);
  for (std::size_t i = 0; i < result.embedding.size(); ++i) {
    float value;
    std::memcpy(&value, embedding.data() + i * 4, 4);
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite embedding");
    result.embedding[i] = value;
  }
  for (std::size_t i = 0; i < result.binary.size(); ++i) {
    std::int64_t value;
    std::memcpy(&value, binary.data() + i * 8, 8);
    if (value != 0 && value != 1)
      throw std::invalid_argument("Binary labels must be 0/1");
    result.binary[i] = static_cast<std::uint8_t>(value);
  }
  return result;
}
std::uint8_t display_component(float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("Nonfinite embedding display value");
  const float scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f;
  const auto base = static_cast<unsigned int>(std::floor(scaled));
  const float fraction = scaled - base;
  return static_cast<std::uint8_t>(
      base + (fraction > .5f || (fraction == .5f && (base % 2))));
}
bool is_s100(std::string soc, std::string board) {
  soc = normalize(soc);
  board = normalize(board);
  return soc == "s100" && board != "s100p" && board != "rdk s100p";
}
void require_s100_board() {
  if (!is_s100(read_identity("/sys/class/boardinfo/soc_name"),
               read_identity("/sys/class/boardinfo/board_type")))
    throw std::invalid_argument(
        "LaneNet requires actual S100 identity; no S100P fallback");
}
} // namespace lanenet
