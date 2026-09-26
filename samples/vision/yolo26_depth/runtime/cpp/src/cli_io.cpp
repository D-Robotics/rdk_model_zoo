// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace yolo26_depth {
RuntimeOptions parse_options(const std::vector<std::string> &args) {
  RuntimeOptions options;
  if (std::find(args.begin(), args.end(), "--help") != args.end()) {
    options.help = true;
    return options;
  }
  if (args.size() == 3 && args[0].rfind("--", 0) != 0) {
    options.model_path = args[0];
    options.image_path = args[1];
    options.output_directory = args[2];
    return options;
  }
  for (std::size_t i = 0; i < args.size(); i += 2) {
    if (i + 1 >= args.size())
      throw std::invalid_argument("Missing option value: " + args[i]);
    const auto &key = args[i];
    const auto &value = args[i + 1];
    if (key == "--model-path" || key == "--model")
      options.model_path = value;
    else if (key == "--test-img" || key == "--input")
      options.image_path = value;
    else if (key == "--output")
      options.output_directory = value;
    else if (key == "--target")
      options.target = value;
    else if (key == "--warmup") {
      std::size_t parsed = 0;
      options.warmup = std::stoi(value, &parsed);
      if (parsed != value.size() || options.warmup < 0)
        throw std::invalid_argument("warmup must be a nonnegative integer");
    } else
      throw std::invalid_argument("Unknown option: " + key);
  }
  if (options.target != "x5")
    throw std::invalid_argument("Native implementation supports x5 only");
  if (options.model_path.empty() || options.image_path.empty() ||
      options.output_directory.empty())
    throw std::invalid_argument("model-path, test-img and output are required");
  return options;
}
std::string json_quote(const std::string &value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      output << '\\' << c;
    else if (c < 32)
      output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<int>(c) << std::dec;
    else
      output << c;
  }
  output << '"';
  return output.str();
}
namespace {
void require_little_endian() {
  const std::uint16_t word = 1;
  if (*reinterpret_cast<const unsigned char *>(&word) != 1 ||
      sizeof(float) != 4)
    throw std::runtime_error(
        "Float serialization requires little-endian IEEE float32 host");
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE float32 required");
}
void write_payload(std::ofstream &stream, const std::vector<float> &values) {
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  stream.flush();
  if (!stream)
    throw std::runtime_error("Failed to write float payload");
}
} // namespace
void write_npy(const std::string &path, const std::vector<float> &values,
               int height, int width) {
  require_little_endian();
  if (height <= 0 || width <= 0 ||
      static_cast<std::size_t>(height) >
          std::numeric_limits<std::size_t>::max() /
              static_cast<std::size_t>(width) ||
      values.size() != static_cast<std::size_t>(height) * width)
    throw std::invalid_argument("NPY dimensions do not match values");
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(height) + ", " + std::to_string(width) +
                       "), }";
  header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
  header += '\n';
  if (header.size() > 65535)
    throw std::invalid_argument("NPY header too large");
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot open NPY: " + path);
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  stream.write(reinterpret_cast<const char *>(magic), 8);
  const unsigned char length[] = {
      static_cast<unsigned char>(header.size() & 255),
      static_cast<unsigned char>(header.size() >> 8)};
  stream.write(reinterpret_cast<const char *>(length), 2);
  stream << header;
  write_payload(stream, values);
}
void write_f32(const std::string &path, const std::vector<float> &values) {
  require_little_endian();
  std::ofstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("Cannot open raw output: " + path);
  write_payload(stream, values);
}
} // namespace yolo26_depth
