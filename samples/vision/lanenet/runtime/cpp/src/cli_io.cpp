// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
namespace lanenet {
RuntimeOptions parse_options(const std::vector<std::string> &args) {
  RuntimeOptions out;
  std::vector<std::string> values;
  for (const auto &arg : args) {
    if (arg == "--help") {
      out.help = true;
      return out;
    }
    const auto equal = arg.find('=');
    if (arg.rfind("--", 0) == 0 && equal != std::string::npos) {
      values.push_back(arg.substr(0, equal));
      values.push_back(arg.substr(equal + 1));
    } else
      values.push_back(arg);
  }
  for (std::size_t i = 0; i < values.size(); i += 2) {
    if (i + 1 >= values.size())
      throw std::invalid_argument("Missing option value");
    auto key = values[i];
    std::replace(key.begin(), key.end(), '_', '-');
    const auto &value = values[i + 1];
    if (key == "--model-path")
      out.model_path = value;
    else if (key == "--test-img")
      out.image_path = value;
    else if (key == "--output")
      out.output_directory = value;
    else if (key == "--target")
      out.target = value;
    else if (key == "--instance-save-path")
      out.instance_path = value;
    else if (key == "--binary-save-path")
      out.binary_path = value;
    else
      throw std::invalid_argument("Unknown option: " + key);
  }
  if (out.target != "s100")
    throw std::invalid_argument("Native LaneNet supports S100 only");
  if (out.model_path.empty() || out.image_path.empty() ||
      out.output_directory.empty())
    throw std::invalid_argument("Explicit model-path and test-img are "
                                "required; output must be nonempty");
  return out;
}
void validate_output_paths(const RuntimeOptions &options) {
  namespace fs = std::filesystem;
  const auto output = fs::weakly_canonical(options.output_directory);
  if (fs::exists(output))
    throw std::invalid_argument("Output directory must be new");
  std::set<fs::path> extras;
  const std::set<std::string> reserved = {
      "embedding.npy",     "binary.npy",       "instance_pred.png",
      "binary_pred.png",   "report.json",      "launch-report.json",
      "native.stdout.log", "native.stderr.log"};
  for (const auto &path : {options.instance_path, options.binary_path}) {
    if (path.empty())
      continue;
    const auto p = fs::weakly_canonical(path);
    const auto name = p.filename().string();
    if (fs::exists(p) || p == output || !extras.insert(p).second ||
        (p.parent_path() == output &&
         (reserved.count(name) || name.rfind("raw_output_", 0) == 0)))
      throw std::invalid_argument(
          "Additional image path exists or conflicts with output records");
  }
}
std::string json_quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<int>(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
std::string dtype_descriptor(ScalarType type) {
  switch (type) {
  case ScalarType::Float32:
    return "<f4";
  case ScalarType::Int64:
    return "<i8";
  case ScalarType::Int32:
    return "<i4";
  case ScalarType::Int16:
    return "<i2";
  case ScalarType::Int8:
    return "|i1";
  case ScalarType::UInt8:
    return "|u1";
  }
  throw std::invalid_argument("Unknown scalar type");
}
void write_npy(const std::string &path, ScalarType type,
               const std::vector<std::size_t> &shape,
               const std::vector<unsigned char> &bytes) {
  const std::uint16_t one = 1;
  if (*reinterpret_cast<const unsigned char *>(&one) != 1)
    throw std::runtime_error("NPY payload requires little-endian host");
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "IEEE float32 required");
  if (shape.empty())
    throw std::invalid_argument("Empty NPY shape");
  std::size_t expected = element_bytes(type);
  std::ostringstream dims;
  for (auto n : shape) {
    if (n == 0 || expected > std::numeric_limits<std::size_t>::max() / n)
      throw std::invalid_argument("Invalid NPY shape");
    expected *= n;
    dims << n << ", ";
  }
  if (expected != bytes.size())
    throw std::invalid_argument("NPY shape/type differs from payload");
  std::string header = "{'descr': '" + dtype_descriptor(type) +
                       "', 'fortran_order': False, 'shape': (" + dims.str() +
                       "), }";
  header.append((64 - (10 + header.size() + 1) % 64) % 64, ' ');
  header += '\n';
  if (header.size() > 65535)
    throw std::invalid_argument("NPY header too large");
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot open NPY output: " + path);
  const unsigned char magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(reinterpret_cast<const char *>(magic), 8);
  const unsigned char length[] = {
      static_cast<unsigned char>(header.size() & 255),
      static_cast<unsigned char>(header.size() >> 8)};
  out.write(reinterpret_cast<const char *>(length), 2);
  out << header;
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  out.flush();
  if (!out)
    throw std::runtime_error("Failed to write NPY output");
}
} // namespace lanenet
