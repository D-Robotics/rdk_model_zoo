// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// LaneNet CLI: option parsing, image loading, NPY/JSON serialization,
// visualization and the run artifact/report writer. No model or SDK work.
#include "cli.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
namespace lanenet {
namespace {
std::string numbers(const std::vector<std::size_t> &values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}
std::string tensor_record(const TensorSpec &spec) {
  return "{\"shape\":" + numbers(spec.shape) + ",\"dtype\":" +
         json_quote(dtype_descriptor(spec.type)) +
         ",\"byte_strides\":" + numbers(spec.strides) +
         ",\"allocation_bytes\":" + std::to_string(spec.capacity) + "}";
}
void save_image(const std::filesystem::path &path, const cv::Mat &image) {
  if (path.has_parent_path())
    std::filesystem::create_directories(path.parent_path());
  if (!cv::imwrite(path.string(), image))
    throw std::runtime_error("Failed to save image: " + path.string());
}
} // namespace
CliOptions parse_options(const std::vector<std::string> &args) {
  CliOptions out;
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
void print_help(const char *program) {
  std::cout << "Usage: " << program
            << " --model-path HBM --test-img IMAGE [--output NEW_DIR]"
               " [--instance-save-path NEW_PNG] [--binary-save-path NEW_PNG]"
               " [--target s100]\n"
               "Source underscore flag aliases are accepted. Outputs are "
               "embeddings and binary labels; no clustering.\n";
}
void validate_output_paths(const CliOptions &options) {
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
cv::Mat load_image(const std::string &path) {
  const auto image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::invalid_argument("Cannot decode input image");
  return image;
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
cv::Mat embedding_image(const LaneResult &result) {
  if (result.embedding.size() != 3 * 256 * 512)
    throw std::invalid_argument("Wrong embedding geometry");
  cv::Mat image(256, 512, CV_8UC3);
  for (int h = 0; h < 256; h++)
    for (int w = 0; w < 512; w++)
      for (int c = 0; c < 3; c++)
        image.at<cv::Vec3b>(h, w)[c] =
            display_component(result.embedding[c * 256 * 512 + h * 512 + w]);
  return image; // preserved channel order; not lane-instance IDs
}
cv::Mat binary_image(const LaneResult &result) {
  if (result.binary.size() != 256 * 512)
    throw std::invalid_argument("Wrong binary geometry");
  cv::Mat image(256, 512, CV_8UC1);
  for (int h = 0; h < 256; h++)
    for (int w = 0; w < 512; w++) {
      const auto value = result.binary[h * 512 + w];
      if (value > 1)
        throw std::invalid_argument("Invalid binary label");
      image.at<unsigned char>(h, w) = value * 255;
    }
  return image;
}
void save_results(const CliOptions &options, const LaneNet &model,
                  const LaneResult &result) {
  namespace fs = std::filesystem;
  const auto roles = bind_roles(model.output_specs());
  const fs::path output = options.output_directory;
  // Decode/display validation completes before creating any result directory.
  const auto instance = embedding_image(result),
             binary = binary_image(result);
  fs::create_directories(output);
  for (std::size_t i = 0; i < result.raw_outputs.size(); ++i)
    write_npy((output / ("raw_output_" + std::to_string(i) + ".npy")).string(),
              result.raw_outputs[i].spec.type,
              result.raw_outputs[i].spec.shape,
              compact_bytes(result.raw_outputs[i]));
  std::vector<unsigned char> embedding_bytes(result.embedding.size() *
                                             sizeof(float));
  std::memcpy(embedding_bytes.data(), result.embedding.data(),
              embedding_bytes.size());
  write_npy((output / "embedding.npy").string(), ScalarType::Float32,
            {3, 256, 512}, embedding_bytes);
  write_npy((output / "binary.npy").string(), ScalarType::UInt8, {256, 512},
            result.binary);
  save_image(output / "instance_pred.png", instance);
  save_image(output / "binary_pred.png", binary);
  if (!options.instance_path.empty())
    save_image(options.instance_path, instance);
  if (!options.binary_path.empty())
    save_image(options.binary_path, binary);
  std::ofstream report(output / "report.json");
  if (!report)
    throw std::runtime_error("Cannot open report");
  report << "{\"schema_version\":\"1.0\",\"target\":\"s100\",\"model_name\":"
         << json_quote(model.model_name())
         << ",\"model_path\":" << json_quote(options.model_path)
         << ",\"input\":" << json_quote(options.image_path)
         << ",\"runtime_version\":\"unknown\",\"artifact_provenance\":"
            "\"caller-provided; see launch-report.json for observed "
            "hashes\",\"input_metadata\":"
         << tensor_record(model.input_spec()) << ",\"raw_outputs\":[";
  for (std::size_t i = 0; i < result.raw_outputs.size(); ++i) {
    if (i)
      report << ',';
    report << "{\"index\":" << i << ",\"file\":"
           << json_quote("raw_output_" + std::to_string(i) + ".npy")
           << ",\"metadata\":" << tensor_record(result.raw_outputs[i].spec)
           << "}";
  }
  report << "],\"embedding_output_index\":" << roles.embedding
         << ",\"binary_output_index\":" << roles.binary
         << ",\"role_binding\":\"unique dtype and shape; SDK output names not "
            "queried\",\"clustering_performed\":false,\"output_grid\":\"256x512;"
            " not original resolution\",\"embedding_display\":\"clip 0..1 and "
            "nearest ties-to-even rounding\",\"scheduling\":\"source UCP "
            "defaults, ANY core\",\"latency\":\"not measured\"}\n";
  report.flush();
  if (!report)
    throw std::runtime_error("Failed to write report");
}
} // namespace lanenet
