// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// CLI ownership: option parsing, image decoding, depth colorization and all
// result/report IO. No model or SDK logic lives here.
#include "cli.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

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
void print_help() {
  std::cout << "Usage: yolo26_depth --model-path MODEL.bin --test-img "
               "INPUT.jpg --output NEW_DIR [--target x5] [--warmup N]\n"
            << "Legacy positional MODEL.bin INPUT.jpg NEW_DIR is also "
               "accepted.\n";
}
cv::Mat load_image(const std::string &path) {
  const auto image = cv::imread(path, cv::IMREAD_COLOR);
  if (image.empty())
    throw std::invalid_argument("Cannot decode input image");
  return image;
}
cv::Mat colorize_depth(const cv::Mat &depth) {
  if (depth.empty() || depth.type() != CV_32FC1 || !cv::checkRange(depth))
    throw std::invalid_argument("Rendering needs finite float32 HxW depth");
  std::vector<float> values;
  values.reserve(depth.total());
  for (int h = 0; h < depth.rows; ++h)
    values.insert(values.end(), depth.ptr<float>(h),
                  depth.ptr<float>(h) + depth.cols);
  const double low = percentile(values, .02), high = percentile(values, .98);
  const double range = std::max(high - low, 1e-6);
  cv::Mat gray(depth.rows, depth.cols, CV_8UC1), color;
  for (int h = 0; h < depth.rows; ++h)
    for (int w = 0; w < depth.cols; ++w) {
      const double normalized =
          std::clamp((depth.ptr<float>(h)[w] - low) / range, 0.0, 1.0);
      gray.ptr<std::uint8_t>(h)[w] =
          255 - static_cast<std::uint8_t>(normalized * 255);
    }
  cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
  return color;
}
void save_results(const RuntimeOptions &options, const cv::Mat &image,
                  const DepthResult &result, const std::string &model_name) {
  const auto color = colorize_depth(result.depth_native);
  cv::Mat overlay;
  cv::addWeighted(image, .45, color, .55, 0, overlay);
  std::vector<float> log_depth, native;
  log_depth.reserve(result.log_depth.total());
  for (int h = 0; h < result.log_depth.rows; ++h)
    log_depth.insert(log_depth.end(), result.log_depth.ptr<float>(h),
                     result.log_depth.ptr<float>(h) + result.log_depth.cols);
  native.reserve(result.depth_native.total());
  for (int h = 0; h < result.depth_native.rows; ++h)
    native.insert(native.end(), result.depth_native.ptr<float>(h),
                  result.depth_native.ptr<float>(h) + result.depth_native.cols);
  const fs::path output = options.output_directory;
  if (!fs::create_directories(output))
    throw std::runtime_error("Could not create new output directory");
  write_npy((output / "log_depth.npy").string(), log_depth, kOutputSize,
            kOutputSize);
  write_npy((output / "depth_native.npy").string(), native, image.rows,
            image.cols);
  write_f32((output / "depth_native.f32").string(), native);
  if (!cv::imwrite((output / "depth.png").string(), color) ||
      !cv::imwrite((output / "overlay.png").string(), overlay))
    throw std::runtime_error("Could not save visualization");
  std::ostringstream report;
  report
      << std::setprecision(17)
      << "{\n  \"schema_version\": \"2.0\",\n  \"target\": \"x5\",\n  "
         "\"model_name\": "
      << json_quote(model_name)
      << ",\n  \"model_path\": " << json_quote(options.model_path)
      << ",\n  \"input_path\": " << json_quote(options.image_path)
      << ",\n  \"profile\": \"nv12\",\n  \"output_semantics\": "
         "\"calibrated_log_depth\","
      << "\n  \"runtime_version\": \"unknown\",\n  \"artifact_provenance\": "
         "\"caller-provided; see launch-report.json when using launcher\","
      << "\n  \"input_size\": 768,\n  \"log_depth_shape\": [192,192],\n  "
         "\"depth_native_shape\": ["
      << image.rows << ',' << image.cols << "],"
      << "\n  \"latency_ms\": " << result.run.latency_ms
      << ",\n  \"warmup\": " << result.run.warmup
      << ",\n  \"latency_scope\": \"one forward including buffer copy, cache "
         "operations, SDK run and raw output copy; not BPU-only\","
      << "\n  \"depth_units\": \"relative; not calibrated metres\"\n}\n";
  std::ofstream file(output / "report.json");
  file << report.str();
  file.flush();
  if (!file)
    throw std::runtime_error("Could not write native report");
  std::cout << report.str();
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
