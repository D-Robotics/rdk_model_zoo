// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include "image_io.hpp"
#include "model_runner.hpp"
#include "yolo26_depth.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <stdexcept>
namespace fs = std::filesystem;
using namespace yolo26_depth;
int main(int argc, char **argv) {
  try {
    const auto options =
        parse_options(std::vector<std::string>(argv + 1, argv + argc));
    if (options.help) {
      std::cout << "Usage: yolo26_depth --model-path MODEL.bin --test-img "
                   "INPUT.jpg --output NEW_DIR [--target x5] [--warmup N]\n"
                << "Legacy positional MODEL.bin INPUT.jpg NEW_DIR is also "
                   "accepted.\n";
      return 0;
    }
    if (fs::exists(options.output_directory))
      throw std::invalid_argument("Output directory must be new");
    const auto image = cv::imread(options.image_path, cv::IMREAD_COLOR);
    if (image.empty())
      throw std::invalid_argument("Cannot decode input image");
    ModelRunner runner(options.model_path);
    Yolo26DepthTask task(
        [&](const auto &tensors) { return runner.run(tensors); });
    const auto prepared = task.pre_process(image);
    for (int i = 0; i < options.warmup; ++i)
      task.forward(prepared.nv12);
    const auto start = std::chrono::steady_clock::now();
    const auto raw = task.forward(prepared.nv12);
    const auto end = std::chrono::steady_clock::now();
    const double latency =
        std::chrono::duration<double, std::milli>(end - start).count();
    const auto result = task.post_process(raw, prepared.context);
    const auto color = colorize_depth(result.depth_native);
    cv::Mat overlay;
    cv::addWeighted(image, .45, color, .55, 0, overlay);
    std::vector<float> native;
    native.reserve(result.depth_native.total());
    for (int h = 0; h < result.depth_native.rows; ++h)
      native.insert(native.end(), result.depth_native.ptr<float>(h),
                    result.depth_native.ptr<float>(h) +
                        result.depth_native.cols);
    const fs::path output = options.output_directory;
    if (!fs::create_directories(output))
      throw std::runtime_error("Could not create new output directory");
    write_npy((output / "log_depth.npy").string(), raw, kOutputSize,
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
        << json_quote(runner.model_name())
        << ",\n  \"model_path\": " << json_quote(options.model_path)
        << ",\n  \"input_path\": " << json_quote(options.image_path)
        << ",\n  \"profile\": \"nv12\",\n  \"output_semantics\": "
           "\"calibrated_log_depth\","
        << "\n  \"runtime_version\": \"unknown\",\n  \"artifact_provenance\": "
           "\"caller-provided; see launch-report.json when using launcher\","
        << "\n  \"input_size\": 768,\n  \"log_depth_shape\": [192,192],\n  "
           "\"depth_native_shape\": ["
        << image.rows << ',' << image.cols << "],"
        << "\n  \"latency_ms\": " << latency
        << ",\n  \"warmup\": " << options.warmup
        << ",\n  \"latency_scope\": \"one forward including buffer copy, cache "
           "operations, SDK run and raw output copy; not BPU-only\","
        << "\n  \"depth_units\": \"relative; not calibrated metres\"\n}\n";
    std::ofstream file(output / "report.json");
    file << report.str();
    file.flush();
    if (!file)
      throw std::runtime_error("Could not write native report");
    std::cout << report.str();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
