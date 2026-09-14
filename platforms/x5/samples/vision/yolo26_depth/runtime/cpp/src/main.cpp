/**
 * @file main.cpp
 * @brief Run YOLO26 Depth inference from the command line on RDK X5.
 */

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "yolo26_depth.hpp"

namespace fs = std::filesystem;

/**
 * @brief Parse command-line arguments and execute one depth inference.
 * @param argc Number of command-line arguments.
 * @param argv Command-line argument values.
 * @return Zero on success, otherwise a non-zero error code.
 */
int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " MODEL.bin INPUT.jpg OUTPUT_DIR\n";
    return 2;
  }

  try {
    const fs::path model_path = argv[1];
    const fs::path image_path = argv[2];
    const fs::path output_dir = argv[3];
    fs::create_directories(output_dir);

    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("failed to decode image: " + image_path.string());
    }

    yolo26_depth::Yolo26Depth estimator(model_path.string());
    const yolo26_depth::InferenceResult result = estimator.Infer(image);
    const cv::Mat depth_color = yolo26_depth::ColorizeDepth(result.depth_native);
    cv::Mat overlay;
    cv::addWeighted(image, 0.45, depth_color, 0.55, 0.0, overlay);

    std::ofstream depth_file(output_dir / "depth_native.f32", std::ios::binary);
    if (!depth_file) {
      throw std::runtime_error("failed to open depth_native.f32");
    }
    depth_file.write(reinterpret_cast<const char*>(result.depth_native.ptr<float>()),
                     result.depth_native.total() * sizeof(float));
    if (!depth_file) {
      throw std::runtime_error("failed to write depth_native.f32");
    }
    if (!cv::imwrite((output_dir / "depth.png").string(), depth_color)) {
      throw std::runtime_error("failed to write depth.png");
    }
    if (!cv::imwrite((output_dir / "overlay.png").string(), overlay)) {
      throw std::runtime_error("failed to write overlay.png");
    }

    std::ofstream report(output_dir / "report.json");
    if (!report) {
      throw std::runtime_error("failed to open report.json");
    }
    report << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"schema_version\": \"1.0\",\n"
           << "  \"model_name\": \"" << estimator.model_name() << "\",\n"
           << "  \"input_size\": " << estimator.input_size() << ",\n"
           << "  \"source_height\": " << image.rows << ",\n"
           << "  \"source_width\": " << image.cols << ",\n"
           << "  \"output_height\": " << result.depth_native.rows << ",\n"
           << "  \"output_width\": " << result.depth_native.cols << ",\n"
           << "  \"latency_ms\": " << result.latency_ms << "\n"
           << "}\n";
    if (!report) {
      throw std::runtime_error("failed to write report.json");
    }
    std::cout << "model=" << estimator.model_name()
              << " input=" << estimator.input_size() << "x" << estimator.input_size()
              << " latency_ms=" << result.latency_ms << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "[ERROR] " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
