// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include "lanenet.hpp"
#include "model_runner.hpp"
#include "visualization.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
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
std::string tensor_record(const lanenet::TensorSpec &spec) {
  return "{\"shape\":" + numbers(spec.shape) + ",\"dtype\":" +
         lanenet::json_quote(lanenet::dtype_descriptor(spec.type)) +
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
int main(int argc, char **argv) {
  using namespace lanenet;
  namespace fs = std::filesystem;
  try {
    const auto options =
        parse_options(std::vector<std::string>(argv + 1, argv + argc));
    if (options.help) {
      std::cout
          << "LaneNet S100: --model-path HBM --test-img IMAGE [--output "
             "NEW_DIR] [--instance-save-path NEW_PNG] [--binary-save-path "
             "NEW_PNG]\nSource underscore flag aliases are accepted. Outputs "
             "are embeddings and binary labels; no clustering.\n";
      return 0;
    }
    validate_output_paths(options);
    const auto image = cv::imread(options.image_path, cv::IMREAD_COLOR);
    if (image.empty())
      throw std::invalid_argument("Cannot decode input image");
    ModelRunner runner(options.model_path);
    LaneNetTask task(
        [&runner](const auto &prepared) { return runner.run(prepared); });
    const auto prepared = task.pre_process(image);
    const auto raw = task.forward(prepared);
    const auto result = task.post_process(raw);
    const auto roles = bind_roles(runner.output_specs());
    const fs::path output = options.output_directory;
    // Decode/display validation completes before creating any result directory.
    const auto instance = embedding_image(result),
               binary = binary_image(result);
    fs::create_directories(output);
    for (std::size_t i = 0; i < raw.size(); ++i)
      write_npy(
          (output / ("raw_output_" + std::to_string(i) + ".npy")).string(),
          raw[i].spec.type, raw[i].spec.shape, compact_bytes(raw[i]));
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
           << json_quote(runner.model_name())
           << ",\"model_path\":" << json_quote(options.model_path)
           << ",\"input\":" << json_quote(options.image_path)
           << ",\"runtime_version\":\"unknown\",\"artifact_provenance\":"
              "\"caller-provided; see launch-report.json for observed "
              "hashes\",\"input_metadata\":"
           << tensor_record(runner.input_spec()) << ",\"raw_outputs\":[";
    for (std::size_t i = 0; i < raw.size(); ++i) {
      if (i)
        report << ',';
      report << "{\"index\":" << i << ",\"file\":"
             << json_quote("raw_output_" + std::to_string(i) + ".npy")
             << ",\"metadata\":" << tensor_record(raw[i].spec) << "}";
    }
    report
        << "],\"embedding_output_index\":" << roles.embedding
        << ",\"binary_output_index\":" << roles.binary
        << ",\"role_binding\":\"unique dtype and shape; SDK output names not "
           "queried\",\"clustering_performed\":false,\"output_grid\":\"256x512;"
           " not original resolution\",\"embedding_display\":\"clip 0..1 and "
           "nearest ties-to-even rounding\",\"scheduling\":\"source UCP "
           "defaults, ANY core\",\"latency\":\"not measured\"}\n";
    report.flush();
    if (!report)
      throw std::runtime_error("Failed to write report");
    std::cout << "Saved embedding, binary labels and raw outputs to " << output
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
