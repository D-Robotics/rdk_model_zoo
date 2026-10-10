// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "segment.hpp"
#include <opencv2/core.hpp>
#include <string>
#include <vector>
namespace lanenet {
struct CliOptions {
  std::string model_path, image_path, instance_path, binary_path;
  std::string output_directory = "outputs/lanenet_cpp", target = "s100";
  bool help = false;
};
CliOptions parse_options(const std::vector<std::string> &args);
void print_help(const char *program);
void validate_output_paths(const CliOptions &options);
cv::Mat load_image(const std::string &path);
std::string json_quote(const std::string &value);
std::string dtype_descriptor(ScalarType type);
void write_npy(const std::string &path, ScalarType type,
               const std::vector<std::size_t> &shape,
               const std::vector<unsigned char> &bytes);
cv::Mat embedding_image(const LaneResult &result);
cv::Mat binary_image(const LaneResult &result);
// Write every run artifact (raw NPY outputs, embedding/binary NPY + PNG
// images and the JSON report) into options.output_directory.
void save_results(const CliOptions &options, const LaneNet &model,
                  const LaneResult &result);
} // namespace lanenet
