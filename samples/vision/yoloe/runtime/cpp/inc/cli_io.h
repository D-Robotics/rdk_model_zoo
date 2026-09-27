// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "config.h"
#include "model_identity.h"
#include "pipeline_io.h"
namespace yoloe {
struct CliOptions {
  SdkModel model;
  Config config;
  std::string model_sha256, image_path, label_path, output;
  bool help = false, contours = true;
  std::vector<std::string> argv;
};
struct CliInputs {
  cv::Mat image;
  std::vector<std::string> labels;
  std::string image_sha256, vocabulary_sha256;
};
CliOptions parse_cli(int argc, char **argv);
std::string cli_help();
CliInputs load_cli_inputs(const CliOptions &options);
void create_output_directory(const std::string &path);
void save_cli_outputs(const CliOptions &options, const CliInputs &inputs,
                      const Result &result);
} // namespace yoloe
