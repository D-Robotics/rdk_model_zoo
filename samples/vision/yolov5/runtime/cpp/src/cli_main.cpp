// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_help(const char* program) {
  std::cout << "Usage: " << program << " --target <x5|s100|s600> --model-path <file>"
            << " --test-img <file> [options]\n"
            << "  --asset-id <published-id>  checked by launcher\n"
            << "  --label-file <file>       labels for rendering\n"
            << "  --output <file>           rendered output (default result.jpg)\n"
            << "  --score-thres <0..1>      default 0.25\n"
            << "  --nms-thres <0..1>        default 0.45\n"
            << "  --priority <0..255>       default 0\n"
            << "  --bpu-core <-1|0..>       default -1 (runtime default)\n"
            << "  --help                    show this message\n";
}

const char* require_value(int argc, char** argv, int* index, const char* option) {
  if (*index + 1 >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
  return argv[++*index];
}

}  // namespace

int main(int argc, char** argv) {
  yolov5::RuntimeOptions options;
  options.output_path = "result.jpg";
  bool help = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") { help = true; continue; }
      if (arg == "--target") options.target = require_value(argc, argv, &i, "--target");
      else if (arg == "--model-path") options.model_path = require_value(argc, argv, &i, "--model-path");
      else if (arg == "--test-img") options.image_path = require_value(argc, argv, &i, "--test-img");
      else if (arg == "--label-file") options.label_path = require_value(argc, argv, &i, "--label-file");
      else if (arg == "--output") options.output_path = require_value(argc, argv, &i, "--output");
      else if (arg == "--asset-id") { (void)require_value(argc, argv, &i, "--asset-id"); }
      else if (arg == "--score-thres") options.score_threshold = std::stof(require_value(argc, argv, &i, "--score-thres"));
      else if (arg == "--nms-thres") options.nms_threshold = std::stof(require_value(argc, argv, &i, "--nms-thres"));
      else if (arg == "--priority") options.priority = std::stoi(require_value(argc, argv, &i, "--priority"));
      else if (arg == "--bpu-core") options.bpu_core = std::stoi(require_value(argc, argv, &i, "--bpu-core"));
      else throw std::invalid_argument("unknown option: " + arg);
    }
    if (help) { print_help(argv[0]); return 0; }
    if (options.target.empty() || options.model_path.empty() || options.image_path.empty())
      throw std::invalid_argument("--target, --model-path and --test-img are required");
    if (options.score_threshold < 0.0F || options.score_threshold > 1.0F ||
        options.nms_threshold < 0.0F || options.nms_threshold > 1.0F ||
        options.priority < 0 || options.priority > 255 || options.bpu_core < -1)
      throw std::invalid_argument("invalid threshold or scheduling parameter");
    return yolov5::run_native(options);
  } catch (const std::exception& error) {
    std::cerr << "yolov5_cpp: " << error.what() << "\n";
    return 2;
  }
}
