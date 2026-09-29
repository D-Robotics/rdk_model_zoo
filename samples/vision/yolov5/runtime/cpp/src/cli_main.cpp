// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"
#include "yolov5_dump.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void print_help(const char* program) {
  std::cout << "Usage: " << program << " --target <x5|s100|s600> --model-path <file>"
            << " --test-img <file> [options]\n"
            << "  --asset-id <published-id>  checked by launcher, recorded in dump\n"
            << "  --label-file <file>       labels for rendering\n"
            << "  --output <file>           rendered output (default result.jpg)\n"
            << "  --dump-dir <dir>          write machine-comparable raw/results evidence\n"
            << "  --score-thres <0..1>      default 0.25\n"
            << "  --nms-thres <0..1>        default 0.45\n"
            << "  --priority <0..255>       default 0 (S only; x5 rejects non-defaults)\n"
            << "  --bpu-core <-1|0..3>      default -1 = any core (S only; x5 rejects "
               "non-defaults)\n"
            << "  --help                    show this message\n";
}

const char* require_value(int argc, char** argv, int* index, const char* option) {
  if (*index + 1 >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
  return argv[++*index];
}

float parse_threshold(const char* text, const char* option) {
  const std::string value = text;
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0F || parsed > 1.0F)
    throw std::invalid_argument(std::string(option) + " must be a finite value in [0,1]");
  return parsed;
}

int parse_int(const char* text, const char* option) {
  const std::string value = text;
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size()) throw std::invalid_argument(std::string(option) + " must be an integer");
  return parsed;
}

}  // namespace

int main(int argc, char** argv) {
  yolov5::RuntimeOptions options;
  options.output_path = "result.jpg";
  for (int i = 0; i < argc; ++i) options.argv.push_back(argv[i]);
  bool help = false;
  int status = 0;
  std::string failure;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") { help = true; continue; }
      if (arg == "--target") options.target = require_value(argc, argv, &i, "--target");
      else if (arg == "--model-path") options.model_path = require_value(argc, argv, &i, "--model-path");
      else if (arg == "--test-img") options.image_path = require_value(argc, argv, &i, "--test-img");
      else if (arg == "--label-file") options.label_path = require_value(argc, argv, &i, "--label-file");
      else if (arg == "--output") options.output_path = require_value(argc, argv, &i, "--output");
      else if (arg == "--dump-dir") options.dump_dir = require_value(argc, argv, &i, "--dump-dir");
      else if (arg == "--asset-id") options.asset_id = require_value(argc, argv, &i, "--asset-id");
      else if (arg == "--score-thres") options.score_threshold = parse_threshold(require_value(argc, argv, &i, "--score-thres"), "--score-thres");
      else if (arg == "--nms-thres") options.nms_threshold = parse_threshold(require_value(argc, argv, &i, "--nms-thres"), "--nms-thres");
      else if (arg == "--priority") options.priority = parse_int(require_value(argc, argv, &i, "--priority"), "--priority");
      else if (arg == "--bpu-core") options.bpu_core = parse_int(require_value(argc, argv, &i, "--bpu-core"), "--bpu-core");
      else throw std::invalid_argument("unknown option: " + arg);
    }
    if (help) { print_help(argv[0]); return 0; }
    if (options.target.empty() || options.model_path.empty() || options.image_path.empty())
      throw std::invalid_argument("--target, --model-path and --test-img are required");
    if (options.priority < 0 || options.priority > 255 || options.bpu_core < -1 ||
        options.bpu_core > 3)
      throw std::invalid_argument(
          "invalid scheduling parameter (--bpu-core must be -1 or a core index 0..3)");
    status = yolov5::run_native(options);
  } catch (const std::exception& error) {
    failure = error.what();
    std::cerr << "yolov5_cpp: " << failure << "\n";
    status = 2;
  }
  if (status != 0 && !options.dump_dir.empty()) {
    // A failed run still has to be traceable on the board, and the failure
    // record keeps the binary identity like a successful one.
    yolov5::DumpRecord record;
    record.dir = options.dump_dir;
    record.utc = yolov5::utc_timestamp();
    record.target = options.target;
    record.asset_id = options.asset_id;
    record.model_path = options.model_path;
    record.image_path = options.image_path;
    record.argv = options.argv;
    record.cwd = std::filesystem::current_path().string();
    record.binary_path =
        yolov5::current_binary_path(options.argv.empty() ? "" : options.argv.front());
    record.return_code = status;
    record.error = failure;
    record.notes = {"Failure record; no tensor payload was produced."};
    std::string ignored;
    yolov5::write_dump(record, &ignored);
  }
  return status;
}
