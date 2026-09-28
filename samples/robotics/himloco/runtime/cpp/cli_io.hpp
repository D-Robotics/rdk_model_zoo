// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sdk_runner.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
namespace himloco {
inline constexpr const char *kAssetId =
    "x5:himloco:himloco_go2_bayese_1x270.bin";
struct Options {
  NativeConfig model{
      "samples/robotics/himloco/model/bayes-e/himloco_go2_bayese_1x270.bin",
      -1};
  std::filesystem::path input =
      "samples/robotics/himloco/test_data/obs_history";
  std::filesystem::path output = "outputs/himloco_cpp", report;
  int warmup = 10;
  bool help = false;
};
struct InputRecord {
  std::int64_t index;
  std::filesystem::path path;
  std::string digest;
};
struct Inputs {
  std::vector<InputRecord> records;
  nlohmann::json manifest = nullptr;
};
Options parse_cli(int argc, char **argv);
std::string cli_help();
Inputs discover_inputs(const std::filesystem::path &path);
std::pair<std::vector<float>, std::string>
load_input(const InputRecord &record);
void execute(const Options &options);
} // namespace himloco
