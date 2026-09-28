// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "sdk_runner.h"
#include <string>
#include <vector>
namespace asr {
struct CliOptions {
  SdkModel model;
  std::string asset_id, model_sha256;
  std::string audio = "samples/speech/asr/test_data/chi_sound.wav";
  std::string vocabulary = "samples/speech/asr/test_data/vocab.json";
  std::string output = "outputs/asr_cpp/result", decode_mode = "ctc";
  bool help = false;
};
CliOptions parse_cli(int argc, const char *const *argv);
std::string cli_help();
std::vector<std::string> load_vocabulary(const std::string &path);
} // namespace asr
