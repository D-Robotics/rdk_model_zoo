// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "preflight.h"
#include <string>
namespace paraformer {
struct CliOptions {
  ModelGroup models;
  std::string manifest, vocabulary, output;
  size_t max_utts = 0;
  bool help = false;
};
CliOptions parse_cli(int argc, const char *const *argv);
std::string cli_help();
std::vector<std::string> load_vocabulary(const std::string &path);
} // namespace paraformer
