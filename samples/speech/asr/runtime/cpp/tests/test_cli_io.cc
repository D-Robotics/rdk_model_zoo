// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include <cassert>
#include <stdexcept>
template <class F> void rejects(F f) {
  bool caught = false;
  try {
    f();
  } catch (const std::exception &) {
    caught = true;
  }
  assert(caught);
}
int main(int argc, char **argv) {
  assert(argc == 2);
  const auto tokens = asr::load_vocabulary(argv[1]);
  assert(tokens.size() == 3503 && tokens[0] == "<pad>" && tokens[5] == "A");
  const char *help[] = {"asr", "--help"};
  assert(asr::parse_cli(2, help).help);
  const char *args[] = {
      "asr",
      "--target",
      "s100",
      "--asset-id",
      "s:asr:s100/asr.hbm",
      "--model-path",
      "model.hbm",
      "--model-sha256",
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"};
  const auto parsed = asr::parse_cli(9, args);
  assert(parsed.model.target == "s100" && parsed.decode_mode == "legacy");
  const char *duplicate[] = {"asr", "--target", "s100", "--target", "s600"};
  rejects([&] { asr::parse_cli(5, duplicate); });
  const char *unknown[] = {"asr", "--unknown"};
  rejects([&] { asr::parse_cli(2, unknown); });
  const char *missing[] = {"asr", "--target"};
  rejects([&] { asr::parse_cli(2, missing); });
  rejects([&] { asr::load_vocabulary("/nonexistent/vocab.json"); });
}
