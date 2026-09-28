// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "platform_identity.h"
#include "sdk_runner.hpp"
#include "sha256.h"
#include <filesystem>
#include <stdexcept>
namespace himloco {
void verify_native_model(const std::string &model_path) {
  if (rdk::identify_target(rdk::read_native_identity()) != "x5")
    throw std::runtime_error(
        "HIMLoco native inference requires actual X5 board identity");
  if (std::filesystem::path(model_path).extension() != ".bin")
    throw std::invalid_argument("HIMLoco native model must use .bin");
  // Published identity from docs/release/x5/models.yaml; parity is host-tested.
  constexpr const char *expected =
      "7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a";
  if (rdk::sha256_file(model_path) != expected)
    throw std::runtime_error(
        "HIMLoco published model SHA-256 mismatch or file unreadable; prepare "
        "the exact published BIN explicitly");
}
} // namespace himloco
