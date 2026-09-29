// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "preflight.h"
#include "sha256.h"
#include <fstream>
#include <stdexcept>
#include <utility>
namespace asr {
namespace {
std::string digest(std::string value) {
  if (value.size() != 64 ||
      !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
      }))
    throw std::invalid_argument("Expected a 64-digit model SHA-256");
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
} // namespace
void verify_preflight(const SdkModel &model, const std::string &expected_sha256,
                      const std::string &vocabulary,
                      const rdk::NativeIdentity &actual) {
  if (model.target != "s100" && model.target != "s600")
    throw std::invalid_argument("ASR native target must be s100 or s600");
  const auto expected = digest(expected_sha256);
  if (rdk::identify_target(actual) != model.target)
    throw std::invalid_argument(
        "Local board identity is unknown or does not match target " +
        model.target);
  std::ifstream file(model.path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing or empty ASR model");
  if (rdk::sha256_file(model.path) != expected)
    throw std::invalid_argument("Model SHA-256 mismatch");
  if (rdk::sha256_file(vocabulary) != kVocabularySha256)
    throw std::invalid_argument("Vocabulary SHA-256 mismatch: expected fixed "
                                "3503-token ASR vocabulary");
}
SdkPreflight make_preflight(std::string expected_sha256,
                            std::string vocabulary) {
  expected_sha256 = digest(std::move(expected_sha256));
  return [expected_sha256, vocabulary](const SdkModel &model) {
    verify_preflight(model, expected_sha256, vocabulary,
                     rdk::read_native_identity());
  };
}
} // namespace asr
