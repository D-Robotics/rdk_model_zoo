// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "preflight.h"
#include "sha256.h"
#include <filesystem>
#include <set>
#include <stdexcept>
#include <utility>
namespace paraformer {
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
std::filesystem::path model_file(const std::string &path) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) == 0)
    throw std::invalid_argument(
        "Missing, empty or non-regular Paraformer model");
  return std::filesystem::canonical(path);
}
} // namespace
std::string expected_asset_id(Stage stage) {
  const std::string prefix = "s:paraformer:s100/paraformer_large_";
  switch (stage) {
  case Stage::Encoder:
    return prefix + "encoder_400x560_s100.hbm";
  case Stage::Predictor:
    return prefix + "predictor_400x512_s100.hbm";
  case Stage::Decoder:
    return prefix + "decoder_400x512_s100.hbm";
  }
  throw std::invalid_argument("Unknown Paraformer stage");
}
void verify_group(const ModelGroup &group, const std::string &vocabulary,
                  const rdk::NativeIdentity &actual) {
  if (rdk::identify_target(actual) != "s100")
    throw std::invalid_argument(
        "Paraformer requires actual local S100 identity");
  std::set<Stage> stages;
  std::set<std::filesystem::path> paths;
  // Validate the complete selection before hashing any model.
  for (const auto &artifact : group) {
    const auto &model = artifact.model;
    if (model.target != "s100" ||
        artifact.asset_id != expected_asset_id(model.stage) ||
        !stages.insert(model.stage).second)
      throw std::invalid_argument(
          "Expected one matching S100 publication for each Paraformer stage");
    (void)digest(artifact.expected_sha256);
    if (!paths.insert(model_file(model.path)).second)
      throw std::invalid_argument(
          "Each Paraformer stage requires a distinct model file");
  }
  // Detect hard-linked aliases too; canonical paths alone do not identify them.
  for (size_t i = 0; i < group.size(); ++i)
    for (size_t j = 0; j < i; ++j)
      if (std::filesystem::equivalent(group[i].model.path, group[j].model.path))
        throw std::invalid_argument(
            "Paraformer stages alias the same model file");
  for (const auto &artifact : group)
    if (rdk::sha256_file(artifact.model.path) !=
        digest(artifact.expected_sha256))
      throw std::invalid_argument("Model SHA-256 mismatch: " +
                                  artifact.asset_id);
  if (rdk::sha256_file(vocabulary) != kVocabularySha256)
    throw std::invalid_argument(
        "Expected fixed published 8404-token vocabulary SHA-256");
}
SdkPreflight make_preflight(ModelGroup group, std::string vocabulary) {
  verify_group(group, vocabulary, rdk::read_native_identity());
  return [group = std::move(group),
          vocabulary = std::move(vocabulary)](const SdkModel &model) {
    const auto selected =
        std::find_if(group.begin(), group.end(), [&](const auto &a) {
          return a.model.stage == model.stage;
        });
    if (selected == group.end() || model.target != selected->model.target ||
        model_file(model.path) != model_file(selected->model.path))
      throw std::invalid_argument(
          "Runner model differs from the verified Paraformer group");
    verify_group(group, vocabulary, rdk::read_native_identity());
  };
}
} // namespace paraformer
