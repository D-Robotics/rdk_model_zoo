// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "platform_identity.h"
#include "sdk_runner.h"
#include <array>
namespace paraformer {
struct ModelArtifact {
  SdkModel model;
  std::string asset_id, expected_sha256;
};
using ModelGroup = std::array<ModelArtifact, 3>;
inline constexpr const char *kVocabularySha256 =
    "2b20c2b12572d682afff84ce1c8d560f67b8b32a4c1f21567411d141ed352127";
std::string expected_asset_id(Stage stage);
// Pure explicit-identity verifier for tests and callers with an actual identity
// snapshot. Production callers should use make_preflight to read local
// identity.
void verify_group(const ModelGroup &, const std::string &vocabulary,
                  const rdk::NativeIdentity &actual);
// Immediately verifies all three artifacts and vocabulary, before constructing
// any runner. Returned callback rechecks the group and exact selected model.
SdkPreflight make_preflight(ModelGroup, std::string vocabulary);
} // namespace paraformer
