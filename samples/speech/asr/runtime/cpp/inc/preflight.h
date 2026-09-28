// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "platform_identity.h"
#include "sdk_runner.h"
namespace asr {
inline constexpr const char *kVocabularySha256 =
    "33fea3444869c2cd2433f59da079b04ce91515f946d21fe1b0ff3825398bcec7";
void verify_preflight(const SdkModel &, const std::string &expected_sha256,
                      const std::string &vocabulary,
                      const rdk::NativeIdentity &actual);
SdkPreflight make_preflight(std::string expected_sha256,
                            std::string vocabulary);
} // namespace asr
