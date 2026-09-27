// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "model_identity.h"
#include "platform_identity.h"
namespace yoloe {
constexpr const char *kVocabularySha256 =
    "1a6c943dd251993770e7cf6fed23a38b7ac068f4c8fbc7a0db85cbe0fe5221b3";
// Pure observation seam for host tests. Production factory always reads local
// sysfs/device-tree; there is no CLI/environment override for board identity.
void verify_preflight(const SdkModel &model, const std::string &expected_sha256,
                      const std::string &labels,
                      const rdk::NativeIdentity &actual);
// Observed/custom digest verifies bytes, not publisher origin or compiler
// provenance.
SdkPreflight make_preflight(std::string expected_sha256, std::string labels);
} // namespace yoloe
