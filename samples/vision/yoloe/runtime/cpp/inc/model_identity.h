// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <string>
namespace yoloe {
struct SdkModel {
  std::string path, target, variant;
};
// Required policy boundary: verify actual board identity, exact selected asset
// or custom float SHA-256 and vocabulary/conversion provenance. Runs before
// any SDK call. This low-level adapter does not supply a publication resolver.
using SdkPreflight = std::function<void(const SdkModel &)>;
} // namespace yoloe
