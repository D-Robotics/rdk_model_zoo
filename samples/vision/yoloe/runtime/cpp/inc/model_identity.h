// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <string>
namespace yoloe {
struct SdkModel {
  std::string path, target, variant;
};
inline bool supported_native_model(const SdkModel &model) {
  const bool e11 = model.variant == "11s" || model.variant == "11m" ||
                   model.variant == "11l";
  const bool e26 = model.variant == "26n" || model.variant == "26s" ||
                   model.variant == "26m" || model.variant == "26l" ||
                   model.variant == "26x";
  return (model.target == "x5" && e11) ||
         (model.target == "s100" && (model.variant == "11s" || e26)) ||
         (model.target == "s100p" && e26);
}
// Required policy boundary: verify actual board identity, exact selected asset
// or custom float SHA-256 and vocabulary/conversion provenance. Runs before
// any SDK call. This low-level adapter does not supply a publication resolver.
using SdkPreflight = std::function<void(const SdkModel &)>;
} // namespace yoloe
