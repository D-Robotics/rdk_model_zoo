// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Explicit host-only replacements for the SDK adapter and local identity
// reader. Never linked into yoloe_demo; there is no runtime switch enabling
// this fixture.
#include "detect.hpp"
#include "platform_identity.h"
#include <stdexcept>
namespace rdk {
NativeIdentity read_native_identity() { return {"s100", "RDK S100P", "", ""}; }
} // namespace rdk
namespace yoloe {
struct SdkRunner::Impl {};
SdkRunner::SdkRunner(SdkModel model, SdkPreflight gate)
    : impl_(std::make_unique<Impl>()) {
  if (model.target != "s100p" || model.variant != "26n" || !gate)
    throw std::invalid_argument("Fixture expects S100P E26n");
  gate(model);
}
SdkRunner::~SdkRunner() = default;
Protocol SdkRunner::protocol() const { return Protocol::E26; }
Heads SdkRunner::infer(const Nv12Input &input) {
  if (input.y.size() != 640 * 640 || input.uv.size() != 640 * 320)
    throw std::invalid_argument("Fixture NV12 length");
  Heads result;
  for (int i = 0; i < 3; ++i) {
    int grid = 80 >> i;
    result[i * 3].assign(grid * grid * 4585, -20.f);
    result[i * 3 + 1].assign(grid * grid * 4, 2.f);
    result[i * 3 + 2].assign(grid * grid * 32, 0.f);
  }
  const int anchor = 40 * 80 + 40;
  result[0][anchor * 4585 + 2] = 2.f;
  result[2][anchor * 32] = 1.f;
  result[9].assign(160 * 160 * 32, 1.f);
  return result;
}
} // namespace yoloe
