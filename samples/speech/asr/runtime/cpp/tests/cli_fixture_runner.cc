// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Linked only into asr_cli_fixture; no vendor SDK or real board identity.
#include "platform_identity.h"
#include "sdk_runner.h"
#include <cstdlib>
#include <stdexcept>
namespace rdk {
NativeIdentity read_native_identity() {
  const char *target = std::getenv("ASR_FIXTURE_TARGET");
  return {target ? target : "s100", "", "", ""};
}
} // namespace rdk
namespace asr {
struct SdkRunner::Impl {
  SdkMetadata metadata{"fixture",         4,      {120000, 4},
                       {56048, 14012, 4}, 120000, 56048};
  size_t calls = 0;
};
SdkRunner::SdkRunner(SdkModel model, SdkPreflight gate)
    : impl_(std::make_unique<Impl>()) {
  if (!gate)
    throw std::invalid_argument("Missing fixture preflight");
  gate(model);
}
SdkRunner::~SdkRunner() = default;
const SdkMetadata &SdkRunner::metadata() const { return impl_->metadata; }
std::vector<float> SdkRunner::infer(const std::vector<float> &prepared) {
  if (prepared.size() != 30000)
    throw std::invalid_argument("Fixture input length");
  const char *fail = std::getenv("ASR_FIXTURE_FAIL_AFTER");
  if (fail && impl_->calls >= static_cast<size_t>(std::stoul(fail)))
    throw std::runtime_error("Injected fixture inference failure");
  ++impl_->calls;
  std::vector<float> out(4 * 3503, 0.f);
  out[5] = out[3503 + 5] = out[3 * 3503 + 5] = 1.f;
  return out;
}
} // namespace asr
