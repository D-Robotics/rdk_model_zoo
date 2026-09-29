// Explicit host transport/identity double; never linked into paraformer_demo.
#include "platform_identity.h"
#include "sdk_runner.h"
#include <fstream>
#include <stdexcept>
namespace rdk {
NativeIdentity read_native_identity() { return {"s100", "", "", ""}; }
} // namespace rdk
namespace paraformer {
struct SdkRunner::Impl {
  SdkModel model;
  SdkMetadata metadata;
  std::string control;
};
SdkRunner::SdkRunner(SdkModel model, SdkPreflight gate)
    : impl_(std::make_unique<Impl>()) {
  gate(model);
  impl_->model = model;
  std::ifstream file(model.path);
  file >> impl_->control;
  if (impl_->control == "fail-load")
    throw std::runtime_error("Injected model load failure");
  impl_->metadata.model_name = "HOST_FIXTURE_NOT_A_MODEL";
}
SdkRunner::~SdkRunner() = default;
const SdkMetadata &SdkRunner::metadata() const { return impl_->metadata; }
RawTensors SdkRunner::infer(const RawTensors &inputs) {
  if (impl_->control == "fail-infer")
    throw std::runtime_error("Injected raw runner failure");
  if (impl_->model.stage == Stage::Encoder) {
    if (std::get<std::vector<float>>(inputs.at("features")).size() != 400 * 560)
      throw std::runtime_error("Fixture features");
    return {{"context", std::vector<float>(400 * 512, 0.f)}};
  }
  if (impl_->model.stage == Stage::Predictor) {
    std::vector<float> weights(401, 0.f);
    if (impl_->control != "zero")
      weights[0] = weights[1] = 1.f;
    return {{"alphas", weights},
            {"hidden", std::vector<float>(401 * 512, 0.f)}};
  }
  const auto count = std::get<std::vector<int32_t>>(inputs.at("count"))[0];
  std::vector<float> logits(100 * 8404, 0.f);
  for (int t = 0; t < count; ++t)
    logits[t * 8404 + 3] = 10.f;
  return {{"logits", std::move(logits)}};
}
} // namespace paraformer
