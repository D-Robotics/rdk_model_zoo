// Test-only implementation linked instead of vendor SDK and model preflight.
#include "sdk_runner.hpp"
#include <cstdlib>
#include <numeric>
#include <stdexcept>
namespace himloco {
class SdkRunner::Impl {
public:
  int calls = 0;
  int priority = -1;
  std::string name = "fixture", version = "fixture-only";
  TensorMetadata input{"obs_history", {1, 1, 1, 270}, {1, 1, 1, 272}, 0, 3, 0,
                       1088};
  TensorMetadata output{"actions", {1, 1, 1, 12}, {1, 1, 1, 16}, 0, 3, 0, 64};
};
void verify_native_model(const std::string &) {}
SdkRunner::SdkRunner(const NativeConfig &c) : impl_(std::make_unique<Impl>()) {
  impl_->priority = c.priority;
}
SdkRunner::~SdkRunner() = default;
RawOutputs SdkRunner::run(const std::vector<float> &) {
  auto *fail = std::getenv("HIMLOCO_FIXTURE_FAIL_AFTER");
  if (fail && impl_->calls >= std::stoi(fail))
    throw std::runtime_error("Injected fixture failure");
  ++impl_->calls;
  std::vector<float> v(12);
  std::iota(v.begin(), v.end(), 0.f);
  return {v, 0.25};
}
const TensorMetadata &SdkRunner::input_metadata() const { return impl_->input; }
const TensorMetadata &SdkRunner::output_metadata() const {
  return impl_->output;
}
const std::string &SdkRunner::model_name() const { return impl_->name; }
const std::string &SdkRunner::runtime_version() const { return impl_->version; }
int SdkRunner::priority() const { return impl_->priority; }
} // namespace himloco
