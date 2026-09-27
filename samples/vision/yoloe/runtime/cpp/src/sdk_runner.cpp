// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "sdk_runner.h"
#include "common/dnn_resources.h"
#include "common/task_output_binding.h"
#include "float_heads.h"
#include <fstream>
#include <utility>
namespace yoloe {
namespace {
void checked(int rc, const char *action) {
  if (rc)
    throw std::runtime_error(std::string(action) +
                             " failed: " + std::to_string(rc));
}
Protocol model_protocol(const SdkModel &model) {
  const bool e11 = model.variant.rfind("11", 0) == 0;
  if (!supported_native_model(model))
    throw std::invalid_argument("Unsupported YOLOE target/variant pair");
#ifdef YOLO_DNN_STACK_X5
  if (model.target != "x5")
    throw std::invalid_argument(
        "Target requires UCP, but this adapter uses X5 SDK");
#else
  if (model.target == "x5")
    throw std::invalid_argument(
        "Target requires X5, but this adapter uses UCP SDK");
#endif
  return e11 ? Protocol::E11 : Protocol::E26;
}
} // namespace
struct SdkRunner::Impl {
  Protocol family;
  yolo::PackedModelOwner packed;
  hbDNNHandle_t model = nullptr;
  yolo::InputPlan plan;
  yolo::Nv12Input input;
  yolo::TaskOutputs output;
  std::array<int, 10> roles{};
};
SdkRunner::SdkRunner(SdkModel spec, SdkPreflight preflight)
    : impl_(std::make_unique<Impl>()) {
  impl_->family = model_protocol(spec);
  if (!preflight)
    throw std::invalid_argument(
        "Provide board/artifact preflight before SDK use");
  preflight(spec);
  std::ifstream file(spec.path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing or empty model file");
  const char *path = spec.path.c_str();
  checked(hbDNNInitializeFromFiles(&impl_->packed.handle, &path, 1),
          "Model initialization");
  if (!impl_->packed.handle)
    throw std::runtime_error("SDK returned a null packed model");
  const char **names = nullptr;
  int count = 0;
  checked(hbDNNGetModelNameList(&names, &count, impl_->packed.handle),
          "Model names");
  if (count != 1 || !names || !names[0] || !names[0][0])
    throw std::invalid_argument("Expected one named model");
  checked(hbDNNGetModelHandle(&impl_->model, impl_->packed.handle, names[0]),
          "Model handle");
  if (!impl_->model)
    throw std::runtime_error("SDK returned a null model handle");
  std::string error;
  impl_->plan = yolo::probe_input_protocol(impl_->model, &error);
  const auto expected = spec.target == "x5" ? yolo::InputProtocol::kPackedNv12
                                            : yolo::InputProtocol::kSplitNv12;
  if (impl_->plan.protocol != expected || impl_->plan.input_h != 640 ||
      impl_->plan.input_w != 640)
    throw std::invalid_argument("Expected target-specific 640x640 NV12: " +
                                error);
  impl_->output.bind(
      impl_->model, 10, [&](const std::vector<yolo::OutputShape> &shapes) {
        impl_->roles =
            bind_heads(shapes, impl_->family == Protocol::E11 ? 64 : 4);
      });
  if (!impl_->input.allocate(impl_->model, impl_->plan))
    throw std::runtime_error("Cannot allocate YOLOE input");
  impl_->output.allocate();
}
SdkRunner::~SdkRunner() = default;
Protocol SdkRunner::protocol() const { return impl_->family; }
Heads SdkRunner::infer(const Nv12Input &input) {
  if (!impl_->input.upload_planes(impl_->plan, input.y.data(), input.y.size(),
                                  input.uv.data(), input.uv.size()))
    throw std::invalid_argument(
        "Invalid NV12 input or failed input cache clean");
  checked(yolo::infer_sync(impl_->output.tensors(), impl_->input.tensors(),
                           impl_->input.input_count(), impl_->model),
          "Inference");
  auto physical = impl_->output.read();
  Heads result;
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = std::move(physical.at(impl_->roles[i]));
  return result;
}
} // namespace yoloe
