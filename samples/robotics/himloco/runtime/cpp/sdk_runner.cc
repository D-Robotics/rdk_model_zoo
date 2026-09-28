// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "sdk_runner.hpp"
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace himloco {
namespace {
void Check(int code, const char *operation) {
  if (code != 0)
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(code));
}
struct PackedModel {
  hbPackedDNNHandle_t handle = nullptr;
  ~PackedModel() {
    if (handle)
      hbDNNRelease(handle);
  }
};
struct Tensor {
  hbDNNTensor value{};
  ~Tensor() {
    if (value.sysMem[0].virAddr)
      hbSysFreeMem(&value.sysMem[0]);
  }
  void Allocate(const hbDNNTensorProperties &properties) {
    value.properties = properties;
    Check(hbSysAllocCachedMem(&value.sysMem[0], properties.alignedByteSize),
          "hbSysAllocCachedMem");
    if (!value.sysMem[0].virAddr)
      throw std::runtime_error("SDK returned null tensor memory");
  }
};
struct Task {
  hbDNNTaskHandle_t handle = nullptr;
  ~Task() {
    if (handle)
      hbDNNReleaseTask(handle);
  }
};
std::vector<int> Shape(const hbDNNTensorShape &shape) {
  if (shape.numDimensions != 4)
    throw std::runtime_error("expected four-dimensional X5 tensor shape");
  std::vector<int> result;
  for (int i = 0; i < 4; ++i) {
    if (shape.dimensionSize[i] <= 0)
      throw std::runtime_error("tensor dimension must be positive");
    result.push_back(shape.dimensionSize[i]);
  }
  return result;
}
std::size_t Elements(const std::vector<int> &shape) {
  std::size_t count = 1;
  for (int d : shape) {
    if (count >
        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(d))
      throw std::runtime_error("tensor size overflow");
    count *= static_cast<std::size_t>(d);
  }
  return count;
}
TensorMetadata Validate(const char *name, const char *expected,
                        const hbDNNTensorProperties &p, std::size_t count) {
  if (!name || std::string(name) != expected)
    throw std::runtime_error("unexpected tensor name");
  if (p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.quantiType != NONE)
    throw std::runtime_error(
        "expected float32 tensor without manual dequantization");
  auto valid = Shape(p.validShape), aligned = Shape(p.alignedShape);
  if (valid[0] != 1 || Elements(valid) != count)
    throw std::runtime_error("unexpected logical tensor shape");
  for (std::size_t i = 0; i < valid.size(); ++i)
    if (aligned[i] < valid[i])
      throw std::runtime_error("aligned shape smaller than logical shape");
  if (p.alignedByteSize <= 0 || p.alignedByteSize % sizeof(float) != 0 ||
      Elements(aligned) >
          static_cast<std::size_t>(p.alignedByteSize) / sizeof(float))
    throw std::runtime_error("aligned tensor exceeds allocation capacity");
  return {name,
          valid,
          aligned,
          p.tensorLayout,
          p.tensorType,
          static_cast<int>(p.quantiType),
          p.alignedByteSize};
}
void ValidateInput(const std::vector<float> &input) {
  if (input.size() != kInputElements ||
      !std::all_of(input.begin(), input.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument(
        "obs_history must contain 270 finite float32 values");
}
} // namespace

class SdkRunner::Impl {
public:
  explicit Impl(const NativeConfig &config) : priority_(config.priority) {
    if (priority_ < -1 || priority_ > 255)
      throw std::invalid_argument("priority must be -1 or [0,255]");
    verify_native_model(config.model_path);
    const char *file = config.model_path.c_str();
    Check(hbDNNInitializeFromFiles(&packed_.handle, &file, 1),
          "hbDNNInitializeFromFiles");
    const char **names = nullptr;
    int count = 0;
    Check(hbDNNGetModelNameList(&names, &count, packed_.handle),
          "hbDNNGetModelNameList");
    if (count != 1 || !names || !names[0])
      throw std::runtime_error("expected exactly one packed model");
    name_ = names[0];
    Check(hbDNNGetModelHandle(&model_, packed_.handle, names[0]),
          "hbDNNGetModelHandle");
    int inputs = 0, outputs = 0;
    Check(hbDNNGetInputCount(&inputs, model_), "hbDNNGetInputCount");
    Check(hbDNNGetOutputCount(&outputs, model_), "hbDNNGetOutputCount");
    if (inputs != 1 || outputs != 1)
      throw std::runtime_error("expected exactly one input and output");
    const char *input_name = nullptr, *output_name = nullptr;
    hbDNNTensorProperties input_properties{}, output_properties{};
    Check(hbDNNGetInputName(&input_name, model_, 0), "hbDNNGetInputName");
    Check(hbDNNGetOutputName(&output_name, model_, 0), "hbDNNGetOutputName");
    Check(hbDNNGetInputTensorProperties(&input_properties, model_, 0),
          "hbDNNGetInputTensorProperties");
    Check(hbDNNGetOutputTensorProperties(&output_properties, model_, 0),
          "hbDNNGetOutputTensorProperties");
    input_meta_ =
        Validate(input_name, "obs_history", input_properties, kInputElements);
    output_meta_ =
        Validate(output_name, "actions", output_properties, kOutputElements);
    input_.Allocate(input_properties);
    // Preserve source X5 contract: explicitly submit compact logical input,
    // backed by the SDK-requested allocation; the remaining bytes are zeroed.
    input_.value.properties.alignedShape = input_properties.validShape;
    output_.Allocate(output_properties);
    const char *version = hbDNNGetVersion();
    version_ = version ? version : "unreported";
  }
  RawOutputs Run(const std::vector<float> &input) {
    ValidateInput(input);
    std::memset(input_.value.sysMem[0].virAddr, 0,
                input_meta_.aligned_byte_size);
    std::memcpy(input_.value.sysMem[0].virAddr, input.data(),
                input.size() * sizeof(float));
    Check(hbSysFlushMem(&input_.value.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "input cache clean");
    Task task;
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    if (priority_ >= 0)
      control.priority = priority_;
    hbDNNTensor *output = &output_.value;
    const auto start = std::chrono::steady_clock::now();
    Check(hbDNNInfer(&task.handle, &output, &input_.value, model_, &control),
          "hbDNNInfer");
    Check(hbDNNWaitTaskDone(task.handle, 0), "hbDNNWaitTaskDone");
    const auto end = std::chrono::steady_clock::now();
    Check(hbSysFlushMem(&output_.value.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
          "output cache invalidate");
    RawOutputs result;
    result.latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    const auto *values =
        static_cast<const float *>(output_.value.sysMem[0].virAddr);
    for (std::size_t i = 0; i < kOutputElements; ++i) {
      std::size_t remaining = i, offset = 0, stride = 1;
      for (int d = 3; d >= 0; --d) {
        offset += (remaining % output_meta_.valid_shape[d]) * stride;
        remaining /= output_meta_.valid_shape[d];
        stride *= output_meta_.aligned_shape[d];
      }
      const float value = values[offset];
      if (!std::isfinite(value))
        throw std::runtime_error("SDK returned NaN/Inf actions");
      result.actions.push_back(value);
    }
    return result;
  }
  // Declaration order ensures buffers are released before the packed model,
  // including exceptions thrown during constructor initialization.
  PackedModel packed_;
  Tensor input_, output_;
  hbDNNHandle_t model_ = nullptr;
  TensorMetadata input_meta_, output_meta_;
  std::string name_, version_;
  int priority_;
};

SdkRunner::SdkRunner(const NativeConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}
SdkRunner::~SdkRunner() = default;
RawOutputs SdkRunner::run(const std::vector<float> &input) {
  return impl_->Run(input);
}
const TensorMetadata &SdkRunner::input_metadata() const {
  return impl_->input_meta_;
}
const TensorMetadata &SdkRunner::output_metadata() const {
  return impl_->output_meta_;
}
const std::string &SdkRunner::model_name() const { return impl_->name_; }
const std::string &SdkRunner::runtime_version() const {
  return impl_->version_;
}
int SdkRunner::priority() const { return impl_->priority_; }
} // namespace himloco
