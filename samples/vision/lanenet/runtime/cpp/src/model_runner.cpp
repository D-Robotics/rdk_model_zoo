// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "model_runner.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace lanenet {
namespace {
void checked(int rc, const char *operation) {
  if (rc != 0)
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(rc));
}
std::size_t allocation(std::int64_t bytes) {
  if (bytes <= 0 || bytes > std::numeric_limits<int>::max())
    throw std::invalid_argument("Invalid SDK allocation size");
  return static_cast<std::size_t>(bytes);
}
ScalarType scalar(int type) {
  switch (type) {
  case HB_DNN_TENSOR_TYPE_F32:
    return ScalarType::Float32;
  case HB_DNN_TENSOR_TYPE_S64:
    return ScalarType::Int64;
  case HB_DNN_TENSOR_TYPE_S32:
    return ScalarType::Int32;
  case HB_DNN_TENSOR_TYPE_S16:
    return ScalarType::Int16;
  case HB_DNN_TENSOR_TYPE_S8:
    return ScalarType::Int8;
  case HB_DNN_TENSOR_TYPE_U8:
    return ScalarType::UInt8;
  default:
    throw std::invalid_argument("Unsupported SDK output scalar type");
  }
}
TensorSpec bind_tensor(hbDNNTensorProperties &properties, bool input) {
  const auto rank = properties.validShape.numDimensions;
  if (rank <= 0 || rank > 8)
    throw std::invalid_argument("Invalid tensor rank");
  TensorSpec spec{scalar(properties.tensorType), {}, {}, 0};
  for (int i = 0; i < rank; ++i) {
    if (properties.validShape.dimensionSize[i] <= 0)
      throw std::invalid_argument("Invalid fixed tensor dimension");
    spec.shape.push_back(properties.validShape.dimensionSize[i]);
  }
  if (input && (spec.type != ScalarType::Float32 ||
                spec.shape != std::vector<std::size_t>{1, 3, 256, 512}))
    throw std::invalid_argument("Expected RGB featuremap input, not NV12");
  // Resolve the source S100 dynamic stride convention without reading rank+1.
  for (int i = rank - 1; i >= 0; --i) {
    if (input && properties.stride[i] == -1) {
      std::int64_t value = static_cast<std::int64_t>(element_bytes(spec.type));
      if (i + 1 < rank) {
        const auto child = properties.stride[i + 1];
        if (child <= 0 ||
            child > std::numeric_limits<int>::max() /
                        static_cast<std::int64_t>(spec.shape[i + 1]))
          throw std::invalid_argument("Dynamic stride overflow");
        value = child * static_cast<std::int64_t>(spec.shape[i + 1]);
        if (value > std::numeric_limits<int>::max() - 31)
          throw std::invalid_argument("Aligned stride overflow");
        value = (value + 31) / 32 * 32;
      }
      properties.stride[i] = value;
    }
  }
  for (int i = 0; i < rank; ++i) {
    if (properties.stride[i] <= 0)
      throw std::invalid_argument("Unresolved/invalid tensor byte stride");
    spec.strides.push_back(properties.stride[i]);
  }
  if (input) {
    const auto outer = properties.stride[0];
    if (outer > std::numeric_limits<int>::max() /
                    static_cast<std::int64_t>(spec.shape[0]))
      throw std::invalid_argument("Input allocation overflow");
    spec.capacity =
        allocation(outer * static_cast<std::int64_t>(spec.shape[0]));
  } else
    spec.capacity = allocation(properties.alignedByteSize);
  validate_layout(spec);
  return spec;
}
struct TaskGuard {
  hbUCPTaskHandle_t handle = nullptr;
  ~TaskGuard() {
    if (handle)
      hbUCPReleaseTask(handle);
  }
};
} // namespace
struct ModelRunner::Impl {
  hbDNNPackedHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;
  hbDNNTensor input{};
  std::vector<hbDNNTensor> outputs;
  TensorSpec input_spec{};
  std::vector<TensorSpec> output_specs;
  std::string name;
  ~Impl() {
    if (input.sysMem.virAddr)
      hbUCPFree(&input.sysMem);
    for (auto &output : outputs)
      if (output.sysMem.virAddr)
        hbUCPFree(&output.sysMem);
    if (packed)
      hbDNNRelease(packed);
  }
};
ModelRunner::ModelRunner(const std::string &path, ExecutionGate gate)
    : impl_(std::make_unique<Impl>()) {
  if (gate)
    gate();
  else
    require_s100_board();
  std::ifstream file(path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing/empty model file");
  const char *filename = path.c_str();
  checked(hbDNNInitializeFromFiles(&impl_->packed, &filename, 1),
          "model initialization");
  const char **names = nullptr;
  int count = 0;
  checked(hbDNNGetModelNameList(&names, &count, impl_->packed), "model names");
  if (count != 1 || !names || !names[0])
    throw std::invalid_argument("Expected one named model");
  impl_->name = names[0];
  checked(hbDNNGetModelHandle(&impl_->model, impl_->packed, names[0]),
          "model handle");
  int32_t inputs = 0, outputs = 0;
  checked(hbDNNGetInputCount(&inputs, impl_->model), "input count");
  checked(hbDNNGetOutputCount(&outputs, impl_->model), "output count");
  if (inputs != 1 || outputs < 2 || outputs > 64)
    throw std::invalid_argument(
        "Expected one input and bounded multiple outputs");
  impl_->outputs.resize(outputs);
  checked(
      hbDNNGetInputTensorProperties(&impl_->input.properties, impl_->model, 0),
      "input properties");
  impl_->input_spec = bind_tensor(impl_->input.properties, true);
  for (int i = 0; i < outputs; ++i) {
    checked(hbDNNGetOutputTensorProperties(&impl_->outputs[i].properties,
                                           impl_->model, i),
            "output properties");
    impl_->output_specs.push_back(
        bind_tensor(impl_->outputs[i].properties, false));
  }
  bind_roles(
      impl_->output_specs); // reject missing/ambiguous roles before allocation
  checked(hbUCPMallocCached(&impl_->input.sysMem,
                            static_cast<int>(impl_->input_spec.capacity), 0),
          "input allocation");
  if (!impl_->input.sysMem.virAddr)
    throw std::runtime_error("Null input allocation");
  for (int i = 0; i < outputs; ++i) {
    checked(hbUCPMallocCached(&impl_->outputs[i].sysMem,
                              static_cast<int>(impl_->output_specs[i].capacity),
                              0),
            "output allocation");
    if (!impl_->outputs[i].sysMem.virAddr)
      throw std::runtime_error("Null output allocation");
  }
}
ModelRunner::~ModelRunner() = default;
const std::string &ModelRunner::model_name() const { return impl_->name; }
const TensorSpec &ModelRunner::input_spec() const { return impl_->input_spec; }
const std::vector<TensorSpec> &ModelRunner::output_specs() const {
  return impl_->output_specs;
}
std::vector<RawTensor> ModelRunner::run(const std::vector<float> &prepared) {
  write_input(prepared, impl_->input_spec, impl_->input.sysMem.virAddr);
  checked(hbUCPMemFlush(&impl_->input.sysMem, HB_SYS_MEM_CACHE_CLEAN),
          "input cache clean");
  TaskGuard task;
  checked(hbDNNInferV2(&task.handle, impl_->outputs.data(), &impl_->input,
                       impl_->model),
          "inference creation");
  hbUCPSchedParam schedule;
  HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
  schedule.backend = HB_UCP_BPU_CORE_ANY;
  checked(hbUCPSubmitTask(task.handle, &schedule), "task submission");
  checked(hbUCPWaitTaskDone(task.handle, 0), "task wait");
  std::vector<RawTensor> result;
  result.reserve(impl_->outputs.size());
  for (std::size_t i = 0; i < impl_->outputs.size(); ++i) {
    auto &output = impl_->outputs[i];
    checked(hbUCPMemFlush(&output.sysMem, HB_SYS_MEM_CACHE_INVALIDATE),
            "output cache invalidate");
    RawTensor raw{impl_->output_specs[i],
                  std::vector<unsigned char>(impl_->output_specs[i].capacity)};
    std::memcpy(raw.bytes.data(), output.sysMem.virAddr, raw.bytes.size());
    if (raw.spec.type == ScalarType::Float32) {
      const auto compact = compact_bytes(raw);
      for (std::size_t j = 0; j < compact.size(); j += 4) {
        float value;
        std::memcpy(&value, compact.data() + j, 4);
        if (!std::isfinite(value))
          throw std::invalid_argument("Nonfinite raw F32 output");
      }
    }
    result.push_back(std::move(raw));
  }
  return result;
}
} // namespace lanenet
