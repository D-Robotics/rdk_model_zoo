// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "model_runner.hpp"
#include "tensor_contract.hpp"
#include <algorithm>
#include <cstring>
#include <dnn/hb_dnn.h>
#include <dnn/hb_sys.h>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace yolo26_depth {
namespace {
void check(int code, const char *operation) {
  if (code)
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(code));
}
struct TaskLease {
  hbDNNTaskHandle_t handle = nullptr;
  ~TaskLease() {
    if (handle)
      hbDNNReleaseTask(handle);
  }
};
TensorLayout output_layout(const hbDNNTensorProperties &p) {
  if (p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.quantiType != NONE ||
      p.validShape.numDimensions != 4 || p.alignedShape.numDimensions != 4 ||
      p.alignedByteSize <= 0)
    throw std::invalid_argument(
        "Expected float32 NONE-quantized four-dimensional depth output");
  TensorLayout layout;
  layout.capacity = static_cast<std::size_t>(p.alignedByteSize);
  for (int i = 0; i < 4; ++i) {
    const auto byte_stride = static_cast<long long>(p.stride[i]);
    if (p.validShape.dimensionSize[i] <= 0 ||
        p.alignedShape.dimensionSize[i] <= 0 || byte_stride < 0)
      throw std::invalid_argument("Invalid output dimensions/strides");
    layout.valid[i] = static_cast<std::size_t>(p.validShape.dimensionSize[i]);
    layout.aligned[i] =
        static_cast<std::size_t>(p.alignedShape.dimensionSize[i]);
    layout.strides[i] = static_cast<std::size_t>(p.stride[i]);
  }
  validated_strides(layout);
  return layout;
}
void validate_input(const hbDNNTensorProperties &p) {
  constexpr int bytes = kInputSize * kInputSize * 3 / 2;
  if (p.tensorType != HB_DNN_IMG_TYPE_NV12 || p.validShape.numDimensions != 4 ||
      p.alignedShape.numDimensions != 4 || p.alignedByteSize < bytes)
    throw std::invalid_argument(
        "Expected compact NV12 input with sufficient storage");
  std::array<int, 4> shape{};
  for (int i = 0; i < 4; ++i) {
    shape[i] = p.validShape.dimensionSize[i];
    if (p.alignedShape.dimensionSize[i] != shape[i])
      throw std::invalid_argument("Padded NV12 input geometry is not supported "
                                  "by this compact pyramid binding");
  }
  if (shape != std::array<int, 4>{1, 3, 768, 768} &&
      shape != std::array<int, 4>{1, 768, 768, 3})
    throw std::invalid_argument(
        "NV12 logical input must describe 768-square geometry");
}
} // namespace
struct ModelRunner::Impl {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;
  hbDNNTensor input{}, output{};
  bool input_allocated = false, output_allocated = false;
  TensorLayout layout;
  std::string name;
  ~Impl() {
    if (input_allocated)
      hbSysFreeMem(&input.sysMem[0]);
    if (output_allocated)
      hbSysFreeMem(&output.sysMem[0]);
    if (packed)
      hbDNNRelease(packed);
  }
};
ModelRunner::ModelRunner(const std::string &path, ExecutionGate gate)
    : impl_(std::make_unique<Impl>()) {
  (gate ? gate : require_x5_board)();
  if (!std::filesystem::is_regular_file(path) ||
      !std::filesystem::file_size(path))
    throw std::invalid_argument("Missing or empty BIN model");
  const char *file = path.c_str();
  check(hbDNNInitializeFromFiles(&impl_->packed, &file, 1),
        "hbDNNInitializeFromFiles");
  const char **names = nullptr;
  int count = 0;
  check(hbDNNGetModelNameList(&names, &count, impl_->packed),
        "hbDNNGetModelNameList");
  if (count != 1 || !names || !names[0])
    throw std::invalid_argument("Expected exactly one named model");
  impl_->name = names[0];
  check(hbDNNGetModelHandle(&impl_->model, impl_->packed, names[0]),
        "hbDNNGetModelHandle");
  check(hbDNNGetInputCount(&count, impl_->model), "hbDNNGetInputCount");
  if (count != 1)
    throw std::invalid_argument("Expected exactly one NV12 input");
  check(hbDNNGetOutputCount(&count, impl_->model), "hbDNNGetOutputCount");
  if (count != 1)
    throw std::invalid_argument("Expected exactly one depth output");
  check(
      hbDNNGetInputTensorProperties(&impl_->input.properties, impl_->model, 0),
      "hbDNNGetInputTensorProperties");
  check(hbDNNGetOutputTensorProperties(&impl_->output.properties, impl_->model,
                                       0),
        "hbDNNGetOutputTensorProperties");
  validate_input(impl_->input.properties);
  impl_->layout = output_layout(impl_->output.properties);
  check(hbSysAllocCachedMem(&impl_->input.sysMem[0],
                            impl_->input.properties.alignedByteSize),
        "hbSysAllocCachedMem input");
  impl_->input_allocated = true;
  check(hbSysAllocCachedMem(&impl_->output.sysMem[0],
                            impl_->output.properties.alignedByteSize),
        "hbSysAllocCachedMem output");
  impl_->output_allocated = true;
}
ModelRunner::~ModelRunner() = default;
const std::string &ModelRunner::model_name() const { return impl_->name; }
std::vector<float> ModelRunner::run(const std::vector<std::uint8_t> &nv12) {
  if (nv12.size() != kInputSize * kInputSize * 3 / 2)
    throw std::invalid_argument("Incorrect compact NV12 byte count");
  std::memset(impl_->input.sysMem[0].virAddr, 0,
              impl_->input.properties.alignedByteSize);
  std::memcpy(impl_->input.sysMem[0].virAddr, nv12.data(), nv12.size());
  check(hbSysFlushMem(&impl_->input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem input");
  TaskLease task;
  hbDNNInferCtrlParam control;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
  auto *output = &impl_->output;
  check(
      hbDNNInfer(&task.handle, &output, &impl_->input, impl_->model, &control),
      "hbDNNInfer");
  check(hbDNNWaitTaskDone(task.handle, 0), "hbDNNWaitTaskDone");
  check(hbSysFlushMem(&impl_->output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
        "hbSysFlushMem output");
  return read_log_depth(impl_->output.sysMem[0].virAddr, impl_->layout);
}
} // namespace yolo26_depth
