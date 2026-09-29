// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "sdk_runner.h"
#include "common/dnn_resources.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#ifndef YOLO_DNN_STACK_UCP
#error "ASR native SDK adapter requires the S-series UCP stack"
#endif
namespace asr {
namespace {
void checked(int rc, const char *action) {
  if (rc)
    throw std::runtime_error(std::string(action) +
                             " failed: " + std::to_string(rc));
}
void validate_tensor(const hbDNNTensorProperties &p,
                     const std::vector<int> &shape) {
  if (p.tensorType != HB_DNN_TENSOR_TYPE_F32 || p.quantiType != NONE ||
      p.validShape.numDimensions != static_cast<int>(shape.size()) ||
      p.alignedByteSize <= 0)
    throw std::invalid_argument("Expected fixed FLOAT32 tensor with explicit "
                                "allocation and no quantization");
  int64_t span = 4;
  for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
    if (shape[axis] <= 0 || p.validShape.dimensionSize[axis] != shape[axis] ||
        p.stride[axis] < span || p.stride[axis] % 4)
      throw std::invalid_argument(
          "Tensor dimensions or byte strides do not match ASR");
    // All accepted spans fit within the signed SDK allocation size; bound each
    // multiplication before computing it, including padded batch stride.
    if (p.stride[axis] > p.alignedByteSize / shape[axis])
      throw std::invalid_argument("Tensor strides exceed allocation");
    span = int64_t(p.stride[axis]) * shape[axis];
  }
}
} // namespace
struct SdkRunner::Impl {
  yolo::PackedModelOwner packed;
  hbDNNHandle_t model = nullptr;
  yolo::OutputTensorOwner input, output;
  SdkMetadata metadata;
};
SdkRunner::SdkRunner(SdkModel spec, SdkPreflight preflight)
    : impl_(std::make_unique<Impl>()) {
  if (spec.target != "s100" && spec.target != "s600")
    throw std::invalid_argument("ASR native target must be s100 or s600");
  if (!preflight)
    throw std::invalid_argument(
        "Provide board/model/vocabulary preflight before SDK use");
  preflight(spec);
  std::ifstream file(spec.path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing or empty ASR model");
  const char *path = spec.path.c_str();
  checked(hbDNNInitializeFromFiles(&impl_->packed.handle, &path, 1),
          "Model initialization");
  if (!impl_->packed.handle)
    throw std::runtime_error("SDK returned null packed model");
  const char **names = nullptr;
  int count = 0;
  checked(hbDNNGetModelNameList(&names, &count, impl_->packed.handle),
          "Model names");
  if (count != 1 || !names || !names[0] || !names[0][0])
    throw std::invalid_argument("Expected one named ASR model");
  impl_->metadata.model_name = names[0];
  checked(hbDNNGetModelHandle(&impl_->model, impl_->packed.handle, names[0]),
          "Model handle");
  if (!impl_->model)
    throw std::runtime_error("SDK returned null model handle");
  int32_t ni = 0, no = 0;
  checked(hbDNNGetInputCount(&ni, impl_->model), "Input count");
  checked(hbDNNGetOutputCount(&no, impl_->model), "Output count");
  if (ni != 1 || no != 1)
    throw std::invalid_argument("Expected one ASR input and output");
  hbDNNTensorProperties input{}, output{};
  checked(hbDNNGetInputTensorProperties(&input, impl_->model, 0),
          "Input properties");
  checked(hbDNNGetOutputTensorProperties(&output, impl_->model, 0),
          "Output properties");
  validate_tensor(input, {1, 30000});
  if (output.validShape.numDimensions != 3 ||
      output.validShape.dimensionSize[1] <= 0)
    throw std::invalid_argument("Expected ASR logits [1,T,3503]");
  validate_tensor(output, {1, output.validShape.dimensionSize[1], 3503});
  impl_->metadata.steps = output.validShape.dimensionSize[1];
  impl_->metadata.input_strides = {input.stride[0], input.stride[1]};
  impl_->metadata.output_strides = {output.stride[0], output.stride[1],
                                    output.stride[2]};
  impl_->metadata.input_bytes = input.alignedByteSize;
  impl_->metadata.output_bytes = output.alignedByteSize;
  checked(impl_->input.allocate(input), "Input allocation");
  checked(impl_->output.allocate(output), "Output allocation");
}
SdkRunner::~SdkRunner() = default;
const SdkMetadata &SdkRunner::metadata() const { return impl_->metadata; }
std::vector<float> SdkRunner::infer(const std::vector<float> &prepared) {
  if (prepared.size() != 30000 ||
      std::any_of(prepared.begin(), prepared.end(),
                  [](float v) { return !std::isfinite(v); }))
    throw std::invalid_argument("Expected finite prepared FLOAT32 [1,30000]");
  auto &input = impl_->input.tensor;
  auto &output = impl_->output.tensor;
  auto *dst = static_cast<unsigned char *>(YOLO_SYS_MEM(input)->virAddr);
  std::memset(dst, 0, impl_->metadata.input_bytes);
  for (size_t i = 0; i < prepared.size(); ++i)
    std::memcpy(dst + i * impl_->metadata.input_strides[1], &prepared[i], 4);
  checked(YOLO_SYS_FLUSH(YOLO_SYS_MEM(input), HB_SYS_MEM_CACHE_CLEAN),
          "Input cache clean");
  checked(yolo::infer_sync(&output, &input, 1, impl_->model), "Inference");
  checked(YOLO_SYS_FLUSH(YOLO_SYS_MEM(output), HB_SYS_MEM_CACHE_INVALIDATE),
          "Output cache invalidate");
  std::vector<float> result(impl_->metadata.steps * 3503);
  const auto *src =
      static_cast<const unsigned char *>(YOLO_SYS_MEM(output)->virAddr);
  for (size_t t = 0; t < impl_->metadata.steps; ++t)
    for (size_t v = 0; v < 3503; ++v)
      std::memcpy(&result[t * 3503 + v],
                  src + t * impl_->metadata.output_strides[1] +
                      v * impl_->metadata.output_strides[2],
                  4);
  return result;
}
} // namespace asr
