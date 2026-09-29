// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "sdk_runner.h"
#include "common/dnn_resources.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#ifndef YOLO_DNN_STACK_UCP
#error "Paraformer native SDK adapter requires the S-series UCP stack"
#endif
namespace paraformer {
namespace {
void checked(int rc, const char *action) {
  if (rc)
    throw std::runtime_error(std::string(action) +
                             " failed: " + std::to_string(rc));
}
struct Contract {
  std::string role;
  std::vector<std::string> aliases;
  std::vector<int> shape;
  bool integer = false, optional = false;
};
std::vector<Contract> contracts(Stage stage, bool output) {
  const std::string context = "/encoder/after_norm/Add_1_output_0";
  switch (stage) {
  case Stage::Encoder:
    if (output)
      return {{"context", {context}, {1, 400, 512}}};
    return {{"features", {"speech"}, {1, 400, 560}}};
  case Stage::Predictor:
    if (output)
      return {{"alphas", {"/predictor/Add_output_0"}, {1, 401}},
              {"hidden", {"/predictor/Concat_5_output_0"}, {1, 401, 512}}};
    return {{"context", {context}, {1, 400, 512}}};
  case Stage::Decoder:
    if (output)
      return {{"logits", {"logits"}, {1, 100, 8404}},
              {"count", {"token_num"}, {1}, true, true}};
    return {{"context", {context}, {1, 400, 512}},
            {"count", {"token_num"}, {1}, true},
            {"bias", {"bias_embed"}, {1, 1, 512}},
            {"acoustic", {"onnx::Shape_8609", "shape_8609"}, {1, 100, 512}}};
  }
  throw std::invalid_argument("Unknown Paraformer stage");
}
TensorMetadata validate(const char *name, const hbDNNTensorProperties &p,
                        const Contract &c) {
  if (p.tensorType !=
          (c.integer ? HB_DNN_TENSOR_TYPE_S32 : HB_DNN_TENSOR_TYPE_F32) ||
      p.quantiType != NONE ||
      p.validShape.numDimensions != int(c.shape.size()) ||
      p.alignedByteSize <= 0)
    throw std::invalid_argument(
        "Expected fixed unquantized tensor type/shape: " + c.role);
  TensorMetadata m{name,    c.role, c.integer ? "int32" : "float32",
                   c.shape, {},     size_t(p.alignedByteSize)};
  int64_t span = 4;
  for (int axis = int(c.shape.size()) - 1; axis >= 0; --axis) {
    if (p.validShape.dimensionSize[axis] != c.shape[axis] ||
        p.stride[axis] < span || p.stride[axis] % 4 ||
        p.stride[axis] > p.alignedByteSize / c.shape[axis])
      throw std::invalid_argument(
          "Invalid tensor dimensions/byte strides/allocation: " + c.role);
    span = int64_t(p.stride[axis]) * c.shape[axis];
  }
  for (size_t i = 0; i < c.shape.size(); ++i)
    m.strides.push_back(p.stride[i]);
  return m;
}
size_t elements(const TensorMetadata &m) {
  size_t n = 1;
  for (int d : m.shape)
    n *= size_t(d);
  return n;
}
size_t offset(size_t flat, const TensorMetadata &m) {
  size_t result = 0;
  for (int axis = int(m.shape.size()) - 1; axis >= 0; --axis) {
    result += (flat % size_t(m.shape[axis])) * size_t(m.strides[axis]);
    flat /= size_t(m.shape[axis]);
  }
  return result;
}
void validate_input(const RawTensor &value, const TensorMetadata &m) {
  if (m.dtype == "int32") {
    const auto *v = std::get_if<std::vector<int32_t>>(&value);
    if (!v || v->size() != elements(m) || (*v)[0] < 0 || (*v)[0] > 100)
      throw std::invalid_argument("Expected token count int32 in [0,100]");
  } else {
    const auto *v = std::get_if<std::vector<float>>(&value);
    if (!v || v->size() != elements(m) ||
        std::any_of(v->begin(), v->end(),
                    [](float x) { return !std::isfinite(x); }))
      throw std::invalid_argument("Expected finite float32 tensor: " + m.role);
  }
}
} // namespace
struct SdkRunner::Impl {
  yolo::PackedModelOwner packed;
  hbDNNHandle_t model = nullptr;
  std::vector<std::unique_ptr<yolo::OutputTensorOwner>> input_owners,
      output_owners;
  std::vector<hbDNNTensor> inputs, outputs;
  SdkMetadata metadata;
  void bind(Stage stage, bool output) {
    const auto expected = contracts(stage, output);
    int32_t count = 0;
    checked(output ? hbDNNGetOutputCount(&count, model)
                   : hbDNNGetInputCount(&count, model),
            "Tensor count");
    const int required =
        std::count_if(expected.begin(), expected.end(),
                      [](const Contract &c) { return !c.optional; });
    if (count < required || count > int(expected.size()))
      throw std::invalid_argument("Unexpected tensor count");
    auto &owners = output ? output_owners : input_owners;
    auto &tensors = output ? outputs : inputs;
    auto &meta = output ? metadata.outputs : metadata.inputs;
    std::set<std::string> seen;
    // Query and validate the complete side before allocating its buffers.
    std::vector<hbDNNTensorProperties> properties;
    for (int i = 0; i < count; ++i) {
      const char *name = nullptr;
      checked(output ? hbDNNGetOutputName(&name, model, i)
                     : hbDNNGetInputName(&name, model, i),
              "Tensor name");
      if (!name || !*name)
        throw std::invalid_argument("Missing physical tensor name");
      const auto match = std::find_if(
          expected.begin(), expected.end(), [&](const Contract &c) {
            return std::find(c.aliases.begin(), c.aliases.end(), name) !=
                   c.aliases.end();
          });
      if (match == expected.end() || !seen.insert(match->role).second)
        throw std::invalid_argument(
            "Unknown or duplicate physical tensor role");
      hbDNNTensorProperties p{};
      checked(output ? hbDNNGetOutputTensorProperties(&p, model, i)
                     : hbDNNGetInputTensorProperties(&p, model, i),
              "Tensor properties");
      meta.push_back(validate(name, p, *match));
      properties.push_back(p);
    }
    for (const auto &c : expected)
      if (!c.optional && !seen.count(c.role))
        throw std::invalid_argument("Missing required tensor: " + c.role);
    for (const auto &p : properties) {
      auto owner = std::make_unique<yolo::OutputTensorOwner>();
      checked(owner->allocate(p), "Tensor allocation");
      tensors.push_back(owner->tensor);
      owners.push_back(std::move(owner));
    }
  }
};
SdkRunner::SdkRunner(SdkModel spec, SdkPreflight preflight)
    : impl_(std::make_unique<Impl>()) {
  if (spec.target != "s100")
    throw std::invalid_argument("Paraformer native target must be s100");
  (void)contracts(spec.stage, false);
  if (!preflight)
    throw std::invalid_argument(
        "Provide identity/artifact preflight before SDK use");
  preflight(spec);
  std::ifstream file(spec.path, std::ios::binary);
  if (!file || file.peek() == std::ifstream::traits_type::eof())
    throw std::invalid_argument("Missing or empty Paraformer model");
  const char *path = spec.path.c_str();
  checked(hbDNNInitializeFromFiles(&impl_->packed.handle, &path, 1),
          "Model initialization");
  if (!impl_->packed.handle)
    throw std::runtime_error("Null packed model");
  const char **names = nullptr;
  int count = 0;
  checked(hbDNNGetModelNameList(&names, &count, impl_->packed.handle),
          "Model names");
  if (count != 1 || !names || !names[0] || !*names[0])
    throw std::invalid_argument(
        "Expected exactly one named model per artifact");
  impl_->metadata.model_name = names[0];
  checked(hbDNNGetModelHandle(&impl_->model, impl_->packed.handle, names[0]),
          "Model handle");
  if (!impl_->model)
    throw std::runtime_error("Null model handle");
  impl_->bind(spec.stage, false);
  impl_->bind(spec.stage, true);
}
SdkRunner::~SdkRunner() = default;
const SdkMetadata &SdkRunner::metadata() const { return impl_->metadata; }
RawTensors SdkRunner::infer(const RawTensors &values) {
  if (values.size() != impl_->metadata.inputs.size())
    throw std::invalid_argument("Expected exactly the bound input roles");
  for (const auto &m : impl_->metadata.inputs) {
    const auto it = values.find(m.role);
    if (it == values.end())
      throw std::invalid_argument("Missing input role: " + m.role);
    validate_input(it->second, m);
  }
  for (size_t i = 0; i < impl_->inputs.size(); ++i) {
    auto &tensor = impl_->inputs[i];
    const auto &m = impl_->metadata.inputs[i];
    auto *dst = static_cast<unsigned char *>(YOLO_SYS_MEM(tensor)->virAddr);
    std::memset(dst, 0, m.allocation_bytes);
    std::visit(
        [&](const auto &v) {
          for (size_t j = 0; j < v.size(); ++j)
            std::memcpy(dst + offset(j, m), &v[j], 4);
        },
        values.at(m.role));
    checked(YOLO_SYS_FLUSH(YOLO_SYS_MEM(tensor), HB_SYS_MEM_CACHE_CLEAN),
            "Input cache clean");
  }
  checked(yolo::infer_tensors_sync(impl_->outputs.data(), impl_->inputs.data(),
                                   int(impl_->inputs.size()), impl_->model),
          "Inference");
  RawTensors result;
  for (size_t i = 0; i < impl_->outputs.size(); ++i) {
    auto &tensor = impl_->outputs[i];
    const auto &m = impl_->metadata.outputs[i];
    checked(YOLO_SYS_FLUSH(YOLO_SYS_MEM(tensor), HB_SYS_MEM_CACHE_INVALIDATE),
            "Output cache invalidate");
    const auto *src =
        static_cast<const unsigned char *>(YOLO_SYS_MEM(tensor)->virAddr);
    RawTensor value = m.dtype == "int32"
                          ? RawTensor(std::vector<int32_t>(elements(m)))
                          : RawTensor(std::vector<float>(elements(m)));
    std::visit(
        [&](auto &v) {
          for (size_t j = 0; j < v.size(); ++j)
            std::memcpy(&v[j], src + offset(j, m), 4);
        },
        value);
    result.emplace(m.role, std::move(value));
  }
  return result;
}
} // namespace paraformer
