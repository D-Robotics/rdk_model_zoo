// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "pipeline.hpp"
#include "sha256.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <utility>

// The S-series UCP adapter is compiled only when the build explicitly defines
// PARAFORMER_ENABLE_UCP=1: PARAFORMER_BUILD_SDK=ON with the board toolchain,
// or the fake-SDK test target that deliberately uses the declaration-only
// fakes under the yolo runtime's test/fake_dnn_io/ucp. Header visibility alone
// must not turn SDK calls on in SDK-free builds; those link a rejected-
// transport stub below instead, exactly as the host CLI fixture links its own
// marked transport double.
#ifdef PARAFORMER_ENABLE_UCP
#define PARAFORMER_HAVE_UCP 1
#include "backend.hpp"
#include <cstring>
#ifndef YOLO_DNN_STACK_UCP
#error "Paraformer native SDK adapter requires the S-series UCP stack"
#endif
#endif

namespace paraformer {
namespace {
// --- shared stage-validation helpers (pipeline) -------------------------------
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
void validate(const std::vector<float> &values, size_t count) {
  if (values.size() != count ||
      std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument("Invalid finite float32 tensor geometry");
}
// --- decode helpers ------------------------------------------------------------
void finite_values(const std::vector<float> &values, size_t expected,
                   const char *label) {
  if (values.size() != expected ||
      std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument(label);
}
// --- preflight helpers ---------------------------------------------------------
std::string digest(std::string value) {
  if (value.size() != 64 ||
      !std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
               (c >= 'A' && c <= 'F');
      }))
    throw std::invalid_argument("Expected a 64-digit model SHA-256");
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
std::filesystem::path model_file(const std::string &path) {
  if (!std::filesystem::is_regular_file(path) ||
      std::filesystem::file_size(path) == 0)
    throw std::invalid_argument(
        "Missing, empty or non-regular Paraformer model");
  return std::filesystem::canonical(path);
}
} // namespace

// --- Preflight -----------------------------------------------------------------
std::string expected_asset_id(Stage stage) {
  const std::string prefix = "s:paraformer:s100/paraformer_large_";
  switch (stage) {
  case Stage::Encoder:
    return prefix + "encoder_400x560_s100.hbm";
  case Stage::Predictor:
    return prefix + "predictor_400x512_s100.hbm";
  case Stage::Decoder:
    return prefix + "decoder_400x512_s100.hbm";
  }
  throw std::invalid_argument("Unknown Paraformer stage");
}
void verify_group(const ModelGroup &group, const std::string &vocabulary,
                  const rdk::NativeIdentity &actual) {
  if (rdk::identify_target(actual) != "s100")
    throw std::invalid_argument(
        "Paraformer requires actual local S100 identity");
  std::set<Stage> stages;
  std::set<std::filesystem::path> paths;
  // Validate the complete selection before hashing any model.
  for (const auto &artifact : group) {
    const auto &model = artifact.model;
    if (model.target != "s100" ||
        artifact.asset_id != expected_asset_id(model.stage) ||
        !stages.insert(model.stage).second)
      throw std::invalid_argument(
          "Expected one matching S100 publication for each Paraformer stage");
    (void)digest(artifact.expected_sha256);
    if (!paths.insert(model_file(model.path)).second)
      throw std::invalid_argument(
          "Each Paraformer stage requires a distinct model file");
  }
  // Detect hard-linked aliases too; canonical paths alone do not identify them.
  for (size_t i = 0; i < group.size(); ++i)
    for (size_t j = 0; j < i; ++j)
      if (std::filesystem::equivalent(group[i].model.path, group[j].model.path))
        throw std::invalid_argument(
            "Paraformer stages alias the same model file");
  for (const auto &artifact : group)
    if (rdk::sha256_file(artifact.model.path) !=
        digest(artifact.expected_sha256))
      throw std::invalid_argument("Model SHA-256 mismatch: " +
                                  artifact.asset_id);
  if (rdk::sha256_file(vocabulary) != kVocabularySha256)
    throw std::invalid_argument(
        "Expected fixed published 8404-token vocabulary SHA-256");
}
SdkPreflight make_preflight(ModelGroup group, std::string vocabulary) {
  verify_group(group, vocabulary, rdk::read_native_identity());
  return [group = std::move(group),
          vocabulary = std::move(vocabulary)](const SdkModel &model) {
    const auto selected =
        std::find_if(group.begin(), group.end(), [&](const auto &a) {
          return a.model.stage == model.stage;
        });
    if (selected == group.end() || model.target != selected->model.target ||
        model_file(model.path) != model_file(selected->model.path))
      throw std::invalid_argument(
          "Runner model differs from the verified Paraformer group");
    verify_group(group, vocabulary, rdk::read_native_identity());
  };
}

// --- Decode contract -------------------------------------------------------------
void validate_vocabulary(const std::vector<std::string> &vocabulary) {
  if (vocabulary.size() != 8404 ||
      std::any_of(vocabulary.begin(), vocabulary.end(),
                  [](const auto &value) { return value.empty(); }) ||
      std::set<std::string>(vocabulary.begin(), vocabulary.end()).size() !=
          8404)
    throw std::invalid_argument("Invalid ordered vocabulary");
}
DecodedText decode(const std::vector<float> &logits, int token_count,
                   const std::vector<std::string> &vocabulary) {
  finite_values(logits, 100 * 8404, "Expected finite logits [1,100,8404]");
  validate_vocabulary(vocabulary);
  if (token_count < 0 || token_count > 100)
    throw std::invalid_argument("Invalid token count");
  DecodedText result;
  for (int t = 0; t < token_count; ++t) {
    const auto start = logits.begin() + size_t(t) * 8404;
    const int id = int(std::max_element(start, start + 8404) - start);
    result.token_ids.push_back(id);
    auto token = vocabulary[size_t(id)];
    if (token.size() >= 2 && token.front() == '<' && token.back() == '>')
      continue;
    size_t marker = 0;
    while ((marker = token.find("@@", marker)) != std::string::npos)
      token.erase(marker, 2);
    result.text += token;
  }
  return result;
}

#ifdef PARAFORMER_HAVE_UCP
// --- UCP adapter ----------------------------------------------------------------
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
#endif
#if !defined(PARAFORMER_HAVE_UCP) && !defined(PARAFORMER_HOST_FIXTURE)
// --- SDK-free linkage stub ------------------------------------------------------
// SDK-free library builds still link the declared runner interface because the
// native Pipeline owns runners by type; every use rejects clearly at runtime.
// The host CLI fixture compiles its own marked transport double instead.
struct SdkRunner::Impl {};
SdkRunner::SdkRunner(SdkModel, SdkPreflight) {
  throw std::runtime_error(
      "Paraformer native runner requires the S-series UCP SDK or the host "
      "fixture");
}
SdkRunner::~SdkRunner() = default;
const SdkMetadata &SdkRunner::metadata() const {
  throw std::runtime_error(
      "Paraformer native runner requires the S-series UCP SDK or the host "
      "fixture");
}
RawTensors SdkRunner::infer(const RawTensors &) {
  throw std::runtime_error(
      "Paraformer native runner requires the S-series UCP SDK or the host "
      "fixture");
}
#endif

// --- Pipeline -------------------------------------------------------------------
namespace {
size_t stage_index(Stage stage) {
  switch (stage) {
  case Stage::Encoder:
    return 0;
  case Stage::Predictor:
    return 1;
  case Stage::Decoder:
    return 2;
  }
  throw std::invalid_argument("Unknown Paraformer stage");
}
} // namespace

Pipeline::Pipeline(Encoder encoder, Predictor predictor, Decoder decoder,
                   std::vector<std::string> vocabulary)
    : encoder_(std::move(encoder)), predictor_(std::move(predictor)),
      decoder_(std::move(decoder)), vocabulary_(std::move(vocabulary)) {
  if (!encoder_ || !predictor_ || !decoder_)
    throw std::invalid_argument("Each model requires a raw runner");
  validate_vocabulary(vocabulary_);
}
Pipeline::Pipeline(const ModelGroup &models, const SdkPreflight &preflight,
                   std::vector<std::string> vocabulary)
    : vocabulary_(std::move(vocabulary)) {
  const std::array<Stage, 3> stages{Stage::Encoder, Stage::Predictor,
                                    Stage::Decoder};
  for (size_t i = 0; i < stages.size(); ++i) {
    const auto selected = std::find_if(
        models.begin(), models.end(), [&](const ModelArtifact &artifact) {
          return artifact.model.stage == stages[i];
        });
    if (selected == models.end())
      throw std::invalid_argument(
          "Expected one artifact per Paraformer stage");
    native_[i] = std::make_unique<SdkRunner>(selected->model, preflight);
  }
  encoder_ = [this](const std::vector<float> &features) {
    auto out = native_[0]->infer({{"features", features}});
    return std::move(std::get<std::vector<float>>(out.at("context")));
  };
  predictor_ = [this](const std::vector<float> &context) {
    auto out = native_[1]->infer({{"context", context}});
    return PredictorOutput{
        std::move(std::get<std::vector<float>>(out.at("alphas"))),
        std::move(std::get<std::vector<float>>(out.at("hidden")))};
  };
  decoder_ = [this](const DecoderInput &input) {
    auto out = native_[2]->infer(
        {{"context", input.context},
         {"acoustic", input.acoustic},
         {"count", std::vector<int32_t>{input.token_count}},
         {"bias", std::vector<float>(input.bias.begin(), input.bias.end())}});
    return std::move(std::get<std::vector<float>>(out.at("logits")));
  };
  if (!encoder_ || !predictor_ || !decoder_)
    throw std::invalid_argument("Each model requires a raw runner");
  validate_vocabulary(vocabulary_);
}
const SdkMetadata &Pipeline::metadata(Stage stage) const {
  const auto &runner = native_[stage_index(stage)];
  if (!runner)
    throw std::logic_error("Pipeline metadata requires the native runtime");
  return runner->metadata();
}
Prediction Pipeline::predict(const std::vector<float> &features,
                             int valid_frames) const {
  validate(features, 400 * 560);
  if (valid_frames < 1 || valid_frames > 400)
    throw std::invalid_argument("valid_frames must be in [1,400]");
  Prediction result;
  auto start = Clock::now();
  auto context = encoder_(features);
  result.timings.encoder_ms = milliseconds(start);
  validate(context, 400 * 512);
  start = Clock::now();
  auto predictor = predictor_(context);
  result.timings.predictor_ms = milliseconds(start);
  validate(predictor.weights, 401);
  validate(predictor.hidden, 401 * 512);
  start = Clock::now();
  auto acoustic = cif(predictor.weights, predictor.hidden, valid_frames);
  result.timings.cif_ms = milliseconds(start);
  result.token_count = acoustic.token_count;
  if (result.token_count == 0)
    return result;
  validate(acoustic.acoustic, 100 * 512);
  start = Clock::now();
  auto logits = decoder_(
      DecoderInput{context, acoustic.acoustic, result.token_count, bias_});
  result.timings.decoder_ms = milliseconds(start);
  auto decoded = decode(logits, result.token_count, vocabulary_);
  result.text = std::move(decoded.text);
  result.token_ids = std::move(decoded.token_ids);
  result.decoder_executed = true;
  return result;
}
} // namespace paraformer
