// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "policy.hpp"

#include "platform_identity.h"
#include "sha256.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

// The X5 BPU binding is compiled only when the build explicitly defines
// HIMLOCO_ENABLE_DNN=1: HIMLOCO_BUILD_SDK=ON on the board toolchain, or a
// test target that deliberately enables the declaration-only host fixtures
// in tests/fixtures. Header visibility alone must not turn SDK calls on in
// SDK-free builds.
#ifdef HIMLOCO_ENABLE_DNN
#define HIMLOCO_HAVE_X5_DNN 1
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include <chrono>
#include <cstring>
#include <limits>
#endif

namespace himloco {
namespace {
void Validate(const std::vector<float> &values, std::size_t count,
              const char *name) {
  if (values.size() != count)
    throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                std::to_string(count) + " float32 values");
  if (!std::all_of(values.begin(), values.end(),
                   [](float value) { return std::isfinite(value); }))
    throw std::invalid_argument(std::string(name) + " contains NaN/Inf");
}
void ValidateRaw(const RawOutputs &raw) {
  Validate(raw.actions, kOutputElements, "actions");
  if (!std::isfinite(raw.latency_ms) || raw.latency_ms < 0)
    throw std::invalid_argument("latency must be finite and non-negative");
}
}  // namespace

void verify_native_model(const std::string &model_path) {
#ifdef HIMLOCO_HOST_FIXTURE
  // Host CLI fixture build: the board/digest gate is inert so the fixture
  // binary exercises the full production run flow without a board.
  (void)model_path;
  return;
#else
  if (rdk::identify_target(rdk::read_native_identity()) != "x5")
    throw std::runtime_error(
        "HIMLoco native inference requires actual X5 board identity");
  if (std::filesystem::path(model_path).extension() != ".bin")
    throw std::invalid_argument("HIMLoco native model must use .bin");
  // Published identity from docs/release/x5/models.yaml; parity is host-tested.
  constexpr const char *expected =
      "7ce46ca2628f8bc236da0e8564180a1de92847bddf1ec00717ce7aa93e8c3e6a";
  if (rdk::sha256_file(model_path) != expected)
    throw std::runtime_error(
        "HIMLoco published model SHA-256 mismatch or file unreadable; prepare "
        "the exact published BIN explicitly");
#endif
}

#ifdef HIMLOCO_HAVE_X5_DNN
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
TensorMetadata ValidateTensor(const char *name, const char *expected,
                              const hbDNNTensorProperties &p,
                              std::size_t count) {
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

/// One packed model and reusable buffers, owned by RAII. Not thread-safe.
/// Validates board/model and initializes the SDK; throws on failure.
class SdkRuntime {
 public:
  SdkRuntime(const NativeConfig &config, const ModelGate &gate)
      : priority_(config.priority) {
    if (priority_ < -1 || priority_ > 255)
      throw std::invalid_argument("priority must be -1 or [0,255]");
    if (gate)
      gate(config.model_path);
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
        ValidateTensor(input_name, "obs_history", input_properties,
                       kInputElements);
    output_meta_ =
        ValidateTensor(output_name, "actions", output_properties,
                       kOutputElements);
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
  RuntimeInfo Info() const {
    return {input_meta_, output_meta_, name_, version_, priority_, true};
  }

 private:
  // Declaration order ensures buffers are released before the packed model,
  // including exceptions thrown during constructor initialization.
  PackedModel packed_;
  Tensor input_, output_;
  hbDNNHandle_t model_ = nullptr;
  TensorMetadata input_meta_, output_meta_;
  std::string name_, version_;
  int priority_;
};
}  // namespace

HimLoco::HimLoco(const NativeConfig &config, ModelGate gate) {
  auto runtime = std::make_shared<SdkRuntime>(config, gate);
  info_ = runtime->Info();
  runner_ = [runtime](const std::vector<float> &values) {
    return runtime->Run(values);
  };
}
#else
HimLoco::HimLoco(const NativeConfig &config, ModelGate gate) {
  // Without the X5 DNN headers there is no SDK binding to load; host builds
  // provide this constructor through test doubles instead.
  static_cast<void>(config);
  static_cast<void>(gate);
  throw std::runtime_error("HIMLoco native runtime requires the X5 DNN SDK");
}
#endif

HimLoco::HimLoco(Runner runner, RuntimeInfo info)
    : runner_(std::move(runner)), info_(std::move(info)) {
  if (!runner_)
    throw std::invalid_argument("HimLoco requires a runner");
}

PreparedInput HimLoco::preprocess(const std::vector<float> &observation) const {
  Validate(observation, kInputElements, "obs_history");
  return {observation};
}
RawOutputs HimLoco::infer(const PreparedInput &input) const {
  Validate(input.values, kInputElements, "obs_history");
  RawOutputs raw = runner_(input.values);
  ValidateRaw(raw);
  return raw;
}
InferenceResult HimLoco::postprocess(const RawOutputs &raw) const {
  ValidateRaw(raw);
  return {raw.actions, raw.latency_ms};
}
InferenceResult HimLoco::predict(const std::vector<float> &observation) const {
  return postprocess(infer(preprocess(observation)));
}

namespace {
const RuntimeInfo &RequireInfo(const RuntimeInfo &info) {
  if (!info.valid)
    throw std::logic_error(
        "HIMLoco runtime metadata requires the native SDK constructor");
  return info;
}
}  // namespace

const TensorMetadata &HimLoco::input_metadata() const {
  return RequireInfo(info_).input;
}
const TensorMetadata &HimLoco::output_metadata() const {
  return RequireInfo(info_).output;
}
const std::string &HimLoco::model_name() const {
  return RequireInfo(info_).model_name;
}
const std::string &HimLoco::runtime_version() const {
  return RequireInfo(info_).runtime_version;
}
int HimLoco::priority() const { return RequireInfo(info_).priority; }
}  // namespace himloco
