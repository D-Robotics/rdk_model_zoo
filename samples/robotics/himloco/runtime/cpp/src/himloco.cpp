/**
 * @file himloco.cpp
 * @brief Implement fused HIMLoco inference with the RDK X5 DNN Runtime SDK.
 */

#include "himloco.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

namespace himloco {
namespace {

void Check(int code, const std::string& operation) {
  if (code != 0) {
    throw std::runtime_error(operation + " failed, error code=" +
                             std::to_string(code));
  }
}

class CachedMemory {
 public:
  explicit CachedMemory(int byte_size) {
    if (byte_size <= 0) {
      throw std::invalid_argument("BPU allocation size must be positive");
    }
    Check(hbSysAllocCachedMem(&memory_, static_cast<std::uint32_t>(byte_size)),
          "hbSysAllocCachedMem");
  }

  ~CachedMemory() {
    if (memory_.virAddr != nullptr) {
      hbSysFreeMem(&memory_);
    }
  }

  CachedMemory(const CachedMemory&) = delete;
  CachedMemory& operator=(const CachedMemory&) = delete;

  hbSysMem& get() { return memory_; }

 private:
  hbSysMem memory_{};
};

class TaskHandle {
 public:
  ~TaskHandle() {
    if (handle != nullptr) {
      hbDNNReleaseTask(handle);
    }
  }

  TaskHandle(const TaskHandle&) = delete;
  TaskHandle& operator=(const TaskHandle&) = delete;
  TaskHandle() = default;

  hbDNNTaskHandle_t handle = nullptr;
};

std::vector<int> ShapeVector(const hbDNNTensorShape& shape) {
  if (shape.numDimensions <= 0 ||
      shape.numDimensions > HB_DNN_TENSOR_MAX_DIMENSIONS) {
    throw std::runtime_error("invalid tensor dimension count: " +
                             std::to_string(shape.numDimensions));
  }
  std::vector<int> result;
  result.reserve(static_cast<std::size_t>(shape.numDimensions));
  for (int index = 0; index < shape.numDimensions; ++index) {
    const int dimension = shape.dimensionSize[index];
    if (dimension <= 0) {
      throw std::runtime_error("tensor shape contains a non-positive dimension");
    }
    result.push_back(dimension);
  }
  return result;
}

std::int64_t ElementCount(const hbDNNTensorShape& shape) {
  const std::vector<int> dimensions = ShapeVector(shape);
  std::int64_t count = 1;
  for (const int dimension : dimensions) {
    if (count > std::numeric_limits<std::int64_t>::max() / dimension) {
      throw std::overflow_error("tensor element count overflow");
    }
    count *= dimension;
  }
  return count;
}

TensorMetadata MakeMetadata(const std::string& name,
                            const hbDNNTensorProperties& properties) {
  TensorMetadata metadata;
  metadata.name = name;
  metadata.valid_shape = ShapeVector(properties.validShape);
  metadata.aligned_shape = ShapeVector(properties.alignedShape);
  metadata.tensor_layout = properties.tensorLayout;
  metadata.tensor_type = properties.tensorType;
  metadata.quanti_type = static_cast<int>(properties.quantiType);
  metadata.aligned_byte_size = properties.alignedByteSize;
  return metadata;
}

void ValidateTensor(const std::string& role, const std::string& actual_name,
                    const std::string& expected_name,
                    const hbDNNTensorProperties& properties,
                    int expected_elements) {
  if (actual_name != expected_name) {
    throw std::runtime_error("expected " + role + " tensor name '" +
                             expected_name + "', got '" + actual_name + "'");
  }
  if (properties.tensorType != HB_DNN_TENSOR_TYPE_F32) {
    throw std::runtime_error(role + " tensor must be float32");
  }
  if (properties.quantiType != NONE) {
    throw std::runtime_error(role + " tensor must have quantiType NONE");
  }
  if (properties.validShape.numDimensions != 4 ||
      properties.alignedShape.numDimensions != 4) {
    throw std::runtime_error(role + " tensor must expose a four-dimensional "
                             "X5 Runtime shape");
  }
  if (properties.validShape.dimensionSize[0] != 1 ||
      ElementCount(properties.validShape) != expected_elements) {
    throw std::runtime_error(role + " tensor does not match the HIMLoco batch-one "
                             "element contract");
  }
  if (properties.alignedByteSize <= 0 ||
      properties.alignedByteSize % static_cast<int>(sizeof(float)) != 0) {
    throw std::runtime_error(role + " tensor has an invalid alignedByteSize");
  }
}

std::vector<float> CopyValidOutput(const hbDNNTensor& tensor,
                                   int expected_elements) {
  const hbDNNTensorProperties& properties = tensor.properties;
  const std::vector<int> valid = ShapeVector(properties.validShape);
  const std::vector<int> aligned = ShapeVector(properties.alignedShape);
  if (valid.size() != aligned.size()) {
    throw std::runtime_error("output valid/aligned rank mismatch");
  }
  for (std::size_t index = 0; index < valid.size(); ++index) {
    if (aligned[index] < valid[index]) {
      throw std::runtime_error("output aligned shape is smaller than valid shape");
    }
  }

  const float* source = static_cast<const float*>(tensor.sysMem[0].virAddr);
  const std::size_t capacity =
      static_cast<std::size_t>(properties.alignedByteSize) / sizeof(float);
  std::vector<float> result(static_cast<std::size_t>(expected_elements));
  for (int logical_index = 0; logical_index < expected_elements; ++logical_index) {
    std::size_t remaining = static_cast<std::size_t>(logical_index);
    std::size_t aligned_offset = 0;
    std::size_t aligned_stride = 1;
    for (std::size_t reverse = valid.size(); reverse > 0; --reverse) {
      const std::size_t dimension = reverse - 1;
      const std::size_t coordinate =
          remaining % static_cast<std::size_t>(valid[dimension]);
      remaining /= static_cast<std::size_t>(valid[dimension]);
      aligned_offset += coordinate * aligned_stride;
      aligned_stride *= static_cast<std::size_t>(aligned[dimension]);
    }
    if (aligned_offset >= capacity) {
      throw std::runtime_error("output aligned offset exceeds allocated memory");
    }
    result[static_cast<std::size_t>(logical_index)] = source[aligned_offset];
  }
  return result;
}

}  // namespace

class HimLoco::Impl {
 public:
  Impl(const std::string& model_path, int requested_priority)
      : priority_(requested_priority) {
    if (priority_ < -1 || priority_ > 255) {
      throw std::invalid_argument("priority must be -1 or in [0,255]");
    }

    const char* model_file = model_path.c_str();
    Check(hbDNNInitializeFromFiles(&packed_handle_, &model_file, 1),
          "hbDNNInitializeFromFiles");
    try {
      InitializeModel();
      AllocateTensors();
    } catch (...) {
      output_memory_.reset();
      input_memory_.reset();
      hbDNNRelease(packed_handle_);
      packed_handle_ = nullptr;
      throw;
    }
  }

  ~Impl() {
    output_memory_.reset();
    input_memory_.reset();
    if (packed_handle_ != nullptr) {
      hbDNNRelease(packed_handle_);
    }
  }

  InferenceResult Infer(const std::vector<float>& observation) {
    if (observation.size() != static_cast<std::size_t>(kInputElements)) {
      throw std::invalid_argument("observation must contain exactly 270 values");
    }
    if (!std::all_of(observation.begin(), observation.end(),
                     [](float value) { return std::isfinite(value); })) {
      throw std::invalid_argument("observation contains NaN/Inf");
    }

    std::memset(input_tensor_.sysMem[0].virAddr, 0,
                static_cast<std::size_t>(input_properties_.alignedByteSize));
    std::memcpy(input_tensor_.sysMem[0].virAddr, observation.data(),
                observation.size() * sizeof(float));
    Check(hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
          "hbSysFlushMem(input, CLEAN)");

    TaskHandle task;
    hbDNNInferCtrlParam control;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
    if (priority_ >= 0) {
      control.priority = priority_;
    }
    hbDNNTensor* output_pointer = &output_tensor_;
    const auto started = std::chrono::steady_clock::now();
    const int infer_code = hbDNNInfer(&task.handle, &output_pointer, &input_tensor_,
                                      model_handle_, &control);
    if (infer_code == 0) {
      Check(hbDNNWaitTaskDone(task.handle, 0), "hbDNNWaitTaskDone");
    }
    const auto stopped = std::chrono::steady_clock::now();
    Check(infer_code, "hbDNNInfer");
    Check(hbSysFlushMem(&output_tensor_.sysMem[0],
                        HB_SYS_MEM_CACHE_INVALIDATE),
          "hbSysFlushMem(output, INVALIDATE)");

    InferenceResult result;
    result.latency_ms =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    result.actions = CopyValidOutput(output_tensor_, kOutputElements);
    if (!std::all_of(result.actions.begin(), result.actions.end(),
                     [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("Runtime output contains NaN/Inf");
    }
    return result;
  }

  const std::string& model_name() const { return model_name_; }
  const std::string& runtime_version() const { return runtime_version_; }
  const TensorMetadata& input_metadata() const { return input_metadata_; }
  const TensorMetadata& output_metadata() const { return output_metadata_; }
  int priority() const { return priority_; }

 private:
  void InitializeModel() {
    const char** names = nullptr;
    int model_count = 0;
    Check(hbDNNGetModelNameList(&names, &model_count, packed_handle_),
          "hbDNNGetModelNameList");
    if (model_count != 1 || names == nullptr || names[0] == nullptr) {
      throw std::runtime_error("expected exactly one packed model");
    }
    model_name_ = names[0];
    Check(hbDNNGetModelHandle(&model_handle_, packed_handle_, names[0]),
          "hbDNNGetModelHandle");

    int input_count = 0;
    int output_count = 0;
    Check(hbDNNGetInputCount(&input_count, model_handle_),
          "hbDNNGetInputCount");
    Check(hbDNNGetOutputCount(&output_count, model_handle_),
          "hbDNNGetOutputCount");
    if (input_count != 1 || output_count != 1) {
      throw std::runtime_error("expected exactly one input and one output");
    }

    const char* input_name = nullptr;
    const char* output_name = nullptr;
    Check(hbDNNGetInputName(&input_name, model_handle_, 0),
          "hbDNNGetInputName");
    Check(hbDNNGetOutputName(&output_name, model_handle_, 0),
          "hbDNNGetOutputName");
    if (input_name == nullptr || output_name == nullptr) {
      throw std::runtime_error("Runtime returned a null tensor name");
    }
    Check(hbDNNGetInputTensorProperties(&input_properties_, model_handle_, 0),
          "hbDNNGetInputTensorProperties");
    Check(hbDNNGetOutputTensorProperties(&output_properties_, model_handle_, 0),
          "hbDNNGetOutputTensorProperties");
    ValidateTensor("input", input_name, "obs_history", input_properties_,
                   kInputElements);
    ValidateTensor("output", output_name, "actions", output_properties_,
                   kOutputElements);
    input_metadata_ = MakeMetadata(input_name, input_properties_);
    output_metadata_ = MakeMetadata(output_name, output_properties_);
    const char* version = hbDNNGetVersion();
    runtime_version_ = version == nullptr ? "unreported" : version;
  }

  void AllocateTensors() {
    input_memory_ =
        std::make_unique<CachedMemory>(input_properties_.alignedByteSize);
    output_memory_ =
        std::make_unique<CachedMemory>(output_properties_.alignedByteSize);

    input_tensor_ = {};
    input_tensor_.properties = input_properties_;
    input_tensor_.properties.alignedShape = input_properties_.validShape;
    input_tensor_.sysMem[0] = input_memory_->get();

    output_tensor_ = {};
    output_tensor_.properties = output_properties_;
    output_tensor_.sysMem[0] = output_memory_->get();
  }

  hbPackedDNNHandle_t packed_handle_ = nullptr;
  hbDNNHandle_t model_handle_ = nullptr;
  hbDNNTensorProperties input_properties_{};
  hbDNNTensorProperties output_properties_{};
  hbDNNTensor input_tensor_{};
  hbDNNTensor output_tensor_{};
  std::unique_ptr<CachedMemory> input_memory_;
  std::unique_ptr<CachedMemory> output_memory_;
  std::string model_name_;
  std::string runtime_version_;
  TensorMetadata input_metadata_;
  TensorMetadata output_metadata_;
  int priority_ = -1;
};

HimLoco::HimLoco(const std::string& model_path, int priority)
    : impl_(std::make_unique<Impl>(model_path, priority)) {}

HimLoco::~HimLoco() = default;
HimLoco::HimLoco(HimLoco&&) noexcept = default;
HimLoco& HimLoco::operator=(HimLoco&&) noexcept = default;

InferenceResult HimLoco::Infer(const std::vector<float>& observation) {
  return impl_->Infer(observation);
}

const std::string& HimLoco::model_name() const { return impl_->model_name(); }
const std::string& HimLoco::runtime_version() const {
  return impl_->runtime_version();
}
const TensorMetadata& HimLoco::input_metadata() const {
  return impl_->input_metadata();
}
const TensorMetadata& HimLoco::output_metadata() const {
  return impl_->output_metadata();
}
int HimLoco::priority() const { return impl_->priority(); }

}  // namespace himloco
