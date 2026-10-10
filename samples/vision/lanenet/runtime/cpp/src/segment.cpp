// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// LaneNet S100 model: tensor contract, runtime ownership and stage math.
// The DNN/UCP handles and buffers live in the private Impl in this file; the
// public header exposes only owned stage-data types.
#include "segment.hpp"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <utility>
namespace lanenet {
namespace {
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("Tensor size overflow");
  return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::invalid_argument("Tensor extent overflow");
  return a + b;
}
std::size_t offset(std::size_t index, const TensorSpec &spec) {
  std::size_t out = 0;
  for (std::size_t i = spec.shape.size(); i-- > 0;) {
    out += index % spec.shape[i] * spec.strides[i];
    index /= spec.shape[i];
  }
  return out; // layout validation proves each term and the sum fit capacity
}
std::string normalize(std::string value) {
  const auto start = value.find_first_not_of(" \t\r\n");
  if (start == std::string::npos)
    return {};
  value = value.substr(start, value.find_last_not_of(" \t\r\n") - start + 1);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
std::string read_identity(const char *path) {
  std::ifstream in(path);
  return std::string(std::istreambuf_iterator<char>(in), {});
}
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
  TaskGuard() = default;
  hbUCPTaskHandle_t handle = nullptr;
  ~TaskGuard() {
    if (handle)
      hbUCPReleaseTask(handle);
  }
  TaskGuard(const TaskGuard &) = delete;
  TaskGuard &operator=(const TaskGuard &) = delete;
};
} // namespace
std::size_t element_bytes(ScalarType type) {
  switch (type) {
  case ScalarType::Float32:
  case ScalarType::Int32:
    return 4;
  case ScalarType::Int64:
    return 8;
  case ScalarType::Int16:
    return 2;
  case ScalarType::Int8:
  case ScalarType::UInt8:
    return 1;
  }
  throw std::invalid_argument("Unknown tensor scalar type");
}
std::size_t validate_layout(const TensorSpec &spec) {
  const auto item = element_bytes(spec.type);
  const auto rank = spec.shape.size();
  if (rank == 0 || rank > 8 || spec.strides.size() != rank ||
      spec.capacity == 0)
    throw std::invalid_argument("Invalid tensor rank/strides/capacity");
  std::size_t span = item, count = 1;
  for (std::size_t i = rank; i-- > 0;) {
    const auto stride = spec.strides[i], dim = spec.shape[i];
    if (dim == 0 || stride == 0 || stride % item || stride < span ||
        stride > spec.capacity)
      throw std::invalid_argument("Overlapping or misaligned tensor strides");
    span = add(span, mul(dim - 1, stride));
    count = mul(count, dim);
  }
  if (span > spec.capacity)
    throw std::invalid_argument("Tensor exceeds buffer capacity");
  return count;
}
OutputRoles bind_roles(const std::vector<TensorSpec> &outputs) {
  const auto none = std::numeric_limits<std::size_t>::max();
  OutputRoles roles{none, none};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto &s = outputs[i];
    validate_layout(s);
    if (s.type == ScalarType::Float32 &&
        s.shape == std::vector<std::size_t>{1, 3, 256, 512}) {
      if (roles.embedding != none)
        throw std::invalid_argument("Ambiguous embedding output role");
      roles.embedding = i;
    }
    if (s.type == ScalarType::Int64 &&
        (s.shape == std::vector<std::size_t>{1, 1, 256, 512} ||
         s.shape == std::vector<std::size_t>{1, 256, 512})) {
      if (roles.binary != none)
        throw std::invalid_argument("Ambiguous binary output role");
      roles.binary = i;
    }
  }
  if (roles.embedding == none || roles.binary == none)
    throw std::invalid_argument("Missing embedding/binary output role");
  return roles;
}
std::vector<unsigned char> compact_bytes(const RawTensor &raw) {
  const auto count = validate_layout(raw.spec),
             item = element_bytes(raw.spec.type);
  if (raw.bytes.size() < raw.spec.capacity)
    throw std::invalid_argument(
        "Owned output is smaller than bound allocation");
  std::vector<unsigned char> result(mul(count, item));
  for (std::size_t index = 0; index < count; ++index)
    std::memcpy(result.data() + index * item,
                raw.bytes.data() + offset(index, raw.spec), item);
  return result;
}
void write_input(const std::vector<float> &input, const TensorSpec &spec,
                 void *destination) {
  if (spec.type != ScalarType::Float32 ||
      spec.shape != std::vector<std::size_t>{1, 3, 256, 512})
    throw std::invalid_argument("Expected float32 NCHW image input");
  const auto count = validate_layout(spec);
  if (!destination || input.size() != count ||
      !std::all_of(input.begin(), input.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument("Invalid prepared input values");
  std::memset(destination, 0, spec.capacity);
  auto *base = static_cast<unsigned char *>(destination);
  for (std::size_t i = 0; i < count; ++i)
    std::memcpy(base + offset(i, spec), &input[i], sizeof(float));
}
LaneResult decode_outputs(const std::vector<RawTensor> &raw) {
  std::vector<TensorSpec> specs;
  for (const auto &value : raw)
    specs.push_back(value.spec);
  const auto roles = bind_roles(specs);
  // Validate every owned allocation, retaining auxiliaries without assigning
  // semantics.
  for (const auto &value : raw)
    if (value.bytes.size() < value.spec.capacity)
      throw std::invalid_argument("Truncated auxiliary/output allocation");
  auto embedding = compact_bytes(raw[roles.embedding]),
       binary = compact_bytes(raw[roles.binary]);
  LaneResult result;
  result.embedding.resize(3 * 256 * 512);
  result.binary.resize(256 * 512);
  for (std::size_t i = 0; i < result.embedding.size(); ++i) {
    float value;
    std::memcpy(&value, embedding.data() + i * 4, 4);
    if (!std::isfinite(value))
      throw std::invalid_argument("Nonfinite embedding");
    result.embedding[i] = value;
  }
  for (std::size_t i = 0; i < result.binary.size(); ++i) {
    std::int64_t value;
    std::memcpy(&value, binary.data() + i * 8, 8);
    if (value != 0 && value != 1)
      throw std::invalid_argument("Binary labels must be 0/1");
    result.binary[i] = static_cast<std::uint8_t>(value);
  }
  return result;
}
std::uint8_t display_component(float value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("Nonfinite embedding display value");
  const float scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f;
  const auto base = static_cast<unsigned int>(std::floor(scaled));
  const float fraction = scaled - base;
  return static_cast<std::uint8_t>(
      base + (fraction > .5f || (fraction == .5f && (base % 2))));
}
bool is_s100(std::string soc, std::string board) {
  soc = normalize(soc);
  board = normalize(board);
  return soc == "s100" && board != "s100p" && board != "rdk s100p";
}
void require_s100_board() {
  if (!is_s100(read_identity("/sys/class/boardinfo/soc_name"),
               read_identity("/sys/class/boardinfo/board_type")))
    throw std::invalid_argument(
        "LaneNet requires actual S100 identity; no S100P fallback");
}
// ===========================================================================
// LaneNet model
// ===========================================================================
struct LaneNet::Impl {
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
LaneNet::LaneNet(const std::string &path, BoardGate gate)
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
LaneNet::~LaneNet() = default;
std::vector<float> LaneNet::preprocess(const cv::Mat &image) const {
  if (image.empty() || image.type() != CV_8UC3)
    throw std::invalid_argument("Expected nonempty BGR uint8 image");
  cv::Mat rgb, resized;
  cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
  cv::resize(rgb, resized, cv::Size(512, 256), 0, 0, cv::INTER_AREA);
  const float mean[] = {.485f, .456f, .406f}, stddev[] = {.229f, .224f, .225f};
  std::vector<float> result(3 * 256 * 512);
  for (int h = 0; h < 256; h++)
    for (int w = 0; w < 512; w++)
      for (int c = 0; c < 3; c++)
        result[c * 256 * 512 + h * 512 + w] =
            (resized.at<cv::Vec3b>(h, w)[c] / 255.0f - mean[c]) / stddev[c];
  return result;
}
std::vector<RawTensor> LaneNet::infer(const std::vector<float> &prepared) {
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
  // Release the finished task and report a failed release; the guard covers
  // every exceptional path above and never releases twice.
  const int rc = hbUCPReleaseTask(task.handle);
  task.handle = nullptr;
  checked(rc, "task release");
  return result;
}
LaneResult LaneNet::postprocess(const std::vector<RawTensor> &raw) const {
  LaneResult result = decode_outputs(raw);
  // The result owns copies of the raw outputs so reporting does not depend
  // on buffers or raw values from a later call.
  result.raw_outputs = raw;
  return result;
}
LaneResult LaneNet::predict(const cv::Mat &image) {
  return postprocess(infer(preprocess(image)));
}
const std::string &LaneNet::model_name() const { return impl_->name; }
const TensorSpec &LaneNet::input_spec() const { return impl_->input_spec; }
const std::vector<TensorSpec> &LaneNet::output_specs() const {
  return impl_->output_specs;
}
} // namespace lanenet
