// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"
#include "yolov5_decode.hpp"
#include "yolov5_dump.hpp"
#include "yolov5_gate.hpp"
#include "yolov5_visualize.hpp"

#include <dnn/hb_dnn.h>
#include <dnn/hb_dnn_ext.h>
#include <hobot/hb_ucp.h>
#include "postprocess.hpp"
#include "preprocess.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef YOLOV5_TARGET_NAME
#define YOLOV5_TARGET_NAME "unknown"
#endif

namespace yolov5 {

namespace {
constexpr int kInputSize = 672;
constexpr int kClasses = 80;
constexpr std::array<float, 18> kAnchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

int tensor_dtype_code(int32_t type) {
  if (type == HB_DNN_TENSOR_TYPE_F32) return kDtypeF32;
  if (type == HB_DNN_TENSOR_TYPE_S32) return kDtypeS32;
  if (type == HB_DNN_TENSOR_TYPE_S8) return kDtypeS8;
  if (type == HB_DNN_TENSOR_TYPE_U8) return kDtypeU8;
  if (type == HB_DNN_TENSOR_TYPE_S16) return kDtypeS16;
  return kDtypeUnknown;
}

int quanti_code(int32_t type) {
  if (type == NONE) return kQuantiNone;
  if (type == SCALE) return kQuantiScale;
  if (type == SHIFT) return kQuantiShift;
  return kQuantiUnknown;
}

TensorMeta project(const hbDNNTensorProperties& properties) {
  TensorMeta meta;
  meta.dtype = tensor_dtype_code(properties.tensorType);
  meta.quanti_type = quanti_code(properties.quantiType);
  meta.num_dimensions = properties.validShape.numDimensions;
  for (int i = 0; i < 4 && i < properties.validShape.numDimensions; ++i)
    meta.valid[i] = properties.validShape.dimensionSize[i];
  for (int i = 0; i < 4 && i < properties.alignedShape.numDimensions; ++i)
    meta.aligned[i] = properties.alignedShape.dimensionSize[i];
  meta.aligned_byte_size = properties.alignedByteSize;
  // Outputs are allocated at alignedByteSize; input planes are allocated by
  // prepare_input_tensor as stride[0] * batch, which the caller overrides below.
  meta.storage_bytes = properties.alignedByteSize;
  for (int i = 0; i < 4; ++i) meta.stride[i] = properties.stride[i];
  if (properties.quantiType == SCALE) {
    meta.scale_len = properties.scale.scaleLen;
    meta.zero_point_len = properties.scale.zeroPointLen;
  }
  return meta;
}

std::string dtype_name(int code) {
  switch (code) {
    case kDtypeF32: return "float32";
    case kDtypeS32: return "int32";
    case kDtypeS8: return "int8";
    case kDtypeU8: return "uint8";
    case kDtypeS16: return "int16";
    default: return "unknown";
  }
}

std::string quanti_name(int code) {
  switch (code) {
    case kQuantiNone: return "none";
    case kQuantiScale: return "scale";
    case kQuantiShift: return "shift";
    default: return "unknown";
  }
}

std::vector<long long> shape_of(const TensorMeta& meta) {
  std::vector<long long> shape;
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i) shape.push_back(meta.valid[i]);
  return shape;
}

void require_gate(const Gate& gate) {
  if (!gate) throw std::runtime_error(gate.reason);
}

}  // namespace

// Frees only tensors that were actually allocated: a partially failed
// allocation must not turn into a blind free of a zeroed sysMem.
struct SResourceGuard {
  hbPackedDNNHandle_t packed = nullptr;
  hbUCPTaskHandle_t task = nullptr;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  ~SResourceGuard() {
    if (task) hbUCPReleaseTask(task);
    for (auto& tensor : inputs)
      if (tensor.sysMem.virAddr != nullptr) hbUCPFree(&tensor.sysMem);
    for (auto& tensor : outputs)
      if (tensor.sysMem.virAddr != nullptr) hbUCPFree(&tensor.sysMem);
    if (packed) hbDNNRelease(packed);
  }
};

int run_native(const RuntimeOptions& options) {
  require_gate(check_target_matches_build(options.target, YOLOV5_TARGET_NAME));
  if (options.target != "s100" && options.target != "s600")
    throw std::invalid_argument("S adapter supports s100 and s600 YOLOv5 assets only");

  hbPackedDNNHandle_t packed = nullptr;
  const char* file = options.model_path.c_str();
  if (hbDNNInitializeFromFiles(&packed, &file, 1) != 0)
    throw std::runtime_error("hbDNNInitializeFromFiles failed");
  SResourceGuard guard;
  guard.packed = packed;
  const char** model_names = nullptr;
  int model_count = 0;
  if (hbDNNGetModelNameList(&model_names, &model_count, packed) != 0 || model_count != 1)
    throw std::runtime_error("S YOLOv5 asset must contain exactly one model");
  hbDNNHandle_t model = nullptr;
  if (hbDNNGetModelHandle(&model, packed, model_names[0]) != 0)
    throw std::runtime_error("hbDNNGetModelHandle failed");
  int32_t input_count = 0;
  int32_t output_count = 0;
  if (hbDNNGetInputCount(&input_count, model) != 0 || input_count != 2 ||
      hbDNNGetOutputCount(&output_count, model) != 0 || output_count != 3)
    throw std::runtime_error("S YOLOv5 requires two NV12 inputs and three outputs");

  guard.inputs.resize(static_cast<std::size_t>(input_count));
  guard.outputs.resize(static_cast<std::size_t>(output_count));
  auto& inputs = guard.inputs;
  auto& outputs = guard.outputs;
  for (auto& input : inputs) std::memset(&input, 0, sizeof(input));
  for (auto& output : outputs) std::memset(&output, 0, sizeof(output));
  for (int i = 0; i < input_count; ++i)
    if (hbDNNGetInputTensorProperties(&inputs[static_cast<std::size_t>(i)].properties, model, i) != 0)
      throw std::runtime_error("S input metadata query failed");

  const auto& y_shape = inputs[0].properties.validShape;
  const auto& uv_shape = inputs[1].properties.validShape;
  if (y_shape.numDimensions != 4 || uv_shape.numDimensions != 4 ||
      y_shape.dimensionSize[0] != 1 || y_shape.dimensionSize[1] != kInputSize ||
      y_shape.dimensionSize[2] != kInputSize || y_shape.dimensionSize[3] != 1 ||
      uv_shape.dimensionSize[0] != 1 || uv_shape.dimensionSize[1] != kInputSize / 2 ||
      uv_shape.dimensionSize[2] != kInputSize / 2 || uv_shape.dimensionSize[3] != 2)
    throw std::runtime_error("S YOLOv5 inputs must be Y[1,672,672,1] and UV[1,336,336,2]");

  std::vector<HeadShape> heads;
  std::vector<TensorMeta> output_meta;
  for (int i = 0; i < output_count; ++i) {
    if (hbDNNGetOutputTensorProperties(&outputs[static_cast<std::size_t>(i)].properties, model, i) != 0)
      throw std::runtime_error("S output metadata query failed");
    const TensorMeta meta = project(outputs[static_cast<std::size_t>(i)].properties);
    if (meta.num_dimensions != 4 || meta.valid[0] != 1)
      throw std::runtime_error("S output must be rank-4 NHWC with batch 1");
    heads.push_back({static_cast<int>(meta.valid[1]), static_cast<int>(meta.valid[2]),
                     static_cast<int>(meta.valid[3])});
    output_meta.push_back(meta);
  }
  if (!validate_head_shapes(heads, kInputSize, kClasses))
    throw std::runtime_error("S output heads must be unique 84/42/21 metadata heads");

  if (prepare_input_tensor(inputs) != 0 || prepare_output_tensor(outputs) != 0)
    throw std::runtime_error("S tensor allocation failed");

  // Allocations exist now, so the stride/capacity assumptions of the writer and
  // of dequantizeTensorS32 are checked against what the runtime actually gave us.
  TensorMeta y_gate = project(inputs[0].properties);
  y_gate.storage_bytes = y_gate.stride[0] * y_gate.valid[0];
  require_gate(check_s_nv12_plane(y_gate, kInputSize, kInputSize, 1));
  TensorMeta uv_gate = project(inputs[1].properties);
  uv_gate.storage_bytes = uv_gate.stride[0] * uv_gate.valid[0];
  require_gate(check_s_nv12_plane(uv_gate, kInputSize / 2, kInputSize / 2, 2));
  std::vector<long long> output_counts;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    long long count = 0;
    const Gate gate = check_s32_dequant(project(outputs[i].properties), &count);
    if (!gate) throw std::runtime_error("S output " + std::to_string(i) + ": " + gate.reason);
    output_counts.push_back(count);
  }

  cv::Mat image = cv::imread(options.image_path);
  if (image.empty()) throw std::invalid_argument("S image is empty");
  cv::Mat boxed(kInputSize, kInputSize, CV_8UC3);
  letterbox_resize(image, boxed, 127);
  if (bgr_to_nv12_tensor(boxed, inputs, kInputSize, kInputSize) != 0)
    throw std::runtime_error("S NV12 preprocessing failed");

  hbUCPTaskHandle_t task = nullptr;
  if (hbDNNInferV2(&task, outputs.data(), inputs.data(), model) != 0)
    throw std::runtime_error("hbDNNInferV2 failed");
  guard.task = task;
  hbUCPSchedParam schedule{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
  // Declared difference from the fixed S source, which hard-codes priority 0 and
  // HB_UCP_BPU_CORE_ANY: the unified CLI exposes scheduling, so silently
  // dropping the caller's value would make the documented parameters lie.
  schedule.backend = options.bpu_core < 0 ? HB_UCP_BPU_CORE_ANY : options.bpu_core;
  schedule.priority = options.priority;
  if (hbUCPSubmitTask(task, &schedule) != 0 || hbUCPWaitTaskDone(task, 0) != 0)
    throw std::runtime_error("S UCP task failed");

  for (auto& output : outputs) hbUCPMemFlush(&output.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);

  DumpRecord dump;
  dump.dir = options.dump_dir;
  dump.utc = utc_timestamp();
  dump.target = options.target;
  dump.build_target = YOLOV5_TARGET_NAME;
  dump.asset_id = options.asset_id;
  dump.model_path = options.model_path;
  dump.image_path = options.image_path;
  dump.argv = options.argv;
  dump.cwd = std::filesystem::current_path().string();
  dump.options = {{"score_thres", std::to_string(options.score_threshold)},
                  {"nms_thres", std::to_string(options.nms_threshold)},
                  {"priority", std::to_string(options.priority)},
                  {"bpu_core", std::to_string(options.bpu_core)},
                  {"preprocess", "letterbox"},
                  {"nms_top_k_per_class", "-1"},
                  {"score_boundary", "greater-or-equal"},
                  {"label_file", options.label_path}};
  dump.notes = {"S UCP adapter; split NV12 672x672 Y/UV inputs; per-channel S32 dequantization.",
                "NMS preserves source S nms_bboxes semantics: keep score >= threshold, no "
                "per-class top_k.",
                "Scheduling honours the caller's priority/BPU core (source S forces priority 0)."};

  const TensorMeta y_meta = project(inputs[0].properties);
  const TensorMeta uv_meta = project(inputs[1].properties);
  dump.inputs.push_back({"y", dtype_name(y_meta.dtype), shape_of(y_meta),
                         quanti_name(y_meta.quanti_type), y_meta.scale_len});
  dump.inputs.push_back({"uv", dtype_name(uv_meta.dtype), shape_of(uv_meta),
                         quanti_name(uv_meta.quanti_type), uv_meta.scale_len});

  std::vector<std::vector<float>> dequantized;
  dequantized.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& meta = output_meta[i];
    dump.outputs.push_back({"output" + std::to_string(i), dtype_name(meta.dtype),
                            shape_of(meta), quanti_name(meta.quanti_type), meta.scale_len});
    const std::size_t bytes = static_cast<std::size_t>(output_counts[i]) * sizeof(int32_t);
    std::vector<unsigned char> raw_bytes(bytes);
    if (bytes > 0) std::memcpy(raw_bytes.data(), outputs[i].sysMem.virAddr, bytes);
    dump.raw_tensors.push_back({"output" + std::to_string(i), dtype_name(meta.dtype),
                                shape_of(meta), std::move(raw_bytes)});
    // Source S32/per-channel helper; the gate above proved the descriptor and
    // layout it relies on, so no unknown layout is ever read.
    dequantized.push_back(dequantizeTensorS32(outputs[i]));
    const auto& values = dequantized.back();
    std::vector<unsigned char> float_bytes(values.size() * sizeof(float));
    if (!values.empty()) std::memcpy(float_bytes.data(), values.data(), float_bytes.size());
    dump.transformed_tensors.push_back({"output" + std::to_string(i), "float32",
                                        shape_of(meta), std::move(float_bytes)});
  }

  DecodePolicy policy;
  policy.score_threshold = options.score_threshold;
  policy.nms_threshold = options.nms_threshold;
  policy.top_k_per_class = -1;
  policy.strict_score_boundary = false;
  const auto detections = decode_heads(dequantized, heads, kInputSize, kClasses, policy, kAnchors);
  dump.detections = detections;
  if (!options.dump_dir.empty()) {
    std::string error;
    if (!write_dump(dump, &error))
      throw std::runtime_error("cannot write YOLOv5 S dump: " + error);
  }
  render_detections(image, detections, kInputSize, true, options.output_path,
                    load_labels(options.label_path));
  return 0;
}

}  // namespace yolov5
