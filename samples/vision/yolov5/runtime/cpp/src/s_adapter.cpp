// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"
#include "yolov5_decode.hpp"
#include "yolov5_dump.hpp"
#include "yolov5_gate.hpp"
#include "yolov5_s_native.hpp"
#include "yolov5_visualize.hpp"

// The S UCP SDK installs its DNN headers under /usr/include/hobot (verified on a
// real S100 2026-09-24): hb_dnn.h lives at hobot/dnn/hb_dnn.h and there is no
// hb_dnn_ext.h. The c_utils headers below expect these to be included first.
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
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
  // Enum names confirmed present in the real S100 hb_dnn.h (on-board evidence
  // 2026-09-24): S8/U8/S16/F32/S32 among others. Values outside the known set
  // stay unknown so the gates reject them.
  if (type == HB_DNN_TENSOR_TYPE_F32) return kDtypeF32;
  if (type == HB_DNN_TENSOR_TYPE_S32) return kDtypeS32;
  if (type == HB_DNN_TENSOR_TYPE_S8) return kDtypeS8;
  if (type == HB_DNN_TENSOR_TYPE_U8) return kDtypeU8;
  if (type == HB_DNN_TENSOR_TYPE_S16) return kDtypeS16;
  return kDtypeUnknown;
}

int quanti_code(int32_t type) {
  // The real S100 hbDNNQuantiType enum only defines NONE and SCALE (verified
  // on-board); any other numeric value stays unknown so the gate rejects it
  // instead of silently misinterpreting it.
  if (type == NONE) return kQuantiNone;
  if (type == SCALE) return kQuantiScale;
  return kQuantiUnknown;
}

TensorMeta project(const hbDNNTensorProperties& properties) {
  TensorMeta meta;
  meta.dtype = tensor_dtype_code(properties.tensorType);
  meta.quanti_type = quanti_code(properties.quantiType);
  meta.num_dimensions = properties.validShape.numDimensions;
  for (int i = 0; i < 4 && i < properties.validShape.numDimensions; ++i)
    meta.valid[i] = properties.validShape.dimensionSize[i];
  // The real S100 hbDNNTensorProperties has no alignedShape field (verified
  // on-board): the stored layout is described by stride[] plus alignedByteSize,
  // which is exactly what the S gates validate, so aligned[] stays unreported.
  meta.aligned_byte_size = properties.alignedByteSize;
  // Outputs are allocated at alignedByteSize; input planes are allocated by
  // prepare_input_tensor as stride[0] * batch, which the caller overrides below.
  meta.storage_bytes = properties.alignedByteSize;
  for (int i = 0; i < 4; ++i) meta.stride[i] = properties.stride[i];
  meta.quantize_axis = properties.quantizeAxis;
  if (properties.quantiType == SCALE) {
    meta.scale_len = properties.scale.scaleLen;
    meta.zero_point_len = properties.scale.zeroPointLen;
  }
  return meta;
}

std::vector<long long> shape_of(const TensorMeta& meta) {
  std::vector<long long> shape;
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i) shape.push_back(meta.valid[i]);
  return shape;
}

DumpTensor gather_plane_rows(const hbDNNTensor& tensor, const TensorMeta& meta,
                             const std::string& name) {
  // Deterministic payload of one NV12 plane: the writer fills each row of
  // valid[2]*valid[3] bytes at a stride[1] pitch, and the padding between rows
  // is uninitialized memory that is deliberately not copied into the evidence.
  DumpTensor payload;
  payload.name = name;
  payload.dtype = dtype_name(meta.dtype);
  payload.shape = shape_of(meta);
  const unsigned char* base = static_cast<const unsigned char*>(tensor.sysMem.virAddr);
  const long long rows = meta.valid[1];
  const long long row_bytes = meta.valid[2] * meta.valid[3];
  payload.bytes.reserve(static_cast<std::size_t>(rows * row_bytes));
  for (long long row = 0; row < rows; ++row) {
    const unsigned char* begin = base + static_cast<std::size_t>(row * meta.stride[1]);
    payload.bytes.insert(payload.bytes.end(), begin, begin + static_cast<std::size_t>(row_bytes));
  }
  return payload;
}

void require_gate(const Gate& gate) {
  if (!gate) throw std::runtime_error(gate.reason);
}

}  // namespace

// Frees only tensors that were actually allocated: a partially failed
// allocation must not turn into a blind free of a zeroed sysMem.
struct SResourceGuard {
  // The real S100 SDK spells the packed handle hbDNNPackedHandle_t (on-board
  // compile evidence 2026-09-24); hbPackedDNNHandle_t is the X5 spelling.
  hbDNNPackedHandle_t packed = nullptr;
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
  unsigned long long backend = 0;
  if (!bpu_core_to_backend(options.bpu_core, &backend))
    throw std::invalid_argument("--bpu-core must be -1 (any) or a core index 0..3");

  hbDNNPackedHandle_t packed = nullptr;
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

  // Allocations exist now, so the stride/capacity assumptions of the NV12
  // writer and of the dequantizer are checked against what the runtime
  // actually reported.
  TensorMeta y_gate = project(inputs[0].properties);
  y_gate.storage_bytes = y_gate.stride[0] * y_gate.valid[0];
  require_gate(check_s_nv12_plane(y_gate, kInputSize, kInputSize, 1));
  TensorMeta uv_gate = project(inputs[1].properties);
  uv_gate.storage_bytes = uv_gate.stride[0] * uv_gate.valid[0];
  require_gate(check_s_nv12_plane(uv_gate, kInputSize / 2, kInputSize / 2, 2));
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    long long count = 0;
    const Gate gate = check_s32_dequant(project(outputs[i].properties), &count);
    if (!gate) throw std::runtime_error("S output " + std::to_string(i) + ": " + gate.reason);
    // The dequantized element count must match what the source helper will
    // produce for the same valid shape, padding notwithstanding.
    const auto& valid = outputs[i].properties.validShape.dimensionSize;
    const long long expected = static_cast<long long>(valid[1]) * valid[2] * valid[3];
    if (count != expected)
      throw std::runtime_error("S output " + std::to_string(i) + " has an inconsistent extent");
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
  // dropping the caller's value would make the documented parameters lie. The
  // backend is a bitmask (CORE_0..3 = 1ULL<<0..3, ANY = 1ULL<<7), so the CLI's
  // core index is converted explicitly by bpu_core_to_backend above instead of
  // being assigned raw (0 would select no backend, 1 would select core 0).
  schedule.backend = backend;
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
  dump.binary_path = current_binary_path(options.argv.empty() ? "" : options.argv.front());
  dump.options = {{"score_thres", std::to_string(options.score_threshold)},
                  {"nms_thres", std::to_string(options.nms_threshold)},
                  {"priority", std::to_string(options.priority)},
                  {"bpu_core", std::to_string(options.bpu_core)},
                  {"bpu_core_backend", std::to_string(backend)},
                  {"preprocess", "letterbox"},
                  {"nms_top_k_per_class", "-1"},
                  {"score_boundary", "greater-or-equal"},
                  {"label_file", options.label_path}};
  dump.notes = {"S UCP adapter; split NV12 672x672 Y/UV inputs; per-channel S32 dequantization.",
                "NMS preserves source S nms_bboxes semantics: keep score >= threshold, no "
                "per-class top_k.",
                "Scheduling honours the caller's priority/BPU core (source S forces priority 0); "
                "the core index is mapped to the SDK's backend bitmask.",
                "Input files hold the deterministic per-row payload the writer produced; "
                "inter-row padding bytes are uninitialized and are not dumped."};

  const TensorMeta y_meta = project(inputs[0].properties);
  const TensorMeta uv_meta = project(inputs[1].properties);
  dump.inputs.push_back(dump_tensor_info("y", y_meta));
  dump.inputs.push_back(dump_tensor_info("uv", uv_meta));
  // The input buffers actually submitted with this inference: the NV12 writer
  // fills each row of cols*channels bytes at stride[1] pitch, so the payload is
  // gathered row by row (the padding between rows is uninitialized memory).
  dump.input_tensors.push_back(gather_plane_rows(inputs[0], y_meta, "input0-y"));
  dump.input_tensors.push_back(gather_plane_rows(inputs[1], uv_meta, "input1-uv"));

  std::vector<std::vector<float>> dequantized;
  dequantized.reserve(outputs.size());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& meta = output_meta[i];
    const auto& props = outputs[i].properties;
    dump.outputs.push_back(dump_tensor_info("output" + std::to_string(i), meta,
                                            props.scale.scaleData,
                                            props.scale.zeroPointData));
    // The raw file keeps the full allocated extent (alignedByteSize), so a
    // padded layout is dumped exactly as the runtime stored it; the manifest
    // records the strides needed to interpret it.
    const std::size_t raw_extent = static_cast<std::size_t>(meta.aligned_byte_size);
    std::vector<unsigned char> raw_bytes(raw_extent);
    if (raw_extent > 0) std::memcpy(raw_bytes.data(), outputs[i].sysMem.virAddr, raw_extent);
    dump.raw_tensors.push_back({"output" + std::to_string(i), dtype_name(meta.dtype),
                                shape_of(meta), std::move(raw_bytes)});
    // Private broadcasting dequantizer (the shared c_utils helper indexes
    // scale_data[c] and would read out of bounds for a scalar descriptor);
    // the gate above proved the addressing contract it relies on.
    dequantized.push_back(dequant_s32_nhwc(
        static_cast<const unsigned char*>(outputs[i].sysMem.virAddr), meta,
        props.scale.scaleData, props.scale.scaleLen, props.scale.zeroPointData,
        props.scale.zeroPointLen));
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
