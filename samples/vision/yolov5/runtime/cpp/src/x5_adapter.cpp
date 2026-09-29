// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"
#include "yolov5_decode.hpp"
#include "yolov5_dump.hpp"
#include "yolov5_gate.hpp"
#include "yolov5_visualize.hpp"

#include <dnn/hb_dnn.h>
#include <dnn/hb_dnn_ext.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstring>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef YOLOV5_TARGET_NAME
#define YOLOV5_TARGET_NAME "unknown"
#endif

namespace yolov5 {
namespace {

constexpr int kClasses = 80;
constexpr int kInput = 640;
// Source X5 main.cc passes NMS_TOP_K = 300 to cv::dnn::NMSBoxes per class.
constexpr int kX5TopK = 300;
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

int image_code(int32_t type) {
  if (type == HB_DNN_IMG_TYPE_NV12) return kImageNv12;
  return kImageUnknown;
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
  // The X5 adapter allocates exactly alignedByteSize for every input and output,
  // so that is the storage the gates must hold the reads against.
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

struct DnnLease {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNTaskHandle_t task = nullptr;
  hbDNNTensor input{};
  std::vector<hbDNNTensor> outputs;
  std::vector<bool> output_allocated;
  bool input_allocated = false;
  ~DnnLease() {
    if (task) hbDNNReleaseTask(task);
    if (input_allocated) hbSysFreeMem(&input.sysMem[0]);
    for (std::size_t i = 0; i < outputs.size(); ++i)
      if (i < output_allocated.size() && output_allocated[i])
        hbSysFreeMem(&outputs[i].sysMem[0]);
    if (packed) hbDNNRelease(packed);
  }
};

void check(int code, const char* what) {
  if (code != 0) throw std::runtime_error(what);
}

void require_gate(const Gate& gate) {
  if (!gate) throw std::runtime_error(gate.reason);
}

std::vector<unsigned char> letterbox_to_nv12(const cv::Mat& image) {
  if (image.empty()) throw std::invalid_argument("X5 image is empty");
  const double scale = std::min(static_cast<double>(kInput) / image.rows,
                                static_cast<double>(kInput) / image.cols);
  if (!(scale > 0.0)) throw std::invalid_argument("X5 image has invalid dimensions");
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(static_cast<int>(image.cols * scale),
                                      static_cast<int>(image.rows * scale)));
  cv::Mat boxed(kInput, kInput, CV_8UC3, cv::Scalar(127, 127, 127));
  resized.copyTo(boxed(cv::Rect((kInput - resized.cols) / 2,
                               (kInput - resized.rows) / 2,
                               resized.cols, resized.rows)));
  cv::Mat yuv;
  cv::cvtColor(boxed, yuv, cv::COLOR_BGR2YUV_I420);
  const std::size_t y_size = static_cast<std::size_t>(kInput) * kInput;
  const std::size_t uv_size = y_size / 4;
  std::vector<unsigned char> bytes(y_size + 2 * uv_size);
  std::memcpy(bytes.data(), yuv.data, y_size);
  const auto* u = yuv.data + y_size;
  const auto* v = u + uv_size;
  for (std::size_t i = 0; i < uv_size; ++i) {
    bytes[y_size + 2 * i] = u[i];
    bytes[y_size + 2 * i + 1] = v[i];
  }
  return bytes;
}

std::vector<unsigned char> to_bytes(const std::vector<float>& values) {
  std::vector<unsigned char> bytes(values.size() * sizeof(float));
  if (!values.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

}  // namespace

int run_native(const RuntimeOptions& options) {
  require_gate(check_target_matches_build(options.target, YOLOV5_TARGET_NAME));
  if (options.priority != 0 || options.bpu_core != -1)
    throw std::invalid_argument(
        "X5 adapter has no verified HB-DNN mapping for --priority/--bpu-core; "
        "only the defaults (priority 0, bpu-core -1) are accepted on x5");

  hbPackedDNNHandle_t packed = nullptr;
  const char* file = options.model_path.c_str();
  check(hbDNNInitializeFromFiles(&packed, &file, 1), "hbDNNInitializeFromFiles failed");
  DnnLease lease;
  lease.packed = packed;
  const char** names = nullptr;
  int model_count = 0;
  check(hbDNNGetModelNameList(&names, &model_count, packed), "hbDNNGetModelNameList failed");
  if (model_count != 1) throw std::runtime_error("YOLOv5 X5 asset must contain exactly one model");
  hbDNNHandle_t model = nullptr;
  check(hbDNNGetModelHandle(&model, packed, names[0]), "hbDNNGetModelHandle failed");
  int32_t input_count = 0;
  check(hbDNNGetInputCount(&input_count, model), "hbDNNGetInputCount failed");
  if (input_count != 1) throw std::runtime_error("YOLOv5 X5 requires one input");
  int32_t output_count = 0;
  check(hbDNNGetOutputCount(&output_count, model), "hbDNNGetOutputCount failed");
  if (output_count != 3) throw std::runtime_error("YOLOv5 X5 requires three outputs");

  hbDNNTensorProperties input_properties{};
  check(hbDNNGetInputTensorProperties(&input_properties, model, 0),
        "hbDNNGetInputTensorProperties failed");
  TensorMeta input_meta = project(input_properties);
  require_gate(check_x5_nv12_input(image_code(input_properties.tensorType), input_meta, kInput));
  // The X5 SDK encodes an image input's format in tensorType (HB_DNN_IMG_TYPE_
  // NV12 for this model), which no data-type name describes; the storage the
  // compact payload occupies is 8-bit, so the dump records uint8.
  input_meta.dtype = kDtypeU8;

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
                  {"preprocess", "letterbox"},
                  {"nms_top_k_per_class", std::to_string(kX5TopK)},
                  {"score_boundary", "strict-greater-than"},
                  {"label_file", options.label_path}};
  dump.notes = {"X5 HB-DNN adapter; packed NV12 640x640 input; native F32 NONE-quantized heads.",
                "NMS preserves source X5 per-class cv::dnn::NMSBoxes semantics: strict score "
                "boundary and top_k=300.",
                "Dump contents are read back from the same buffers the decoder consumed.",
                "The input tensor file holds the compact NV12 payload actually submitted; "
                "storage beyond the payload is uninitialized and is not dumped."};
  dump.inputs.push_back(dump_tensor_info("input0", input_meta));

  const long long input_bytes = input_meta.aligned_byte_size;
  check(hbSysAllocCachedMem(&lease.input.sysMem[0], static_cast<int>(input_bytes)),
        "hbSysAllocCachedMem input failed");
  lease.input_allocated = true;
  lease.input.properties = input_properties;
  cv::Mat image = cv::imread(options.image_path);
  const auto nv12 = letterbox_to_nv12(image);
  if (nv12.size() > static_cast<std::size_t>(input_bytes))
    throw std::runtime_error("X5 NV12 payload exceeds aligned input storage");
  std::memcpy(lease.input.sysMem[0].virAddr, nv12.data(), nv12.size());
  check(hbSysFlushMem(&lease.input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem input failed");
  dump.input_tensors.push_back({"input0", "uint8", shape_of(input_meta), nv12});

  lease.outputs.resize(static_cast<std::size_t>(output_count));
  lease.output_allocated.assign(static_cast<std::size_t>(output_count), false);
  std::vector<HeadShape> shapes;
  std::set<int> seen_strides;
  for (int i = 0; i < output_count; ++i) {
    hbDNNTensorProperties properties{};
    check(hbDNNGetOutputTensorProperties(&properties, model, i),
          "hbDNNGetOutputTensorProperties failed");
    const TensorMeta meta = project(properties);
    int stride_level = 0;
    const Gate gate = check_x5_head(meta, kInput, kClasses, &stride_level);
    if (!gate) throw std::runtime_error("X5 output " + std::to_string(i) + ": " + gate.reason);
    if (!seen_strides.insert(stride_level).second)
      throw std::runtime_error("X5 output stride is duplicated across heads");
    shapes.push_back({static_cast<int>(meta.valid[1]), static_cast<int>(meta.valid[2]),
                      static_cast<int>(meta.valid[3])});
    dump.outputs.push_back(dump_tensor_info("output" + std::to_string(i), meta));
    lease.outputs[static_cast<std::size_t>(i)].properties = properties;
    check(hbSysAllocCachedMem(&lease.outputs[static_cast<std::size_t>(i)].sysMem[0],
                              static_cast<int>(meta.aligned_byte_size)),
          "hbSysAllocCachedMem output failed");
    lease.output_allocated[static_cast<std::size_t>(i)] = true;
  }
  if (!validate_head_shapes(shapes, kInput, kClasses))
    throw std::runtime_error("X5 output metadata does not match published heads");

  hbDNNTensor* output_ptr = lease.outputs.data();
  hbDNNInferCtrlParam ctrl;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl);
  check(hbDNNInfer(&lease.task, &output_ptr, &lease.input, model, &ctrl), "hbDNNInfer failed");
  check(hbDNNWaitTaskDone(lease.task, 0), "hbDNNWaitTaskDone failed");
  for (auto& tensor : lease.outputs)
    check(hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
          "hbSysFlushMem output failed");

  std::vector<std::vector<float>> raw;
  raw.reserve(lease.outputs.size());
  for (std::size_t i = 0; i < lease.outputs.size(); ++i) {
    const auto& shape = shapes[i];
    const std::size_t count = static_cast<std::size_t>(shape.height) *
                              static_cast<std::size_t>(shape.width) *
                              static_cast<std::size_t>(shape.channels);
    const float* values = static_cast<const float*>(lease.outputs[i].sysMem[0].virAddr);
    raw.emplace_back(values, values + count);
    dump.raw_tensors.push_back({"output" + std::to_string(i), "float32",
                                {shape.height, shape.width, shape.channels}, to_bytes(raw.back())});
    dump.transformed_tensors.push_back({"output" + std::to_string(i), "float32",
                                        {shape.height, shape.width, shape.channels},
                                        to_bytes(raw.back())});
  }

  DecodePolicy policy;
  policy.score_threshold = options.score_threshold;
  policy.nms_threshold = options.nms_threshold;
  policy.top_k_per_class = kX5TopK;
  policy.strict_score_boundary = true;
  const auto detections = decode_heads(raw, shapes, kInput, kClasses, policy, kAnchors);
  dump.detections = detections;
  dump.detections_original = map_to_original(detections, image.cols, image.rows, kInput);
  if (!options.dump_dir.empty()) {
    std::string error;
    if (!write_dump(dump, &error))
      throw std::runtime_error("cannot write YOLOv5 X5 dump: " + error);
  }
  render_detections(image, detections, kInput, true, options.output_path,
                    load_labels(options.label_path));
  hbDNNReleaseTask(lease.task);
  lease.task = nullptr;
  return 0;
}

}  // namespace yolov5
