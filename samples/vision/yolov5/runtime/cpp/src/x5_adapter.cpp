// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"
#include "yolov5_decode.hpp"
#include "yolov5_visualize.hpp"

#include <dnn/hb_dnn.h>
#include <dnn/hb_dnn_ext.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

namespace yolov5 {
namespace {

constexpr int kClasses = 80;
constexpr int kInput = 640;
constexpr std::array<float, 18> kAnchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};

struct DnnLease {
  hbPackedDNNHandle_t packed = nullptr;
  hbDNNTaskHandle_t task = nullptr;
  hbDNNTensor input{};
  std::vector<hbDNNTensor> outputs;
  bool input_allocated = false;
  ~DnnLease() {
    if (task) hbDNNReleaseTask(task);
    if (input_allocated) hbSysFreeMem(&input.sysMem[0]);
    for (auto& tensor : outputs) hbSysFreeMem(&tensor.sysMem[0]);
    if (packed) hbDNNRelease(packed);
  }
};

void check(int code, const char* what) {
  if (code != 0) throw std::runtime_error(what);
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

}  // namespace

int run_native(const RuntimeOptions& options) {
  if (options.target != "x5") throw std::invalid_argument("X5 adapter received a non-X5 target");
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
  const auto& input_shape = input_properties.validShape;
  if (input_properties.tensorType != HB_DNN_IMG_TYPE_NV12 ||
      input_shape.numDimensions != 4 || input_shape.dimensionSize[0] != 1 ||
      input_shape.dimensionSize[1] != 3 || input_shape.dimensionSize[2] != kInput ||
      input_shape.dimensionSize[3] != kInput || input_properties.alignedByteSize == 0)
    throw std::runtime_error("X5 input must be NV12 [1,3,640,640] with aligned storage");
  const std::size_t input_bytes = input_properties.alignedByteSize;
  check(hbSysAllocCachedMem(&lease.input.sysMem[0], input_bytes),
        "hbSysAllocCachedMem input failed");
  lease.input_allocated = true;
  lease.input.properties = input_properties;
  cv::Mat image = cv::imread(options.image_path);
  const auto nv12 = letterbox_to_nv12(image);
  if (nv12.size() > input_bytes)
    throw std::runtime_error("X5 NV12 payload exceeds aligned input storage");
  std::memcpy(lease.input.sysMem[0].virAddr, nv12.data(), nv12.size());
  check(hbSysFlushMem(&lease.input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem input failed");

  lease.outputs.resize(3);
  std::vector<HeadShape> shapes;
  std::set<int> seen_strides;
  for (int i = 0; i < output_count; ++i) {
    auto& tensor = lease.outputs[static_cast<std::size_t>(i)];
    check(hbDNNGetOutputTensorProperties(&tensor.properties, model, i),
          "hbDNNGetOutputTensorProperties failed");
    if (tensor.properties.tensorType != HB_DNN_TENSOR_TYPE_F32 ||
        tensor.properties.quantiType != NONE)
      throw std::runtime_error("X5 YOLOv5 outputs must be native F32 with NONE quantization");
    if (tensor.properties.validShape.numDimensions != 4 ||
        tensor.properties.validShape.dimensionSize[0] != 1)
      throw std::runtime_error("X5 YOLOv5 outputs must be NHWC rank-4 tensors");
    const int h = tensor.properties.validShape.dimensionSize[1];
    const int w = tensor.properties.validShape.dimensionSize[2];
    const int channels = tensor.properties.validShape.dimensionSize[3];
    if (h <= 0 || w <= 0 || channels <= 0)
      throw std::runtime_error("X5 output metadata contains a non-positive dimension");
    const int classes_num = channels / 3 - 5;
    if (channels % 3 != 0 || classes_num != kClasses || channels !=
                                    3 * (5 + classes_num))
      throw std::runtime_error("X5 output channels are not 3*(5+classes)");
    const int stride = kInput / h;
    if (h != w || (stride != 8 && stride != 16 && stride != 32) || !seen_strides.insert(stride).second)
      throw std::runtime_error("X5 output stride is missing or duplicated");
    shapes.push_back({h, w, channels});
    const std::size_t bytes = tensor.properties.alignedByteSize;
    if (bytes == 0) throw std::runtime_error("X5 output has no aligned storage");
    check(hbSysAllocCachedMem(&tensor.sysMem[0], bytes),
          "hbSysAllocCachedMem output failed");
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
  for (const auto& tensor : lease.outputs) {
    const auto& shape = tensor.properties.validShape;
    const std::size_t count = static_cast<std::size_t>(shape.dimensionSize[1]) * shape.dimensionSize[2] * shape.dimensionSize[3];
    const float* values = static_cast<const float*>(tensor.sysMem[0].virAddr);
    raw.emplace_back(values, values + count);
  }
  const auto detections = decode_heads(raw, shapes, kInput, kClasses,
                                       options.score_threshold, options.nms_threshold, kAnchors);
  render_detections(image, detections, kInput, true, options.output_path,
                    load_labels(options.label_path));
  hbDNNReleaseTask(lease.task);
  lease.task = nullptr;
  return 0;
}

}  // namespace yolov5
