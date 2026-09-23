// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_adapter.hpp"
#include "yolov5_decode.hpp"
#include "yolov5_visualize.hpp"

#include <dnn/hb_dnn.h>
#include <dnn/hb_dnn_ext.h>
#include <hobot/hb_ucp.h>
#include "postprocess.hpp"
#include "preprocess.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolov5 {

namespace {
constexpr std::array<float, 18> kAnchors = {
    10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
}

struct SResourceGuard {
  hbPackedDNNHandle_t packed = nullptr;
  hbUCPTaskHandle_t task = nullptr;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  ~SResourceGuard() {
    if (task) hbUCPReleaseTask(task);
    for (auto& tensor : inputs) hbUCPFree(&tensor.sysMem);
    for (auto& tensor : outputs) hbUCPFree(&tensor.sysMem);
    if (packed) hbDNNRelease(packed);
  }
};

int run_native(const RuntimeOptions& options) {
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
  if (hbDNNGetModelNameList(&model_names, &model_count, packed) != 0 || model_count != 1) {
    throw std::runtime_error("S YOLOv5 asset must contain exactly one model");
  }
  hbDNNHandle_t model = nullptr;
  if (hbDNNGetModelHandle(&model, packed, model_names[0]) != 0) {
    throw std::runtime_error("hbDNNGetModelHandle failed");
  }
  int32_t input_count = 0;
  int32_t output_count = 0;
  if (hbDNNGetInputCount(&input_count, model) != 0 || input_count != 2 ||
      hbDNNGetOutputCount(&output_count, model) != 0 || output_count != 3) {
    throw std::runtime_error("S YOLOv5 requires two NV12 inputs and three outputs");
  }
  // The native S path keeps UCP scheduling independent from the X5 DNN adapter.
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
      y_shape.dimensionSize[0] != 1 || y_shape.dimensionSize[1] != 672 ||
      y_shape.dimensionSize[2] != 672 || y_shape.dimensionSize[3] != 1 ||
      uv_shape.dimensionSize[0] != 1 || uv_shape.dimensionSize[1] != 336 ||
      uv_shape.dimensionSize[2] != 336 || uv_shape.dimensionSize[3] != 2)
    throw std::runtime_error("S YOLOv5 inputs must be Y[1,672,672,1] and UV[1,336,336,2]");
  std::vector<yolov5::HeadShape> heads;
  for (int i = 0; i < output_count; ++i) {
    if (hbDNNGetOutputTensorProperties(&outputs[static_cast<std::size_t>(i)].properties, model, i) != 0)
      throw std::runtime_error("S output metadata query failed");
    const auto& shape = outputs[static_cast<std::size_t>(i)].properties.validShape;
    if (shape.numDimensions != 4 || shape.dimensionSize[0] != 1)
      throw std::runtime_error("S output must be rank-4 NHWC");
    heads.push_back({shape.dimensionSize[1], shape.dimensionSize[2], shape.dimensionSize[3]});
  }
  if (!validate_head_shapes(heads, 672, 80))
    throw std::runtime_error("S output heads must be unique 84/42/21 metadata heads");
  if (prepare_input_tensor(inputs) != 0 || prepare_output_tensor(outputs) != 0) {
    throw std::runtime_error("S tensor allocation failed");
  }
  cv::Mat image = cv::imread(options.image_path);
  if (image.empty()) {
    throw std::invalid_argument("S image is empty");
  }
  cv::Mat boxed(672, 672, CV_8UC3);
  letterbox_resize(image, boxed, 127);
  if (bgr_to_nv12_tensor(boxed, inputs, 672, 672) != 0) {
    throw std::runtime_error("S NV12 preprocessing failed");
  }
  hbUCPTaskHandle_t task = nullptr;
  if (hbDNNInferV2(&task, outputs.data(), inputs.data(), model) != 0) {
    throw std::runtime_error("hbDNNInferV2 failed");
  }
  guard.task = task;
  hbUCPSchedParam schedule{};
  HB_UCP_INITIALIZE_SCHED_PARAM(&schedule);
  schedule.backend = options.bpu_core < 0 ? HB_UCP_BPU_CORE_ANY : options.bpu_core;
  schedule.priority = options.priority;
  if (hbUCPSubmitTask(task, &schedule) != 0 || hbUCPWaitTaskDone(task, 0) != 0) {
    throw std::runtime_error("S UCP task failed");
  }
  // Source S outputs are dequantized only after cache invalidation, using their actual descriptors.
  // No fixed scale is inferred from the file name and no output is silently cast here.
  for (auto& output : outputs) hbUCPMemFlush(&output.sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
  std::vector<std::vector<float>> dequantized;
  dequantized.reserve(outputs.size());
  for (const auto& output : outputs) {
    // This is the source S32/per-channel helper; descriptor scales and strides are authoritative.
    dequantized.push_back(dequantizeTensorS32(output));
  }
  const auto detections = decode_heads(dequantized, heads, 672, 80,
                                       options.score_threshold, options.nms_threshold, kAnchors);
  render_detections(image, detections, 672, true, options.output_path,
                    load_labels(options.label_path));
  hbUCPReleaseTask(task);
  guard.task = nullptr;
  for (auto& input : inputs) hbUCPFree(&input.sysMem);
  for (auto& output : outputs) hbUCPFree(&output.sysMem);
  inputs.clear();
  outputs.clear();
  hbDNNRelease(packed);
  guard.packed = nullptr;
  return 0;
}

}  // namespace yolov5
