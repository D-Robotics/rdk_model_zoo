/*
 * Copyright (c) 2026, D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// YoloClassify: ImageNet classification over the single 1000-way FLOAT32
// logit output. Board resources live on the shared backend; this file owns
// the task lifecycle and the owned copy of the raw logits.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "backend.hpp"
#include "yolo.hpp"

namespace yolo {
namespace {

// Letterbox (resize_type 1) or plain resize (0), tracking the source
// mapping. Arithmetic identical to the released classify sample.
cv::Mat preprocess_frame(const cv::Mat& image, int input_h, int input_w,
                         int resize_type, ImageTransform* transform) {
  cv::Mat result;
  if (resize_type == 0) {
    cv::resize(image, result, cv::Size(input_w, input_h));
    transform->scale_x = 1.0f * input_w / image.cols;
    transform->scale_y = 1.0f * input_h / image.rows;
    transform->shift_x = 0;
    transform->shift_y = 0;
    return result;
  }

  transform->scale_x =
      std::min(1.0f * input_h / image.rows, 1.0f * input_w / image.cols);
  transform->scale_y = transform->scale_x;
  if (transform->scale_x <= 0 || transform->scale_y <= 0)
    throw std::runtime_error("Invalid scale factor");
  const int new_w = static_cast<int>(image.cols * transform->scale_x);
  const int new_h = static_cast<int>(image.rows * transform->scale_y);
  transform->shift_x = (input_w - new_w) / 2;
  transform->shift_y = (input_h - new_h) / 2;
  const int x_other = input_w - new_w - transform->shift_x;
  const int y_other = input_h - new_h - transform->shift_y;
  cv::resize(image, result, cv::Size(new_w, new_h));
  cv::copyMakeBorder(result, result, transform->shift_y, y_other,
                     transform->shift_x, x_other, cv::BORDER_CONSTANT,
                     cv::Scalar(127, 127, 127));
  return result;
}

}  // namespace

struct YoloClassify::Impl {
  Impl(const Config& config) : cfg(config) { initialize(); }

  void initialize() {
    const char* model_file = cfg.model_path.c_str();
    if (hbDNNInitializeFromFiles(&packed_model.handle, &model_file, 1) != 0)
      throw std::runtime_error("Failed to initialize model from file");

    const char** model_name_list = nullptr;
    int model_count = 0;
    if (hbDNNGetModelNameList(&model_name_list, &model_count,
                              packed_model.handle) != 0 ||
        model_count != 1 || model_name_list == nullptr ||
        model_name_list[0] == nullptr)
      throw std::runtime_error("Expected exactly one named model");
    const char* model_name = model_name_list[0];

    if (hbDNNGetModelHandle(&model_handle, packed_model.handle, model_name) !=
        0)
      throw std::runtime_error("Failed to get model handle");

    std::string protocol_error;
    plan = probe_input_protocol(model_handle, &protocol_error);
    if (plan.protocol == InputProtocol::kUnknown)
      throw std::runtime_error("Unsupported model input: " + protocol_error);

    int32_t output_count = 0;
    if (hbDNNGetOutputCount(&output_count, model_handle) != 0)
      throw std::runtime_error("Failed to get output count");
    if (output_count != 1)
      throw std::runtime_error(
          "Classification model should have exactly 1 output, but has " +
          std::to_string(output_count));

    if (hbDNNGetOutputTensorProperties(&output_properties, model_handle, 0) !=
        0)
      throw std::runtime_error("Failed to get output tensor properties");

    plan1000 = bind_classification(output_properties);
    std::cout << "[INFO] Output: 1000 FLOAT32 logits, class byte stride="
              << plan1000.class_stride << std::endl;

    if (!input.allocate(model_handle, plan))
      throw std::runtime_error("Failed to allocate model input tensors");
    if (output_owner.allocate(output_properties) != 0)
      throw std::runtime_error("Failed to allocate output tensor");

    std::cout << "[INFO] Model name: " << model_name << std::endl;
    std::cout << "[INFO] Input: "
              << (plan.protocol == InputProtocol::kPackedNv12
                      ? "packed NV12 "
                      : "split Y/UV NV12 ")
              << plan.input_w << "x" << plan.input_h << std::endl;
  }

  Config cfg;
  PackedModelOwner packed_model;
  hbDNNHandle_t model_handle = nullptr;
  InputPlan plan;
  Nv12Input input;
  OutputTensorOwner output_owner;
  hbDNNTensorProperties output_properties{};
  ClassificationPlan plan1000{};
};

YoloClassify::YoloClassify(const Config& config)
    : impl_(new Impl(config)) {}
YoloClassify::~YoloClassify() = default;

int YoloClassify::input_h() const { return impl_->plan.input_h; }
int YoloClassify::input_w() const { return impl_->plan.input_w; }

YoloClassify::Prepared YoloClassify::preprocess(const Input& input) const {
  if (input.source_rows <= 0 || input.source_cols <= 0 ||
      input.bgr.size() != static_cast<size_t>(input.source_rows) *
                              input.source_cols * 3)
    throw std::invalid_argument(
        "input pixels do not match the declared source geometry");

  const cv::Mat source(input.source_rows, input.source_cols, CV_8UC3,
                       const_cast<uint8_t*>(input.bgr.data()));
  Prepared prepared;
  const cv::Mat preprocessed =
      preprocess_frame(source, impl_->plan.input_h, impl_->plan.input_w,
                       impl_->cfg.resize_type, &prepared.transform);
  // Classification reports text only; no frame is carried through the stages.
  cv::Mat yuv;
  cv::cvtColor(preprocessed, yuv, cv::COLOR_BGR2YUV_I420);
  const uint8_t* y = yuv.ptr<uint8_t>();
  const size_t count = static_cast<size_t>(impl_->plan.input_h) *
                       impl_->plan.input_w;
  const uint8_t* u = y + count;
  const uint8_t* v = u + count / 4;
  prepared.y_plane.assign(y, y + count);
  prepared.uv_plane.resize(count / 2);
  for (size_t i = 0; i < count / 4; ++i) {
    prepared.uv_plane[2 * i] = u[i];
    prepared.uv_plane[2 * i + 1] = v[i];
  }
  prepared.source_rows = input.source_rows;
  prepared.source_cols = input.source_cols;
  return prepared;
}

YoloClassify::RawResult YoloClassify::infer(const Prepared& prepared) {
  // Length validation at entry, before any SDK call.
  const size_t count = static_cast<size_t>(impl_->plan.input_h) *
                       impl_->plan.input_w;
  if (prepared.source_rows <= 0 || prepared.source_cols <= 0 ||
      prepared.y_plane.size() != count || prepared.uv_plane.size() != count / 2)
    throw std::invalid_argument(
        "prepared input does not match the NV12 contract");

  if (!impl_->input.upload_planes(impl_->plan, prepared.y_plane.data(),
                                  prepared.y_plane.size(),
                                  prepared.uv_plane.data(),
                                  prepared.uv_plane.size()))
    throw std::runtime_error("Failed to upload the prepared frame");

  hbDNNTensor* output = &impl_->output_owner.tensor;
  if (infer_sync(output, impl_->input.tensors(), impl_->input.input_count(),
                 impl_->model_handle) != 0)
    throw std::runtime_error("Inference failed");

  // Invalidate and take an owned copy of the raw logits.
  if (YOLO_SYS_FLUSH(YOLO_SYS_MEM(*output), HB_SYS_MEM_CACHE_INVALIDATE) != 0)
    throw std::runtime_error("Failed to invalidate output cache");
  const auto* bytes = static_cast<const uint8_t*>(
      YOLO_SYS_MEM(*output)->virAddr);
  const size_t byte_count =
      static_cast<size_t>(impl_->output_properties.alignedByteSize);

  RawResult raw;
  raw.output_bytes.assign(bytes, bytes + byte_count);
  raw.plan = impl_->plan1000;
  raw.model_frame_bgr = prepared.model_frame_bgr;
  raw.transform = prepared.transform;
  raw.source_rows = prepared.source_rows;
  raw.source_cols = prepared.source_cols;
  return raw;
}

YoloClassify::Result YoloClassify::postprocess(const RawResult& raw) const {
  Result result;
  result.topk = classification_topk(raw.output_bytes.data(),
                                    raw.output_bytes.size(), raw.plan,
                                    impl_->cfg.topk);
  return result;
}

YoloClassify::Prediction YoloClassify::predict(const Input& input) {
  const Prepared prepared = preprocess(input);
  RawResult raw = infer(prepared);
  Result result = postprocess(raw);
  Prediction prediction;
  prediction.result = std::move(result);
  prediction.model_frame_bgr = std::move(raw.model_frame_bgr);
  prediction.model_rows = impl_->plan.input_h;
  prediction.model_cols = impl_->plan.input_w;
  prediction.transform = raw.transform;
  return prediction;
}

}  // namespace yolo
