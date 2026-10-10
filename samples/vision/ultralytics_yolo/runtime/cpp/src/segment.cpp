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

// YoloSegment: instance segmentation over the ten-output YOLO11/YOLO26
// contract (three scales of 80-class + box + 32-coefficient maps plus a
// stride-4 prototype), with class-agnostic NMS and prototype mask
// generation. Board resources live on the shared backend; this file owns
// the task lifecycle, decode and mask math.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "backend.hpp"
#include "yolo.hpp"

namespace yolo {
namespace {

const int kClasses = 80;   // CLASSES_NUM
const int kStrides[3] = {8, 16, 32};
const int kMces = 32;      // mask coefficients

// Letterbox (resize_type 1) or plain resize (0), tracking the source
// mapping. Arithmetic identical to the released segment sample.
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

struct YoloSegment::Impl {
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

    output_set.bind(model_handle, plan.input_h, plan.input_w, true);

    if (!input.allocate(model_handle, plan))
      throw std::runtime_error("Failed to allocate model input tensors");
    output_set.allocate();

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
  TaskOutputs output_set;
};

YoloSegment::YoloSegment(const Config& config)
    : impl_(new Impl(config)) {}
YoloSegment::~YoloSegment() = default;

int YoloSegment::input_h() const { return impl_->plan.input_h; }
int YoloSegment::input_w() const { return impl_->plan.input_w; }
bool YoloSegment::direct_ltrb() const { return impl_->output_set.heads.direct_ltrb; }

YoloSegment::Prepared YoloSegment::preprocess(const Input& input) const {
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
  // The segment renderer composites on the model-input frame, so the frame
  // is owned per call and carried through the stages.
  prepared.model_frame_bgr.assign(
      preprocessed.ptr<uint8_t>(),
      preprocessed.ptr<uint8_t>() +
          static_cast<size_t>(impl_->plan.input_h) * impl_->plan.input_w * 3);

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

YoloSegment::RawResult YoloSegment::infer(const Prepared& prepared) {
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

  if (infer_sync(impl_->output_set.tensors(), impl_->input.tensors(),
                 impl_->input.input_count(), impl_->model_handle) != 0)
    throw std::runtime_error("Inference failed");

  std::vector<std::vector<float>> values = impl_->output_set.read();

  RawResult raw;
  raw.outputs = std::move(values);
  raw.heads = impl_->output_set.heads;
  raw.model_frame_bgr = prepared.model_frame_bgr;
  raw.transform = prepared.transform;
  raw.model_rows = impl_->plan.input_h;
  raw.model_cols = impl_->plan.input_w;
  raw.source_rows = prepared.source_rows;
  raw.source_cols = prepared.source_cols;
  return raw;
}

YoloSegment::Result YoloSegment::postprocess(const RawResult& raw) const {
  const int input_h = raw.model_rows ? raw.model_rows : impl_->plan.input_h;
  const int input_w = raw.model_cols ? raw.model_cols : impl_->plan.input_w;
  const float conf_thres_raw =
      -std::log(1.0f / impl_->cfg.score_threshold - 1.0f);

  // Internal working set: detections in model-input coordinates with their
  // mask coefficients, exactly as the released sample accumulated them.
  struct Working {
    float x1, y1, x2, y2;
    float score;
    int class_id;
    std::vector<float> mask_coeffs;
  };
  std::vector<Working> detections;

  const int proto_h = input_h / 4;
  const int proto_w = input_w / 4;
  const bool direct_ltrb = raw.heads.direct_ltrb;

  for (int scale = 0; scale < 3; ++scale) {
    const int cls_idx = raw.heads.cls[scale];
    const int box_idx = raw.heads.box[scale];
    const int mce_idx = raw.heads.extra[scale];

    const int grid = input_h / kStrides[scale];
    const float stride = static_cast<float>(kStrides[scale]);

    const float* cls_raw = raw.outputs[cls_idx].data();
    const float* box_raw = raw.outputs[box_idx].data();
    const float* mce_raw = raw.outputs[mce_idx].data();

    for (int h = 0; h < grid; ++h) {
      for (int w = 0; w < grid; ++w) {
        const int offset = h * grid + w;

        const float* cur_cls = cls_raw + offset * kClasses;
        const float* cur_box = box_raw + offset * (direct_ltrb ? 4 : 4 * 16);
        const float* cur_mce = mce_raw + offset * kMces;

        // Find max class score, then threshold before sigmoid.
        int cls_id = 0;
        for (int i = 1; i < kClasses; i++) {
          if (cur_cls[i] > cur_cls[cls_id]) {
            cls_id = i;
          }
        }
        if (cur_cls[cls_id] < conf_thres_raw) {
          continue;
        }
        const float score = 1.0f / (1.0f + std::exp(-cur_cls[cls_id]));

        float ltrb[4] = {0.0f};
        if (direct_ltrb)
          decode_box_ltrb(cur_box, ltrb);
        else
          decode_box_dfl(cur_box, ltrb);

        const float cx = (w + 0.5f) * stride;
        const float cy = (h + 0.5f) * stride;
        const float x1 = cx - ltrb[0] * stride;
        const float y1 = cy - ltrb[1] * stride;
        const float x2 = cx + ltrb[2] * stride;
        const float y2 = cy + ltrb[3] * stride;

        if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= input_w &&
            y2 <= input_h) {
          Working det;
          det.x1 = x1;
          det.y1 = y1;
          det.x2 = x2;
          det.y2 = y2;
          det.score = score;
          det.class_id = cls_id;
          det.mask_coeffs.assign(cur_mce, cur_mce + kMces);
          detections.push_back(std::move(det));
        }
      }
    }
  }

  std::cout << "[INFO] Detections before NMS: " << detections.size()
            << std::endl;

  // Class-agnostic NMS over every detection.
  std::vector<cv::Rect2d> nms_boxes;
  std::vector<float> nms_scores;
  std::vector<int> nms_indices;
  nms_boxes.reserve(detections.size());
  nms_scores.reserve(detections.size());
  for (const Working& det : detections) {
    nms_boxes.push_back(cv::Rect2d(det.x1, det.y1, det.x2 - det.x1,
                                   det.y2 - det.y1));
    nms_scores.push_back(det.score);
  }
  if (!nms_boxes.empty()) {
    cv::dnn::NMSBoxes(nms_boxes, nms_scores, impl_->cfg.score_threshold,
                      impl_->cfg.nms_threshold, nms_indices);
  }
  std::cout << "[INFO] Detections after NMS: " << nms_indices.size()
            << std::endl;

  // Prototype masks: (proto_h*proto_w x kMces) x (kMces x 1) -> sigmoid ->
  // resize -> binarize -> crop to the box, per surviving detection.
  cv::Mat proto_mat(proto_h * proto_w, kMces, CV_32F,
                    const_cast<float*>(raw.outputs[raw.heads.prototype].data()));

  Result result;
  for (int idx : nms_indices) {
    const Working& det = detections[idx];
    Detection out;
    out.class_id = det.class_id;
    out.score = det.score;
    out.x1 = det.x1;
    out.y1 = det.y1;
    out.x2 = det.x2;
    out.y2 = det.y2;

    cv::Mat mce_mat(kMces, 1, CV_32F,
                    const_cast<float*>(det.mask_coeffs.data()));
    cv::Mat mask_flat = proto_mat * mce_mat;
    cv::Mat mask_low_res = mask_flat.reshape(1, proto_h);

    cv::Mat sigmoid_mask;
    cv::exp(-mask_low_res, sigmoid_mask);
    sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);

    cv::Mat resized_mask;
    cv::resize(sigmoid_mask, resized_mask, cv::Size(input_w, input_h), 0, 0,
               cv::INTER_LINEAR);

    cv::Mat binary_mask;
    cv::threshold(resized_mask, binary_mask, impl_->cfg.mask_threshold, 1.0,
                  cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U, 255);

    const int x1 = std::max(0.0, static_cast<double>(det.x1));
    const int y1 = std::max(0.0, static_cast<double>(det.y1));
    const int x2 = std::min(static_cast<double>(input_w),
                            static_cast<double>(det.x2));
    const int y2 = std::min(static_cast<double>(input_h),
                            static_cast<double>(det.y2));
    const int mask_w = x2 - x1;
    const int mask_h = y2 - y1;
    // A degenerate crop skips the whole detection (box included), exactly
    // like the released sample's `continue`.
    if (mask_w <= 0 || mask_h <= 0) continue;

    cv::Rect roi(x1, y1, mask_w, mask_h);
    cv::Mat roi_mask = binary_mask(roi);
    out.mask.x = x1;
    out.mask.y = y1;
    out.mask.rows = mask_h;
    out.mask.cols = mask_w;
    // The crop is a view into the full-frame mask: its rows are spaced by
    // the frame width, so copy row by row into the owned plane.
    out.mask.bytes.resize(static_cast<size_t>(mask_h) * mask_w);
    for (int row = 0; row < mask_h; ++row) {
      std::memcpy(&out.mask.bytes[static_cast<size_t>(row) * mask_w],
                  roi_mask.ptr<uint8_t>(row), mask_w);
    }
    result.detections.push_back(std::move(out));
  }
  return result;
}

YoloSegment::Prediction YoloSegment::predict(const Input& input) {
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
