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

// YoloPose: human pose over the nine-output YOLO11/YOLO26 contract (three
// scales of 1-class + box + 51-channel keypoint maps), class-agnostic NMS,
// COCO 17-keypoint skeletons. Keypoint scores stay raw logits and the
// renderer thresholds them in logit space. Board resources live on the
// shared backend; this file owns the task lifecycle and decode.

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

const int kClasses = 1;    // pose models detect person only
const int kStrides[3] = {8, 16, 32};
const int kKptNum = 17;    // COCO keypoints
const int kKptEncode = 3;  // x, y, confidence

// Letterbox (resize_type 1) or plain resize (0), tracking the source
// mapping. Arithmetic identical to the released pose sample.
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

struct YoloPose::Impl {
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

    output_set.bind(model_handle, plan.input_h, plan.input_w, false);

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

YoloPose::YoloPose(const Config& config)
    : impl_(new Impl(config)) {}
YoloPose::~YoloPose() = default;

int YoloPose::input_h() const { return impl_->plan.input_h; }
int YoloPose::input_w() const { return impl_->plan.input_w; }
bool YoloPose::direct_ltrb() const { return impl_->output_set.heads.direct_ltrb; }

YoloPose::Prepared YoloPose::preprocess(const Input& input) const {
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
  // The pose renderer draws on the caller's source image, so the model frame
  // is not carried through the stages.
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

YoloPose::RawResult YoloPose::infer(const Prepared& prepared) {
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
  raw.source_rows = prepared.source_rows;
  raw.source_cols = prepared.source_cols;
  return raw;
}

YoloPose::Result YoloPose::postprocess(const RawResult& raw) const {
  const int input_h = impl_->plan.input_h;
  const int input_w = impl_->plan.input_w;
  const float conf_thres_raw =
      -std::log(1.0f / impl_->cfg.score_threshold - 1.0f);

  // Working detections in model-input coordinates; mapped to the source
  // image only after NMS, exactly like the released sample.
  struct Working {
    float x1, y1, x2, y2;
    float score;
    std::array<cv::Point2f, kKptNum> keypoints;
    std::array<float, kKptNum> keypoint_scores;
  };
  std::vector<Working> detections;

  const bool direct_ltrb = raw.heads.direct_ltrb;

  for (int scale = 0; scale < 3; ++scale) {
    const int cls_idx = raw.heads.cls[scale];
    const int box_idx = raw.heads.box[scale];
    const int kpt_idx = raw.heads.extra[scale];

    const int grid = input_h / kStrides[scale];
    const float stride = static_cast<float>(kStrides[scale]);

    const float* cls_raw = raw.outputs[cls_idx].data();
    const float* box_raw = raw.outputs[box_idx].data();
    const float* kpt_raw = raw.outputs[kpt_idx].data();

    for (int h = 0; h < grid; ++h) {
      for (int w = 0; w < grid; ++w) {
        const int offset = h * grid + w;

        const float* cur_box = box_raw + offset * (direct_ltrb ? 4 : 4 * 16);
        const float* cur_cls = cls_raw + offset * kClasses;
        const float* cur_kpt = kpt_raw + offset * (kKptNum * kKptEncode);

        // Threshold the raw logit before sigmoid.
        if (cur_cls[0] < conf_thres_raw) {
          continue;
        }
        const float score = 1.0f / (1.0f + std::exp(-cur_cls[0]));

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

          for (int k = 0; k < kKptNum; k++) {
            const float kpt_x = cur_kpt[k * 3 + 0];
            const float kpt_y = cur_kpt[k * 3 + 1];
            const float kpt_conf = cur_kpt[k * 3 + 2];

            float decoded_x;
            float decoded_y;
            if (direct_ltrb) {
              // YOLO26: keypoints regress directly from the grid centre.
              decoded_x = (kpt_x + w + 0.5f) * stride;
              decoded_y = (kpt_y + h + 0.5f) * stride;
            } else {
              // YOLO11-family:
              // kpts_xy = (kpts[:, :, :2] * 2.0 + (anchor - 0.5)) * stride
              decoded_x = (kpt_x * 2.0f + (w + 0.5f) - 0.5f) * stride;
              decoded_y = (kpt_y * 2.0f + (h + 0.5f) - 0.5f) * stride;
            }

            det.keypoints[k] = cv::Point2f(decoded_x, decoded_y);
            // Raw confidence; the draw pass compares against a raw-logit
            // threshold, which is equivalent to sigmoid-space thresholding.
            det.keypoint_scores[k] = kpt_conf;
          }

          detections.push_back(std::move(det));
        }
      }
    }
  }

  std::cout << "[INFO] Detections before NMS: " << detections.size()
            << std::endl;

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

  // Map survivors back to source coordinates. The width/height are scaled
  // WITHOUT re-adjusting for the shift (preserving the released mapping).
  Result result;
  const float inv_x_scale = 1.0f / raw.transform.scale_x;
  const float inv_y_scale = 1.0f / raw.transform.scale_y;
  for (int idx : nms_indices) {
    const Working& det = detections[idx];
    Detection out;
    out.score = det.score;
    out.x1 = (det.x1 - raw.transform.shift_x) * inv_x_scale;
    out.y1 = (det.y1 - raw.transform.shift_y) * inv_y_scale;
    out.x2 = out.x1 + (det.x2 - det.x1) * inv_x_scale;
    out.y2 = out.y1 + (det.y2 - det.y1) * inv_y_scale;
    for (int k = 0; k < kKptNum; ++k) {
      out.keypoints[k].x =
          (det.keypoints[k].x - raw.transform.shift_x) * inv_x_scale;
      out.keypoints[k].y =
          (det.keypoints[k].y - raw.transform.shift_y) * inv_y_scale;
      out.keypoints[k].score = det.keypoint_scores[k];
    }
    result.detections.push_back(out);
  }
  return result;
}

YoloPose::Prediction YoloPose::predict(const Input& input) {
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
