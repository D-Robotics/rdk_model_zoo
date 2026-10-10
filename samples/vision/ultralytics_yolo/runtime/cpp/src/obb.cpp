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

// YoloObb: YOLO26 oriented boxes (direct-LTRB + angle head, nine outputs).
// The decode matches the Python task entry (runtime/python/obb.py),
// including its platform policy: X5 wraps angles to [-pi/2, pi/2), runs
// per-class rotated NMS and clips restored boxes; the S series runs
// class-agnostic rotated NMS and keeps unclipped geometry. The board
// resources live on the shared backend (inc/backend.hpp); this file owns
// the task lifecycle and decode. Raw results are owned compact copies of
// the physical outputs; the decoder validates only the score, box and
// angle values it consumes, so unused background nonfinite values do not
// fail an otherwise valid frame.

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
#if CV_VERSION_MAJOR >= 5
// OpenCV 5 moved rotatedRectangleIntersection/contourArea from imgproc
// into the geometry module (and renamed the version macros).
#include <opencv2/geometry/2d.hpp>
#endif

#include "backend.hpp"
#include "yolo.hpp"

namespace yolo {
namespace {

const int kStrides[3] = {8, 16, 32};
const int kOutputCount = 9;
const float kRadToDeg = 180.0f / static_cast<float>(M_PI);

// Letterbox (resize_type 1) or plain resize (0) to the model input, tracking
// the source mapping. Same arithmetic as the other task runtimes.
cv::Mat preprocess_frame(const cv::Mat& image, int input_h, int input_w,
                         int resize_type, ImageTransform* transform) {
  cv::Mat result;
  if (resize_type == 0) {
    cv::resize(image, result, cv::Size(input_w, input_h));
    transform->scale_x = static_cast<float>(input_w) / image.cols;
    transform->scale_y = static_cast<float>(input_h) / image.rows;
    transform->shift_x = 0;
    transform->shift_y = 0;
    return result;
  }

  const float scale = std::min(static_cast<float>(input_h) / image.rows,
                               static_cast<float>(input_w) / image.cols);
  const int resized_w = static_cast<int>(image.cols * scale);
  const int resized_h = static_cast<int>(image.rows * scale);
  transform->scale_x = scale;
  transform->scale_y = scale;
  transform->shift_x = (input_w - resized_w) / 2;
  transform->shift_y = (input_h - resized_h) / 2;
  const int right = input_w - resized_w - transform->shift_x;
  const int bottom = input_h - resized_h - transform->shift_y;

  cv::resize(image, result, cv::Size(resized_w, resized_h));
  cv::copyMakeBorder(result, result, transform->shift_y, bottom,
                     transform->shift_x, right, cv::BORDER_CONSTANT,
                     cv::Scalar(127, 127, 127));
  return result;
}

cv::RotatedRect to_rect(const RotatedBox& box) {
  return cv::RotatedRect(cv::Point2f(box.cx, box.cy),
                         cv::Size2f(box.width, box.height),
                         box.angle_rad * kRadToDeg);
}

float rotated_iou(const RotatedBox& a, const RotatedBox& b) {
  std::vector<cv::Point2f> points;
  if (cv::rotatedRectangleIntersection(to_rect(a), to_rect(b), points) <= 0 ||
      points.empty())
    return 0.0f;
  const double intersection = cv::contourArea(points);
  const double union_area = static_cast<double>(a.width) * a.height +
                            static_cast<double>(b.width) * b.height -
                            intersection;
  return union_area <= 0.0 ? 0.0f
                           : static_cast<float>(intersection / union_area);
}

// X5 policy: per-class greedy rotated NMS in descending score order.
std::vector<int> classwise_rotated_nms(
    const std::vector<YoloObb::Detection>& detections, int classes,
    float threshold) {
  std::vector<int> keep;
  for (int class_id = 0; class_id < classes; ++class_id) {
    std::vector<int> order;
    for (size_t i = 0; i < detections.size(); ++i)
      if (detections[i].class_id == class_id)
        order.push_back(static_cast<int>(i));
    std::stable_sort(order.begin(), order.end(),
                     [&detections](int l, int r) {
                       return detections[l].score > detections[r].score;
                     });
    while (!order.empty()) {
      const int current = order.front();
      keep.push_back(current);
      std::vector<int> remaining;
      for (size_t i = 1; i < order.size(); ++i)
        if (rotated_iou(detections[current].box, detections[order[i]].box) <
            threshold)
          remaining.push_back(order[i]);
      order.swap(remaining);
    }
  }
  return keep;
}

}  // namespace

struct YoloObb::Impl {
  Impl(const Config& config) : cfg(config) {
    if (cfg.classes <= 0)
      throw std::invalid_argument("OBB requires a positive class count.");
    initialize();
  }

  void initialize() {
    const char* model_file = cfg.model_path.c_str();
    if (hbDNNInitializeFromFiles(&packed_model.handle, &model_file, 1) != 0)
      throw std::runtime_error("Failed to initialize model from file");

    const char** model_names = nullptr;
    int model_count = 0;
    if (hbDNNGetModelNameList(&model_names, &model_count,
                              packed_model.handle) != 0 ||
        model_count <= 0 || model_names == nullptr)
      throw std::runtime_error("Failed to get model name list");
    model_name = model_names[0];
    if (hbDNNGetModelHandle(&model_handle, packed_model.handle,
                            model_names[0]) != 0)
      throw std::runtime_error("Failed to get model handle");

    std::string protocol_error;
    plan = probe_input_protocol(model_handle, &protocol_error);
    if (plan.protocol == InputProtocol::kUnknown)
      throw std::runtime_error("Unsupported model input: " + protocol_error);
    if (plan.input_h <= 0 || plan.input_h != plan.input_w ||
        plan.input_h % 32)
      throw std::invalid_argument(
          "OBB requires square stride-32 input geometry.");

    // Bind the nine OBB outputs: per stride one class map, one 4-channel
    // direct-LTRB box map and one 1-channel angle map. Class counts that
    // collide with the box/angle roles are rejected by the binder.
    outputs.bind(model_handle, kOutputCount,
                 [&](const std::vector<OutputShape>& shapes) {
                   heads = bind_obb_heads(shapes, plan.input_h, plan.input_w,
                                          cfg.classes);
                 });

    if (!input.allocate(model_handle, plan))
      throw std::runtime_error("Failed to allocate model input tensors");
    outputs.allocate();

    std::cout << "[INFO] Model name: " << model_name << std::endl;
    std::cout << "[INFO] Input: "
              << (plan.protocol == InputProtocol::kPackedNv12
                      ? "packed NV12 "
                      : "split Y/UV NV12 ")
              << plan.input_w << "x" << plan.input_h << std::endl;
    std::cout << "[INFO] Head: direct-LTRB + angle, classes=" << cfg.classes
              << std::endl;
  }

  Config cfg;
  PackedModelOwner packed_model;
  hbDNNHandle_t model_handle = nullptr;
  InputPlan plan;
  Nv12Input input;
  TaskOutputs outputs;
  ObbHeadPlan heads;
  std::string model_name;
};

YoloObb::YoloObb(const Config& config)
    : impl_(new Impl(config)) {}
YoloObb::~YoloObb() = default;

int YoloObb::input_h() const { return impl_->plan.input_h; }
int YoloObb::input_w() const { return impl_->plan.input_w; }
int YoloObb::classes() const { return impl_->cfg.classes; }

YoloObb::Prepared YoloObb::preprocess(const Input& input) const {
  if (input.source_rows <= 0 || input.source_cols <= 0 ||
      input.bgr.size() != static_cast<size_t>(input.source_rows) *
                              input.source_cols * 3)
    throw std::invalid_argument(
        "input pixels do not match the declared source geometry");

  const cv::Mat source(input.source_rows, input.source_cols, CV_8UC3,
                       const_cast<uint8_t*>(input.bgr.data()));
  Prepared prepared;
  const cv::Mat resized =
      preprocess_frame(source, impl_->plan.input_h, impl_->plan.input_w,
                       impl_->cfg.resize_type, &prepared.transform);
  if (impl_->cfg.resize_type == 1) {
    // Restore with the realized per-axis resize ratio (the integer-rounded
    // size the resize actually produced), not the ideal letterbox scale.
    prepared.transform.scale_x =
        static_cast<float>(static_cast<int>(input.source_cols *
                                            prepared.transform.scale_x)) /
        input.source_cols;
    prepared.transform.scale_y =
        static_cast<float>(static_cast<int>(input.source_rows *
                                            prepared.transform.scale_y)) /
        input.source_rows;
  }
  cv::Mat yuv;
  cv::cvtColor(resized, yuv, cv::COLOR_BGR2YUV_I420);
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

YoloObb::RawResult YoloObb::infer(const Prepared& prepared) {
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

  if (infer_sync(impl_->outputs.tensors(), impl_->input.tensors(),
                 impl_->input.input_count(), impl_->model_handle) != 0)
    throw std::runtime_error("Inference failed");

  RawResult raw;
  // Owned compact copies of the cache-invalidated physical outputs. Unlike
  // the other tasks' read(), values are not validated here: the OBB decoder
  // checks only the score, box and angle it actually consumes.
  const std::vector<TensorView> views = impl_->outputs.views();
  raw.outputs.resize(views.size());
  for (size_t i = 0; i < views.size(); ++i) {
    const TensorView& view = views[i];
    std::vector<float>& compact = raw.outputs[i];
    compact.reserve(static_cast<size_t>(view.h) * view.w * view.channels);
    for (int y = 0; y < view.h; ++y) {
      for (int x = 0; x < view.w; ++x) {
        const float* values = view.cell(y, x);
        compact.insert(compact.end(), values, values + view.channels);
      }
    }
  }
  raw.heads = impl_->heads;
  raw.classes = impl_->cfg.classes;
  raw.transform = prepared.transform;
  raw.source_rows = prepared.source_rows;
  raw.source_cols = prepared.source_cols;
  return raw;
}

YoloObb::Result YoloObb::postprocess(const RawResult& raw) const {
#if defined(YOLO_DNN_STACK_X5)
  const bool x5 = true;
#else
  const bool x5 = false;
#endif
  const float raw_threshold = raw_logit_threshold(impl_->cfg.score_threshold);
  const float angle_offset = impl_->cfg.angle_offset_degrees / kRadToDeg;
  std::vector<YoloObb::Detection> candidates;

  for (int scale = 0; scale < 3; ++scale) {
    const int stride = kStrides[scale];
    const int grid = impl_->plan.input_h / stride;
    const std::vector<float>& cls_values = raw.outputs[raw.heads.cls[scale]];
    const std::vector<float>& box_values = raw.outputs[raw.heads.box[scale]];
    const std::vector<float>& angle_values =
        raw.outputs[raw.heads.angle[scale]];
    const float* cls_raw = cls_values.data();
    const float* box_raw = box_values.data();
    const float* angle_raw = angle_values.data();

    for (int h = 0; h < grid; ++h) {
      for (int w = 0; w < grid; ++w) {
        const int offset = h * grid + w;
        const float* logits = cls_raw + offset * raw.classes;
        const int class_id = static_cast<int>(
            std::max_element(logits, logits + raw.classes) - logits);
        // Consumed score must be finite; unused background box/angle
        // values are never read and may be nonfinite.
        if (!std::isfinite(logits[class_id]))
          throw std::runtime_error("Output contains nonfinite values.");
        if (logits[class_id] < raw_threshold) continue;

        Detection detection;
        detection.class_id = class_id;
        detection.score = sigmoid(logits[class_id]);
        if (!decode_obb_cell(box_raw + offset * 4, angle_raw[offset],
                             w + 0.5f, h + 0.5f, static_cast<float>(stride),
                             impl_->cfg.angle_sign, angle_offset,
                             &detection.box))
          throw std::runtime_error("Decoded rotated box is nonfinite.");
        regularize_obb(&detection.box, impl_->cfg.regularize_obb, x5);
        candidates.push_back(detection);
      }
    }
  }

  std::vector<int> keep;
  if (x5) {
    keep = classwise_rotated_nms(candidates, raw.classes,
                                 impl_->cfg.nms_threshold);
  } else if (!candidates.empty()) {
    std::vector<cv::RotatedRect> rects;
    std::vector<float> scores;
    for (const Detection& detection : candidates) {
      rects.push_back(to_rect(detection.box));
      scores.push_back(detection.score);
    }
    cv::dnn::NMSBoxes(rects, scores, impl_->cfg.score_threshold,
                      impl_->cfg.nms_threshold, keep);
  }

  Result result;
  for (int index : keep) {
    Detection detection = candidates[index];
    if (!map_obb_to_source(&detection.box, raw.transform, raw.source_cols,
                           raw.source_rows, x5))
      throw std::runtime_error(
          "Cannot restore rotated box to the source image.");
    result.detections.push_back(detection);
  }
  return result;
}

YoloObb::Prediction YoloObb::predict(const Input& input) {
  const Prepared prepared = preprocess(input);
  RawResult raw = infer(prepared);
  Result result = postprocess(raw);
  Prediction prediction;
  prediction.result = std::move(result);
  prediction.model_rows = impl_->plan.input_h;
  prediction.model_cols = impl_->plan.input_w;
  prediction.transform = raw.transform;
  return prediction;
}

}  // namespace yolo
