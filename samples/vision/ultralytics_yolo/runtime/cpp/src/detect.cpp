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

// YoloDetect: detection over both published head contracts (YOLO26
// direct-LTRB and YOLO11-family DFL, auto-selected from the model's own
// output shapes) and both input protocols (packed NV12 on X5 .bin, split
// Y/UV NV12 on S-series .hbm). The board resources live on the shared
// backend (inc/backend.hpp); this file owns the task lifecycle and decode.

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

const int kClasses = 80;
const int kStrides[3] = {8, 16, 32};
const int kOutputCount = 6;

// Letterbox (resize_type 1) or plain resize (0) to the model input, tracking
// the source mapping. Same arithmetic as the released detect sample.
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

}  // namespace

struct YoloDetect::Impl {
  Impl(const Config& config) : cfg(config) { initialize(); }

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

    // Bind the six detect outputs: per stride one 80-channel class map plus
    // a box map whose channel count selects the head contract. Auto mode
    // tries direct-LTRB first, then DFL; --head pins one contract.
    outputs.bind(model_handle, kOutputCount,
                 [&](const std::vector<OutputShape>& shapes) {
                   bind_outputs(shapes);
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
    std::cout << "[INFO] Head: "
              << (direct_ltrb ? "direct-LTRB" : "DFL") << std::endl;
    std::cout << "[INFO] Output order: [";
    for (int i = 0; i < kOutputCount; ++i) {
      if (i) std::cout << ", ";
      std::cout << output_order[i];
    }
    std::cout << "]" << std::endl;
  }

  void bind_outputs(const std::vector<OutputShape>& shapes) {
    // Try the requested protocol first, then fall back for auto mode.
    std::vector<BoxDecode> candidates;
    if (cfg.head == "ltrb") {
      candidates.push_back(BoxDecode::kDirectLtrb);
    } else if (cfg.head == "dfl") {
      candidates.push_back(BoxDecode::kDfl);
    } else {
      candidates.push_back(BoxDecode::kDirectLtrb);
      candidates.push_back(BoxDecode::kDfl);
    }

    for (size_t candidate = 0; candidate < candidates.size(); ++candidate) {
      const int box_channel_count = box_channels(candidates[candidate]);
      bool complete = true;
      for (int scale = 0; scale < 3 && complete; ++scale) {
        const int h = plan.input_h / kStrides[scale];
        const int w = plan.input_w / kStrides[scale];
        const int cls_index = find_output_by_shape(shapes, h, w, kClasses);
        const int box_index =
            find_output_by_shape(shapes, h, w, box_channel_count);
        if (cls_index < 0 || box_index < 0) {
          complete = false;
          break;
        }
        output_order[scale * 2] = cls_index;
        output_order[scale * 2 + 1] = box_index;
      }
      if (complete) {
        direct_ltrb = candidates[candidate] == BoxDecode::kDirectLtrb;
        return;
      }
    }

    throw std::invalid_argument(
        "Missing detect outputs: expected per stride a 80-channel class map "
        "plus a 4-channel (YOLO26 direct-LTRB) or 64-channel (DFL) box map");
  }

  Config cfg;
  PackedModelOwner packed_model;
  hbDNNHandle_t model_handle = nullptr;
  InputPlan plan;
  Nv12Input input;
  TaskOutputs outputs;
  int output_order[kOutputCount] = {-1, -1, -1, -1, -1, -1};
  bool direct_ltrb = false;
  std::string model_name;
};

YoloDetect::YoloDetect(const Config& config)
    : impl_(new Impl(config)) {}
YoloDetect::~YoloDetect() = default;

int YoloDetect::input_h() const { return impl_->plan.input_h; }
int YoloDetect::input_w() const { return impl_->plan.input_w; }
bool YoloDetect::direct_ltrb() const { return impl_->direct_ltrb; }

YoloDetect::Prepared YoloDetect::preprocess(const Input& input) const {
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
  // The detect renderer draws on the caller's source image, so the model
  // frame is not carried through the stages.
  prepared.source_rows = input.source_rows;
  prepared.source_cols = input.source_cols;
  return prepared;
}

YoloDetect::RawResult YoloDetect::infer(const Prepared& prepared) {
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

  std::vector<std::vector<float>> values = impl_->outputs.read();
  std::vector<OutputShape> shapes;
  shapes.reserve(values.size());
  const int box_channel_count = box_channels(
      impl_->direct_ltrb ? BoxDecode::kDirectLtrb : BoxDecode::kDfl);
  for (int i = 0; i < kOutputCount; ++i) {
    const int scale = i / 2;
    const int grid = impl_->plan.input_h / kStrides[scale];
    const int channels =
        impl_->output_order[i] < 0 ? 0
        : i % 2 == 0              ? kClasses
                                   : box_channel_count;
    shapes.push_back(OutputShape(grid, grid, channels));
  }

  RawResult raw;
  raw.outputs = std::move(values);
  raw.shapes = std::move(shapes);
  for (int i = 0; i < kOutputCount; ++i) raw.output_order[i] = impl_->output_order[i];
  raw.direct_ltrb = impl_->direct_ltrb;
  raw.model_frame_bgr = prepared.model_frame_bgr;
  raw.transform = prepared.transform;
  raw.source_rows = prepared.source_rows;
  raw.source_cols = prepared.source_cols;
  return raw;
}

YoloDetect::Result YoloDetect::postprocess(const RawResult& raw) const {
  const float raw_threshold = raw_logit_threshold(impl_->cfg.score_threshold);
  std::vector<std::vector<cv::Rect2d> > boxes(kClasses);
  std::vector<std::vector<float> > scores(kClasses);

  for (int scale = 0; scale < 3; ++scale) {
    const int stride = kStrides[scale];
    const int grid = impl_->plan.input_h / stride;
    const std::vector<float>& cls_values = raw.outputs[raw.output_order[scale * 2]];
    const std::vector<float>& box_values =
        raw.outputs[raw.output_order[scale * 2 + 1]];
    const float* cls_raw = cls_values.data();
    const float* box_raw = box_values.data();
    const int box_channel_count =
        box_channels(raw.direct_ltrb ? BoxDecode::kDirectLtrb : BoxDecode::kDfl);

    for (int h = 0; h < grid; ++h) {
      for (int w = 0; w < grid; ++w) {
        const int offset = h * grid + w;
        const float* logits = cls_raw + offset * kClasses;
        int class_id = 0;
        for (int c = 1; c < kClasses; ++c) {
          if (logits[c] > logits[class_id]) class_id = c;
        }
        if (logits[class_id] < raw_threshold) continue;

        float ltrb[4];
        const float* box_cell = box_raw + offset * box_channel_count;
        if (raw.direct_ltrb) {
          decode_box_ltrb(box_cell, ltrb);
        } else {
          decode_box_dfl(box_cell, ltrb);
        }

        float x1, y1, x2, y2;
        box_from_distances(w + 0.5f, h + 0.5f, ltrb,
                           static_cast<float>(stride), &x1, &y1, &x2, &y2);
        if (!map_to_source(&x1, &y1, &x2, &y2, raw.transform,
                           raw.source_cols, raw.source_rows)) {
          continue;
        }

        boxes[class_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
        scores[class_id].push_back(sigmoid(logits[class_id]));
      }
    }
  }

  Result result;
  for (int class_id = 0; class_id < kClasses; ++class_id) {
    if (boxes[class_id].empty()) continue;
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes[class_id], scores[class_id],
                      impl_->cfg.score_threshold, impl_->cfg.nms_threshold,
                      keep, 1.0f, 0);
    for (size_t i = 0; i < keep.size(); ++i) {
      const int index = keep[i];
      const cv::Rect2d& box = boxes[class_id][index];
      Detection detection;
      detection.class_id = class_id;
      detection.score = scores[class_id][index];
      detection.x1 = static_cast<float>(box.x);
      detection.y1 = static_cast<float>(box.y);
      detection.x2 = static_cast<float>(box.x + box.width);
      detection.y2 = static_cast<float>(box.y + box.height);
      result.detections.push_back(detection);
    }
  }
  return result;
}

YoloDetect::Prediction YoloDetect::predict(const Input& input) {
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
