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

// YOLO26 oriented-box reference (direct LTRB + angle head). Decoding follows
// runtime/python/obb_decode.py, including its platform policy: X5 wraps angles
// to [-pi/2, pi/2), runs per-class rotated NMS and clips restored boxes; the
// S series runs class-agnostic rotated NMS and keeps unclipped geometry.
// Attention: This program runs on RDK board.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "common/decode.h"
#include "common/obb_decode.h"
#include "common/task_benchmark.h"
#include "common/task_output_binding.h"
#include "common/task_session.h"

#define MODEL_PATH "yolo26n_obb_640x640_nv12.bin"
#define TEST_IMG_PATH "dota.jpg"
#define IMG_SAVE_PATH "obb_result.jpg"
// Letterbox, as used by the Python OBB runtime.
#define PREPROCESS_TYPE 1
// DOTA-v1 class count of the reference YOLO26 OBB checkpoints.
#define CLASSES_NUM 15
#define NMS_THRESHOLD 0.2
#define SCORE_THRESHOLD 0.25

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl
#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

namespace {

const int kStrides[3] = {8, 16, 32};
const float kRadToDeg = 180.0f / static_cast<float>(M_PI);

const cv::Scalar kClassColors[] = {
    cv::Scalar(56, 56, 255),   cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255),
    cv::Scalar(29, 178, 255),  cv::Scalar(49, 210, 207),  cv::Scalar(10, 249, 72),
    cv::Scalar(23, 204, 146),  cv::Scalar(134, 219, 61),  cv::Scalar(52, 147, 26),
    cv::Scalar(187, 212, 0),   cv::Scalar(168, 153, 44),  cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52),   cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0)};

struct Detection {
  int class_id = 0;
  float score = 0.0f;
  yolo::RotatedBox box;
};

cv::RotatedRect to_rect(const yolo::RotatedBox& box) {
  return cv::RotatedRect(cv::Point2f(box.cx, box.cy),
                         cv::Size2f(box.width, box.height),
                         box.angle_rad * kRadToDeg);
}

float rotated_iou(const yolo::RotatedBox& a, const yolo::RotatedBox& b) {
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
std::vector<int> classwise_rotated_nms(const std::vector<Detection>& detections,
                                       int classes, float threshold) {
  std::vector<int> keep;
  for (int class_id = 0; class_id < classes; ++class_id) {
    std::vector<int> order;
    for (size_t i = 0; i < detections.size(); ++i)
      if (detections[i].class_id == class_id) order.push_back(static_cast<int>(i));
    std::stable_sort(order.begin(), order.end(), [&detections](int l, int r) {
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

// One model context, input and output tensor set. Each benchmark stream owns
// its own runtime; `run` leaves the restored rotated boxes for reporting.
class ObbRuntime {
 public:
  ObbRuntime(const std::string& model_path, const yolo::BenchmarkOptions& options)
      : options_(options) {
    session_.initialize(model_path);
    classes_ = options.classes > 0 ? options.classes : CLASSES_NUM;
    const int h = session_.input_h();
    const int w = session_.input_w();
    if (h <= 0 || h != w || h % 32)
      throw std::invalid_argument("OBB requires square stride-32 input geometry.");
    outputs_.bind(session_.model(), 9, [&](const std::vector<yolo::OutputShape>& shapes) {
      for (int i = 0; i < 3; ++i) {
        const int grid = h / kStrides[i];
        cls_[i] = yolo::find_output_by_shape(shapes, grid, grid, classes_);
        box_[i] = yolo::find_output_by_shape(shapes, grid, grid, 4);
        angle_[i] = yolo::find_output_by_shape(shapes, grid, grid, 1);
        if (cls_[i] < 0 || box_[i] < 0 || angle_[i] < 0)
          throw std::invalid_argument(
              "Expected unique class, LTRB box and angle NHWC outputs per "
              "stride; check --classes (1 and 4 collide with angle/box).");
      }
    });
    outputs_.allocate();
  }

  int input_h() const { return session_.input_h(); }
  int input_w() const { return session_.input_w(); }
  const char* implementation() const { return "native_cpp_yolo26_obb_ltrb"; }
  const std::vector<Detection>& results() const { return results_; }

  // Timing starts with the in-memory BGR image and ends with NMS-kept rotated
  // boxes restored to source pixels.
  size_t run(const cv::Mat& image, int resize_type, yolo::StageTiming* timing) {
    const auto start = std::chrono::steady_clock::now();
    yolo::ImageTransform transform;
    const cv::Mat resized = yolo::preprocess_image(image, input_h(), input_w(),
                                                   resize_type, &transform);
    if (resize_type == 1) {
      // Restore with the realized per-axis resize ratio, as geometry.py does,
      // rather than the ideal letterbox scale.
      transform.scale_x = static_cast<float>(
          static_cast<int>(image.cols * transform.scale_x)) / image.cols;
      transform.scale_y = static_cast<float>(
          static_cast<int>(image.rows * transform.scale_y)) / image.rows;
    }
    cv::Mat i420;
    cv::cvtColor(resized, i420, cv::COLOR_BGR2YUV_I420);
    session_.upload(i420.ptr<uint8_t>());
    const auto preprocessed = std::chrono::steady_clock::now();
    session_.infer(outputs_.tensors());
    const auto inferred = std::chrono::steady_clock::now();
    decode(transform, image.cols, image.rows);
    const auto finished = std::chrono::steady_clock::now();
    timing->preprocess_ms = yolo::elapsed_ms(start, preprocessed);
    timing->runtime_ms = yolo::elapsed_ms(preprocessed, inferred);
    timing->postprocess_ms = yolo::elapsed_ms(inferred, finished);
    timing->end_to_end_ms = yolo::elapsed_ms(start, finished);
    return results_.size();
  }

 private:
  void decode(const yolo::ImageTransform& transform, int image_w, int image_h) {
#if defined(YOLO_DNN_STACK_X5)
    const bool x5 = true;
#else
    const bool x5 = false;
#endif
    const std::vector<yolo::TensorView> views = outputs_.views();
    const float raw_threshold = -std::log(1.0f / options_.score_threshold - 1.0f);
    const float angle_offset = options_.angle_offset_degrees / kRadToDeg;
    std::vector<Detection> candidates;
    for (int scale = 0; scale < 3; ++scale) {
      const int stride = kStrides[scale];
      const int grid = input_h() / stride;
      // Strided views over the physical outputs; consumed values are checked.
      const yolo::TensorView& cls = views[cls_[scale]];
      const yolo::TensorView& box = views[box_[scale]];
      const yolo::TensorView& angle = views[angle_[scale]];
      for (int y = 0; y < grid; ++y) {
        for (int x = 0; x < grid; ++x) {
          const float* cell = cls.cell(y, x);
          const int class_id =
              static_cast<int>(std::max_element(cell, cell + classes_) - cell);
          yolo::require_finite(cell + class_id, 1);
          if (cell[class_id] < raw_threshold) continue;
          Detection detection;
          detection.class_id = class_id;
          detection.score = 1.0f / (1.0f + std::exp(-cell[class_id]));
          if (!yolo::decode_obb_cell(box.cell(y, x), angle.cell(y, x)[0], x + 0.5f,
                                     y + 0.5f, static_cast<float>(stride),
                                     options_.angle_sign, angle_offset,
                                     &detection.box))
            throw std::runtime_error("Decoded rotated box is nonfinite.");
          yolo::regularize_obb(&detection.box, options_.regularize_obb, x5);
          candidates.push_back(detection);
        }
      }
    }

    std::vector<int> keep;
    if (x5) {
      keep = classwise_rotated_nms(candidates, classes_, options_.nms_threshold);
    } else if (!candidates.empty()) {
      std::vector<cv::RotatedRect> rects;
      std::vector<float> scores;
      for (const Detection& detection : candidates) {
        rects.push_back(to_rect(detection.box));
        scores.push_back(detection.score);
      }
      cv::dnn::NMSBoxes(rects, scores, options_.score_threshold,
                        options_.nms_threshold, keep);
    }
    results_.clear();
    for (int index : keep) {
      Detection detection = candidates[index];
      if (!yolo::map_obb_to_source(&detection.box, transform, image_w, image_h, x5))
        throw std::runtime_error("Cannot restore rotated box to the source image.");
      results_.push_back(detection);
    }
  }

  // Declared first so outputs are released before the model context.
  yolo::TaskSession session_;
  yolo::TaskOutputs outputs_;
  yolo::BenchmarkOptions options_;
  int classes_ = CLASSES_NUM;
  int cls_[3] = {-1, -1, -1};
  int box_[3] = {-1, -1, -1};
  int angle_[3] = {-1, -1, -1};
  std::vector<Detection> results_;
};

void draw(cv::Mat* image, const std::vector<Detection>& detections) {
  for (const Detection& detection : detections) {
    cv::Point2f points[4];
    to_rect(detection.box).points(points);
    const cv::Scalar color = kClassColors[detection.class_id % 15];
    for (int p = 0; p < 4; ++p)
      cv::line(*image, points[p], points[(p + 1) % 4], color, 2, cv::LINE_AA);
    const std::string label = "class=" + std::to_string(detection.class_id) +
                              " " + std::to_string(detection.score).substr(0, 4);
    cv::putText(*image, label, points[0], cv::FONT_HERSHEY_SIMPLEX, 0.5, color,
                1, cv::LINE_AA);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    yolo::BenchmarkOptions defaults;
    defaults.classes = CLASSES_NUM;
    defaults.score_threshold = SCORE_THRESHOLD;
    defaults.nms_threshold = NMS_THRESHOLD;
    const yolo::TaskCommand command = yolo::parse_task_command(
        argc, argv, {MODEL_PATH, TEST_IMG_PATH, IMG_SAVE_PATH}, defaults);
    if (command.help) {
      yolo::print_task_usage(argv[0], "MODEL IMAGE OUTPUT", true);
      std::cout << "  --classes N              Class channels (15)\n"
                << "  --angle-sign X, --angle-offset DEGREES, --no-regularize\n";
      return 0;
    }
    LOG_INFO("=== Ultralytics YOLO26 OBB Demo (C++) ===");
    LOG_INFO("Loading model: " << command.paths[0]);

    yolo::BenchmarkMeta meta;
    meta.output_kind = "rotated_boxes";
    meta.timing_scope = "in_memory_bgr_to_rotated_boxes";
    return yolo::run_task<ObbRuntime>(
        command, PREPROCESS_TYPE, meta,
        [&command]() {
          return std::unique_ptr<ObbRuntime>(
              new ObbRuntime(command.paths[0], command.options));
        },
        [&command](ObbRuntime& runtime, const cv::Mat& image, int) {
          for (const Detection& d : runtime.results())
            LOG_INFO("class=" << d.class_id << " score=" << std::fixed
                              << std::setprecision(3) << d.score << " center=("
                              << d.box.cx << "," << d.box.cy << ") size=("
                              << d.box.width << "," << d.box.height
                              << ") angle_deg=" << d.box.angle_rad * kRadToDeg);
          if (!command.options.save_result) return;
          cv::Mat rendered = image.clone();
          draw(&rendered, runtime.results());
          if (!cv::imwrite(command.paths[2], rendered))
            throw std::runtime_error("Failed to save OBB image");
          LOG_INFO("Result saved to: " << command.paths[2]);
        });
  } catch (const std::exception& error) {
    LOG_ERROR(error.what());
    return 1;
  }
}
