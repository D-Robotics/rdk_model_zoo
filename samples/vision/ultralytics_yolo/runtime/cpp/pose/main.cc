/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

Copyright (c) 2024-2025, WuChao && MaChao D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// 注意: 此程序在RDK板端运行
// Attention: This program runs on RDK board.

// ============================================================================
// Configuration Parameters
// ============================================================================

// D-Robotics *.bin 模型路径
// Path to D-Robotics *.bin model
#define MODEL_PATH "source/reference_bin_models/pose/yolo11n_pose_bayese_640x640_nv12.bin"

// 测试图片路径
// Path to test image
#define TEST_IMG_PATH "../../../../../datasets/coco/assets/bus.jpg"

// 前处理方式: 0=Resize, 1=LetterBox
// Preprocessing method: 0=Resize, 1=LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 推理结果保存路径
// Path where the inference result will be saved
#define IMG_SAVE_PATH "pose_result.jpg"

// 模型的类别数量 (Pose模型只检测person)
// Number of classes in the model (Pose model only detects person)
#define CLASSES_NUM 1

// NMS的阈值
// Non-Maximum Suppression (NMS) threshold
#define NMS_THRESHOLD 0.45

// 分数阈值
// Score threshold
#define SCORE_THRESHOLD 0.25

// 关键点置信度阈值
// Keypoint confidence threshold
#define KPT_SCORE_THRESHOLD 0.5

// 控制回归部分离散化程度的超参数, DFL
// A hyperparameter that controls the discretization level of the regression part
#define REG 16

// 关键点数量 (COCO人体姿态: 17个关键点)
// Number of keypoints (COCO human pose: 17 keypoints)
#define KPT_NUM 17

// 关键点编码维度 (x, y, confidence)
// Keypoint encoding dimension
#define KPT_ENCODE 3

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <string>

// OpenCV
#include <opencv2/opencv.hpp>

// RDK BPU libDNN API (stack-portable layer; pulls in the X5 hbSys or the
// S-series UCP headers itself)
#include "common/decode.h"
#include "common/task_benchmark.h"
#include "common/task_output_binding.h"
#include "common/task_session.h"

// ============================================================================
// Macros
// ============================================================================

#define CHECK_SUCCESS(value, errmsg)                                         \
    do {                                                                     \
        auto ret_code = value;                                               \
        if (ret_code != 0) {                                                 \
            std::cerr << "\033[1;31m[ERROR]\033[0m " << __FILE__ << ":"     \
                      << __LINE__ << " " << errmsg                           \
                      << ", error code: " << ret_code << std::endl;          \
            return ret_code;                                                 \
        }                                                                    \
    } while (0)

#define LOG_INFO(msg) \
    std::cout << "\033[1;32m[INFO]\033[0m " << msg << std::endl

#define LOG_WARN(msg) \
    std::cout << "\033[1;33m[WARN]\033[0m " << msg << std::endl

#define LOG_ERROR(msg) \
    std::cerr << "\033[1;31m[ERROR]\033[0m " << msg << std::endl

#define LOG_TIME(msg, duration) \
    std::cout << "\033[1;31m" << msg << " = " << std::fixed            \
              << std::setprecision(2) << (duration) << " ms\033[0m"    \
              << std::endl

// ============================================================================
// COCO Keypoint Names and Skeleton
// ============================================================================

const std::vector<std::string> KEYPOINT_NAMES = {
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
};

// Skeleton connections (pairs of keypoint indices)
const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},           // Head
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // Arms
    {5, 11}, {6, 12}, {11, 12},               // Torso
    {11, 13}, {13, 15}, {12, 14}, {14, 16}    // Legs
};

const cv::Scalar KEYPOINT_COLOR = cv::Scalar(0, 0, 255);      // Red
const cv::Scalar SKELETON_COLOR = cv::Scalar(255, 0, 0);      // Blue
const cv::Scalar BBOX_COLOR = cv::Scalar(0, 255, 0);          // Green

// ============================================================================
// Pose Detection Result Structure
// ============================================================================

struct PoseDetection {
    cv::Rect2d bbox;                          // Bounding box
    float score;                              // Confidence score
    std::vector<cv::Point2f> keypoints;       // 17 keypoints (x, y)
    std::vector<float> keypoint_scores;       // 17 keypoint confidences
};

// ============================================================================
// Drawing
// ============================================================================

/**
 * @brief Draw pose detection results
 */
void draw_pose(cv::Mat& img, const PoseDetection& det, float kpt_threshold_raw) {
    // Draw bounding box
    int x1 = static_cast<int>(det.bbox.x);
    int y1 = static_cast<int>(det.bbox.y);
    int x2 = static_cast<int>(det.bbox.x + det.bbox.width);
    int y2 = static_cast<int>(det.bbox.y + det.bbox.height);

    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), BBOX_COLOR, 2);

    // Draw label
    std::string label = "person: " + std::to_string(det.score).substr(0, 4);
    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    int label_y = std::max(y1, label_size.height);
    cv::rectangle(img, cv::Point(x1, label_y - label_size.height),
                 cv::Point(x1 + label_size.width, label_y + baseline),
                 BBOX_COLOR, cv::FILLED);
    cv::putText(img, label, cv::Point(x1, label_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

    // Draw skeleton connections
    for (const auto& connection : SKELETON) {
        int idx1 = connection.first;
        int idx2 = connection.second;

        if (det.keypoint_scores[idx1] >= kpt_threshold_raw &&
            det.keypoint_scores[idx2] >= kpt_threshold_raw) {
            cv::Point pt1(static_cast<int>(det.keypoints[idx1].x),
                         static_cast<int>(det.keypoints[idx1].y));
            cv::Point pt2(static_cast<int>(det.keypoints[idx2].x),
                         static_cast<int>(det.keypoints[idx2].y));
            cv::line(img, pt1, pt2, SKELETON_COLOR, 2);
        }
    }

    // Draw keypoints
    for (int i = 0; i < KPT_NUM; i++) {
        if (det.keypoint_scores[i] >= kpt_threshold_raw) {
            int x = static_cast<int>(det.keypoints[i].x);
            int y = static_cast<int>(det.keypoints[i].y);

            cv::circle(img, cv::Point(x, y), 5, KEYPOINT_COLOR, -1);
            cv::circle(img, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1);

            // Draw keypoint index
            cv::putText(img, std::to_string(i), cv::Point(x, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, KEYPOINT_COLOR, 2);
            cv::putText(img, std::to_string(i), cv::Point(x, y),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255), 1);
        }
    }
}

// ============================================================================
// Pose Runtime
// ============================================================================

// One model context, input and output tensor set. Each benchmark stream owns
// its own runtime; `run` leaves the NMS result for reporting.
class PoseRuntime {
 public:
  PoseRuntime(const std::string& model_path, float score_threshold,
              float nms_threshold)
      : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {
    session_.initialize(model_path);
    outputs_.bind(session_.model(), session_.input_h(), session_.input_w(),
                  false);
    outputs_.allocate();
  }

  int input_h() const { return session_.input_h(); }
  int input_w() const { return session_.input_w(); }
  const char* implementation() const {
    return outputs_.heads.direct_ltrb ? "native_cpp_yolo26_pose_ltrb"
                                      : "native_cpp_yolo_pose_dfl";
  }

  // Timing starts with the in-memory BGR image and ends with NMS-kept poses.
  size_t run(const cv::Mat& image, int resize_type, yolo::StageTiming* timing) {
    const auto start = std::chrono::steady_clock::now();
    const cv::Mat resized = yolo::preprocess_image(image, input_h(), input_w(),
                                                   resize_type, &transform_);
    cv::Mat i420;
    cv::cvtColor(resized, i420, cv::COLOR_BGR2YUV_I420);
    session_.upload(i420.ptr<uint8_t>());
    const auto preprocessed = std::chrono::steady_clock::now();
    session_.infer(outputs_.tensors());
    const auto inferred = std::chrono::steady_clock::now();
    decode();
    const auto finished = std::chrono::steady_clock::now();
    timing->preprocess_ms = yolo::elapsed_ms(start, preprocessed);
    timing->runtime_ms = yolo::elapsed_ms(preprocessed, inferred);
    timing->postprocess_ms = yolo::elapsed_ms(inferred, finished);
    timing->end_to_end_ms = yolo::elapsed_ms(start, finished);
    return keep_.size();
  }

  // Kept detections with boxes and keypoints restored to the source image.
  std::vector<PoseDetection> results() const {
    const float inv_x_scale = 1.0f / transform_.scale_x;
    const float inv_y_scale = 1.0f / transform_.scale_y;
    std::vector<PoseDetection> restored;
    for (int idx : keep_) {
      PoseDetection det = detections_[idx];
      det.bbox.x = (det.bbox.x - transform_.shift_x) * inv_x_scale;
      det.bbox.y = (det.bbox.y - transform_.shift_y) * inv_y_scale;
      det.bbox.width *= inv_x_scale;
      det.bbox.height *= inv_y_scale;
      for (auto& kpt : det.keypoints) {
        kpt.x = (kpt.x - transform_.shift_x) * inv_x_scale;
        kpt.y = (kpt.y - transform_.shift_y) * inv_y_scale;
      }
      restored.push_back(det);
    }
    return restored;
  }

 private:
  void decode() {
    const std::vector<yolo::TensorView> views = outputs_.views();
    const float conf_thres_raw = -std::log(1.0f / score_threshold_ - 1.0f);
    detections_.clear();

    // All scales have been bound and validated with the same box encoding.
    const bool direct_ltrb = outputs_.heads.direct_ltrb;
    const int strides[3] = {8, 16, 32};
    for (int scale = 0; scale < 3; scale++) {
      const int grid_h = input_h() / strides[scale];
      const int grid_w = input_w() / strides[scale];
      const float stride = strides[scale];
      // Strided views over the physical outputs; consumed values are checked.
      const yolo::TensorView& box_view = views[outputs_.heads.box[scale]];
      const yolo::TensorView& cls_view = views[outputs_.heads.cls[scale]];
      const yolo::TensorView& kpt_view = views[outputs_.heads.extra[scale]];

      for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
          const float* cur_box = box_view.cell(h, w);
          const float* cur_cls = cls_view.cell(h, w);
          const float* cur_kpt = kpt_view.cell(h, w);

          // Check threshold (before sigmoid)
          yolo::require_finite(cur_cls, CLASSES_NUM);
          if (cur_cls[0] < conf_thres_raw) continue;
          yolo::require_finite(cur_box, direct_ltrb ? 4 : 4 * REG);
          yolo::require_finite(cur_kpt, KPT_NUM * KPT_ENCODE);
          const float score = 1.0f / (1.0f + std::exp(-cur_cls[0]));

          // Decode bbox: YOLO26 stores direct LTRB distances,
          // YOLO11-family uses DFL (Distribution Focal Loss).
          float ltrb[4] = {0.0f};
          if (direct_ltrb)
            yolo::decode_box_ltrb(cur_box, ltrb);
          else
            yolo::decode_box_dfl(cur_box, ltrb);
          const float cx = (w + 0.5f) * stride;
          const float cy = (h + 0.5f) * stride;
          const float x1 = cx - ltrb[0] * stride;
          const float y1 = cy - ltrb[1] * stride;
          const float x2 = cx + ltrb[2] * stride;
          const float y2 = cy + ltrb[3] * stride;
          if (!(x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && x2 <= input_w() &&
                y2 <= input_h()))
            continue;

          PoseDetection det;
          det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
          det.score = score;
          det.keypoints.resize(KPT_NUM);
          det.keypoint_scores.resize(KPT_NUM);
          for (int k = 0; k < KPT_NUM; k++) {
            const float kpt_x = cur_kpt[k * 3 + 0];
            const float kpt_y = cur_kpt[k * 3 + 1];
            if (direct_ltrb) {
              // YOLO26: keypoints regress directly from the grid centre, as
              // in runtime/python/pose_decode.py.
              det.keypoints[k] = cv::Point2f((kpt_x + w + 0.5f) * stride,
                                             (kpt_y + h + 0.5f) * stride);
            } else {
              // YOLO11-family:
              // kpts_xy = (kpts[:, :, :2] * 2.0 + (anchor - 0.5)) * stride
              det.keypoints[k] =
                  cv::Point2f((kpt_x * 2.0f + (w + 0.5f) - 0.5f) * stride,
                              (kpt_y * 2.0f + (h + 0.5f) - 0.5f) * stride);
            }
            // Both families keep the raw confidence here; the draw pass
            // compares against a raw-logit threshold, which is equivalent to
            // sigmoid-space thresholding.
            det.keypoint_scores[k] = cur_kpt[k * 3 + 2];
          }
          detections_.push_back(det);
        }
      }
    }

    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float> nms_scores;
    for (const auto& det : detections_) {
      nms_boxes.push_back(det.bbox);
      nms_scores.push_back(det.score);
    }
    keep_.clear();
    if (!nms_boxes.empty())
      cv::dnn::NMSBoxes(nms_boxes, nms_scores, score_threshold_, nms_threshold_,
                        keep_);
  }

  // Declared first so outputs are released before the model context.
  yolo::TaskSession session_;
  yolo::TaskOutputs outputs_;
  yolo::ImageTransform transform_;
  float score_threshold_;
  float nms_threshold_;
  std::vector<PoseDetection> detections_;
  std::vector<int> keep_;
};

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
  try {
    yolo::BenchmarkOptions defaults;
    defaults.score_threshold = SCORE_THRESHOLD;
    defaults.nms_threshold = NMS_THRESHOLD;
    const yolo::TaskCommand command = yolo::parse_task_command(
        argc, argv, {MODEL_PATH, TEST_IMG_PATH, IMG_SAVE_PATH}, defaults);
    if (command.help) {
      yolo::print_task_usage(argv[0], "MODEL IMAGE OUTPUT", true);
      return 0;
    }
    LOG_INFO("=== Ultralytics YOLO Pose Demo (C++) ===");
    LOG_INFO("Loading model: " << command.paths[0]);

    yolo::BenchmarkMeta meta;
    meta.output_kind = "pose_instances";
    meta.timing_scope = "in_memory_bgr_to_pose_instances";
    return yolo::run_task<PoseRuntime>(
        command, PREPROCESS_TYPE, meta,
        [&command]() {
          return std::unique_ptr<PoseRuntime>(new PoseRuntime(
              command.paths[0], command.options.score_threshold,
              command.options.nms_threshold));
        },
        [&command](PoseRuntime& runtime, const cv::Mat& image, int) {
          cv::Mat result_img = image.clone();
          const float kpt_thres_raw =
              -std::log(1.0f / KPT_SCORE_THRESHOLD - 1.0f);
          for (const PoseDetection& det : runtime.results()) {
            LOG_INFO("Person detected: score="
                     << std::fixed << std::setprecision(3) << det.score
                     << ", bbox=(" << det.bbox.x << "," << det.bbox.y << ","
                     << det.bbox.width << "," << det.bbox.height << ")");
            draw_pose(result_img, det, kpt_thres_raw);
          }
          if (!command.options.save_result) return;
          if (!cv::imwrite(command.paths[2], result_img))
            throw std::runtime_error("Failed to save pose image");
          LOG_INFO("Result saved to: " << command.paths[2]);
        });
  } catch (const std::exception& error) {
    LOG_ERROR(error.what());
    return 1;
  }
}
