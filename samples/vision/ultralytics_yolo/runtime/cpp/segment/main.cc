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
#define MODEL_PATH "source/reference_bin_models/seg/yolo11n_seg_bayese_640x640_nv12.bin"

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
#define IMG_SAVE_PATH "segment_result.jpg"

// 模型的类别数量
// Number of classes in the model
#define CLASSES_NUM 80

// NMS的阈值
// Non-Maximum Suppression (NMS) threshold
#define NMS_THRESHOLD 0.45

// 分数阈值
// Score threshold
#define SCORE_THRESHOLD 0.25

// 控制回归部分离散化程度的超参数, DFL
// A hyperparameter that controls the discretization level of the regression part
#define REG 16

// 掩码系数数量
// Mask Coefficients
#define MCES 32

// 掩码阈值
// Mask threshold
#define MASK_THRESHOLD 0.5

// ============================================================================
// Includes
// ============================================================================

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
// COCO Names and Colors
// ============================================================================

const std::vector<std::string> COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

const std::vector<cv::Scalar> RDK_COLORS = {
    cv::Scalar(56, 56, 255), cv::Scalar(151, 157, 255), cv::Scalar(31, 112, 255), cv::Scalar(29, 178, 255),
    cv::Scalar(49, 210, 207), cv::Scalar(10, 249, 72), cv::Scalar(23, 204, 146), cv::Scalar(134, 219, 61),
    cv::Scalar(52, 147, 26), cv::Scalar(187, 212, 0), cv::Scalar(168, 153, 44), cv::Scalar(255, 194, 0),
    cv::Scalar(147, 69, 52), cv::Scalar(255, 115, 100), cv::Scalar(236, 24, 0), cv::Scalar(255, 56, 132),
    cv::Scalar(133, 0, 82), cv::Scalar(255, 56, 203), cv::Scalar(200, 149, 255), cv::Scalar(199, 55, 255)
};

// ============================================================================
// Detection Result Structure
// ============================================================================

struct Detection {
    cv::Rect2d bbox;           // Bounding box
    float score;               // Confidence score
    int class_id;              // Class ID
    std::vector<float> mask_coeffs;  // Mask coefficients (32)
};

// ============================================================================
// Drawing
// ============================================================================

/**
 * @brief Draw detection results on image
 */
void draw_detection(cv::Mat& img, const Detection& det) {
    int x1 = static_cast<int>(det.bbox.x);
    int y1 = static_cast<int>(det.bbox.y);
    int x2 = static_cast<int>(det.bbox.x + det.bbox.width);
    int y2 = static_cast<int>(det.bbox.y + det.bbox.height);

    cv::Scalar color = RDK_COLORS[det.class_id % RDK_COLORS.size()];
    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    std::string label = COCO_NAMES[det.class_id] + ": " +
                       std::to_string(det.score).substr(0, 4);

    int baseline;
    cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    int label_y = std::max(y1, label_size.height);
    cv::rectangle(img, cv::Point(x1, label_y - label_size.height),
                 cv::Point(x1 + label_size.width, label_y + baseline),
                 color, cv::FILLED);
    cv::putText(img, label, cv::Point(x1, label_y),
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

// ============================================================================
// Segmentation Runtime
// ============================================================================

// One binarized instance mask cropped to its box, in model-input space.
struct InstanceMask {
    cv::Rect roi;
    cv::Mat pixels;  // CV_8U, 255 inside the instance
    int detection_index;
};

// One model context, input and output tensor set. Each benchmark stream owns
// its own runtime; `run` leaves the kept detections and masks for reporting.
class SegmentRuntime {
 public:
  SegmentRuntime(const std::string& model_path, float score_threshold,
                 float nms_threshold)
      : score_threshold_(score_threshold), nms_threshold_(nms_threshold) {
    session_.initialize(model_path);
    outputs_.bind(session_.model(), session_.input_h(), session_.input_w(),
                  true);
    outputs_.allocate();
  }

  int input_h() const { return session_.input_h(); }
  int input_w() const { return session_.input_w(); }
  const char* implementation() const {
    return outputs_.heads.direct_ltrb ? "native_cpp_yolo26_seg_ltrb"
                                      : "native_cpp_yolo_seg_dfl";
  }

  // Timing starts with the in-memory BGR image and ends with binarized
  // instance masks for every NMS-kept detection.
  size_t run(const cv::Mat& image, int resize_type, yolo::StageTiming* timing) {
    const auto start = std::chrono::steady_clock::now();
    yolo::ImageTransform transform;
    preprocessed_ = yolo::preprocess_image(image, input_h(), input_w(),
                                           resize_type, &transform);
    cv::Mat i420;
    cv::cvtColor(preprocessed_, i420, cv::COLOR_BGR2YUV_I420);
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
    return masks_.size();
  }

  // Detection | mask | combined panels in model-input space.
  cv::Mat render() const {
    cv::Mat img_display = preprocessed_.clone();
    cv::Mat mask_overlay = cv::Mat::zeros(input_h(), input_w(), CV_8UC3);
    for (int idx : keep_) draw_detection(img_display, detections_[idx]);
    for (const InstanceMask& mask : masks_) {
      const Detection& det = detections_[mask.detection_index];
      cv::Scalar color = RDK_COLORS[det.class_id % RDK_COLORS.size()];
      cv::Mat color_mask(mask.roi.height, mask.roi.width, CV_8UC3, color);
      cv::Mat masked_color;
      cv::bitwise_and(color_mask, color_mask, masked_color, mask.pixels);
      cv::Mat overlay_roi = mask_overlay(mask.roi);
      cv::addWeighted(overlay_roi, 1.0, masked_color, 0.6, 0, overlay_roi);
    }
    cv::Mat final_result;
    cv::addWeighted(img_display, 0.7, mask_overlay, 0.3, 0, final_result);
    cv::Mat concatenated;
    cv::hconcat(img_display, mask_overlay, concatenated);
    cv::hconcat(concatenated, final_result, concatenated);
    return concatenated;
  }

  const std::vector<Detection>& detections() const { return detections_; }
  const std::vector<int>& keep() const { return keep_; }

 private:
  void decode() {
    const std::vector<yolo::TensorView> views = outputs_.views();
    const float conf_thres_raw = -std::log(1.0f / score_threshold_ - 1.0f);
    detections_.clear();

    // Use the uniquely bound stride-4 NHWC prototype.
    const int proto_h = input_h() / 4;
    const int proto_w = input_w() / 4;
    // The matrix product needs compact (proto_h*proto_w × MCES) rows; view
    // the physical buffer in place when it is already compact.
    const yolo::TensorView& proto = views[outputs_.heads.prototype];
    cv::Mat proto_mat;
    if (proto.cell_step == MCES && proto.row_step == proto_w * MCES) {
      proto_mat = cv::Mat(proto_h * proto_w, MCES, CV_32F,
                          const_cast<float*>(proto.data));
    } else {
      proto_mat.create(proto_h * proto_w, MCES, CV_32F);
      for (int y = 0; y < proto_h; ++y)
        for (int x = 0; x < proto_w; ++x)
          std::memcpy(proto_mat.ptr<float>(y * proto_w + x), proto.cell(y, x),
                      MCES * sizeof(float));
    }

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
      const yolo::TensorView& mce_view = views[outputs_.heads.extra[scale]];

      for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
          const float* cur_cls = cls_view.cell(h, w);
          const float* cur_box = box_view.cell(h, w);
          const float* cur_mce = mce_view.cell(h, w);

          // Find max class score
          int cls_id = 0;
          for (int i = 1; i < CLASSES_NUM; i++)
            if (cur_cls[i] > cur_cls[cls_id]) cls_id = i;
          // Check threshold (before sigmoid)
          yolo::require_finite(cur_cls + cls_id, 1);
          if (cur_cls[cls_id] < conf_thres_raw) continue;
          yolo::require_finite(cur_box, direct_ltrb ? 4 : 4 * REG);
          yolo::require_finite(cur_mce, MCES);
          const float score = 1.0f / (1.0f + std::exp(-cur_cls[cls_id]));

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

          Detection det;
          det.bbox = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
          det.score = score;
          det.class_id = cls_id;
          det.mask_coeffs.assign(cur_mce, cur_mce + MCES);
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

    masks_.clear();
    for (int idx : keep_) {
      const Detection& det = detections_[idx];
      // 1. Matrix multiplication: (proto_h*proto_w × MCES) × (MCES × 1)
      cv::Mat mce_mat(MCES, 1, CV_32F, const_cast<float*>(det.mask_coeffs.data()));
      cv::Mat mask_flat = proto_mat * mce_mat;
      cv::Mat mask_low_res = mask_flat.reshape(1, proto_h);
      if (!cv::checkRange(mask_low_res))
        throw std::runtime_error("Output contains nonfinite values.");
      // 2. Apply sigmoid
      cv::Mat sigmoid_mask;
      cv::exp(-mask_low_res, sigmoid_mask);
      sigmoid_mask = 1.0 / (1.0 + sigmoid_mask);
      // 3. Resize to input size
      cv::Mat resized_mask;
      cv::resize(sigmoid_mask, resized_mask, cv::Size(input_w(), input_h()), 0,
                 0, cv::INTER_LINEAR);
      // 4. Binarize mask
      cv::Mat binary_mask;
      cv::threshold(resized_mask, binary_mask, MASK_THRESHOLD, 1.0,
                    cv::THRESH_BINARY);
      binary_mask.convertTo(binary_mask, CV_8U, 255);
      // 5. Crop mask to bbox region
      const int x1 = std::max(0.0, det.bbox.x);
      const int y1 = std::max(0.0, det.bbox.y);
      const int x2 = std::min(static_cast<double>(input_w()),
                              det.bbox.x + det.bbox.width);
      const int y2 = std::min(static_cast<double>(input_h()),
                              det.bbox.y + det.bbox.height);
      if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;
      InstanceMask mask;
      mask.roi = cv::Rect(x1, y1, x2 - x1, y2 - y1);
      mask.pixels = binary_mask(mask.roi).clone();
      mask.detection_index = idx;
      masks_.push_back(mask);
    }
  }

  // Declared first so outputs are released before the model context.
  yolo::TaskSession session_;
  yolo::TaskOutputs outputs_;
  float score_threshold_;
  float nms_threshold_;
  cv::Mat preprocessed_;
  std::vector<Detection> detections_;
  std::vector<int> keep_;
  std::vector<InstanceMask> masks_;
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
    LOG_INFO("=== Ultralytics YOLO Segmentation Demo (C++) ===");
    LOG_INFO("Loading model: " << command.paths[0]);

    yolo::BenchmarkMeta meta;
    meta.output_kind = "instance_masks";
    meta.timing_scope = "in_memory_bgr_to_instance_masks";
    return yolo::run_task<SegmentRuntime>(
        command, PREPROCESS_TYPE, meta,
        [&command]() {
          return std::unique_ptr<SegmentRuntime>(new SegmentRuntime(
              command.paths[0], command.options.score_threshold,
              command.options.nms_threshold));
        },
        [&command](SegmentRuntime& runtime, const cv::Mat&, int) {
          for (int idx : runtime.keep()) {
            const Detection& det = runtime.detections()[idx];
            LOG_INFO("Detection: " << COCO_NAMES[det.class_id] << ", score="
                                   << std::fixed << std::setprecision(3)
                                   << det.score);
          }
          if (!command.options.save_result) return;
          // Output size: input_h × (input_w * 3)
          const cv::Mat concatenated = runtime.render();
          if (!cv::imwrite(command.paths[2], concatenated))
            throw std::runtime_error("Failed to save segmentation image");
          LOG_INFO("Result saved to: " << command.paths[2]);
          LOG_INFO("Output size: " << concatenated.cols << "x"
                                   << concatenated.rows);
        });
  } catch (const std::exception& error) {
    LOG_ERROR(error.what());
    return 1;
  }
}
