/**
 * @file yolo26_depth.hpp
 * @brief Declare reusable RDK X5 inference APIs for YOLO26 Depth models.
 */

#pragma once

#include <string>

#include <opencv2/core.hpp>

#include "dnn/hb_dnn.h"

namespace yolo26_depth {

/** @brief Store letterbox offsets used to restore source-image geometry. */
struct LetterboxGeometry {
  /** @brief Original image height in pixels. */
  int original_height = 0;
  /** @brief Original image width in pixels. */
  int original_width = 0;
  /** @brief Top padding in pixels. */
  int top = 0;
  /** @brief Bottom padding in pixels. */
  int bottom = 0;
  /** @brief Left padding in pixels. */
  int left = 0;
  /** @brief Right padding in pixels. */
  int right = 0;
};

/** @brief Store model output, restored depth, and measured BPU latency. */
struct InferenceResult {
  /** @brief Raw calibrated log-depth returned by the model. */
  cv::Mat log_depth;
  /** @brief Relative depth restored to the source resolution. */
  cv::Mat depth_native;
  /** @brief Measured BPU inference latency in milliseconds. */
  double latency_ms = 0.0;
  /** @brief Letterbox metadata used for restoration. */
  LetterboxGeometry geometry;
};

/** @brief Own an X5 packed model and execute YOLO26 Depth inference. */
class Yolo26Depth {
 public:
  /**
   * @brief Load one YOLO26 Depth BIN model.
   * @param model_path Path to the X5 ``.bin`` model.
   */
  explicit Yolo26Depth(const std::string& model_path);

  /** @brief Release the packed model and associated runtime resources. */
  ~Yolo26Depth();

  Yolo26Depth(const Yolo26Depth&) = delete;
  Yolo26Depth& operator=(const Yolo26Depth&) = delete;

  /**
   * @brief Run inference and restore relative depth to source resolution.
   * @param image Non-empty BGR ``CV_8UC3`` image.
   * @return Inference output and measured model-execution latency.
   */
  InferenceResult Infer(const cv::Mat& image) const;

  /** @return Square model input size in pixels. */
  int input_size() const { return input_size_; }

  /** @return Packed model name reported by the DNN runtime. */
  const std::string& model_name() const { return model_name_; }

 private:
  hbPackedDNNHandle_t packed_handle_ = nullptr;
  hbDNNHandle_t model_handle_ = nullptr;
  hbDNNTensorProperties input_properties_{};
  hbDNNTensorProperties output_properties_{};
  std::string model_name_;
  int input_size_ = 0;
};

/**
 * @brief Convert float32 relative depth to a Turbo color visualization.
 * @param depth Non-empty ``CV_32FC1`` depth map.
 * @return BGR ``CV_8UC3`` visualization.
 */
cv::Mat ColorizeDepth(const cv::Mat& depth);

}  // namespace yolo26_depth
