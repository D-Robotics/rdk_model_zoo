/**
 * @file gemma4_vision_preprocess.cpp
 * @brief Prepare input images for Gemma4-E2B vision inference.
 *
 * The implementation resizes, normalizes, and arranges in-memory image tensors
 * in the layout expected by the compiled vision HBM.
 */

#include "gemma4_vision_preprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "gemma4_config.hpp"

namespace gemma4 {

std::vector<float> PreprocessImage(const cv::Mat &bgr) {
  if (bgr.empty() || bgr.dims != 2 || bgr.type() != CV_8UC3) {
    throw std::invalid_argument(
        "Vision input requires a nonempty 2D CV_8UC3 BGR image");
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(kImageWidth, kImageHeight), 0, 0,
             cv::INTER_CUBIC);

  cv::Mat f32;
  resized.convertTo(f32, CV_32FC3, 1.0 / 255.0);

  std::vector<float> patches(static_cast<size_t>(kVisionPatches) *
                             static_cast<size_t>(kVisionPatchDim));

  const int hp = kImageHeight / kPatchSize;
  const int wp = kImageWidth / kPatchSize;
  if (hp * wp != kVisionPatches) {
    throw std::runtime_error("unexpected patch grid");
  }

  int patch_idx = 0;
  for (int y = 0; y < hp; ++y) {
    for (int x = 0; x < wp; ++x) {
      float *dst =
          patches.data() + static_cast<size_t>(patch_idx) * kVisionPatchDim;
      int out = 0;
      for (int py = 0; py < kPatchSize; ++py) {
        for (int px = 0; px < kPatchSize; ++px) {
          for (int c = 0; c < 3; ++c) {
            dst[out++] =
                f32.at<cv::Vec3f>(y * kPatchSize + py, x * kPatchSize + px)[c];
          }
        }
      }
      ++patch_idx;
    }
  }

  return patches;
}

} // namespace gemma4
