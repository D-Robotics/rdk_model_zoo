#include "gemma4_vision_preprocess.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "gemma4_config.h"

namespace gemma4 {

std::vector<float> PreprocessImage(const std::string& image_path) {
  cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("failed to read image: " + image_path);
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
      float* dst = patches.data() +
                   static_cast<size_t>(patch_idx) * kVisionPatchDim;
      int out = 0;
      for (int py = 0; py < kPatchSize; ++py) {
        for (int px = 0; px < kPatchSize; ++px) {
          for (int c = 0; c < 3; ++c) {
            dst[out++] = f32.at<cv::Vec3f>(y * kPatchSize + py,
                                           x * kPatchSize + px)[c];
          }
        }
      }
      ++patch_idx;
    }
  }

  return patches;
}

}  // namespace gemma4
