// Host compile-check stub for OpenCV's core header.
//
// This is NOT OpenCV and decodes nothing. The production chat-app host check
// compiles the real gemma4_chat_app.cpp / main.cpp sources; gemma4_image_io
// and gemma4_vision_* headers include <opencv2/core.hpp> for cv::Mat, whose
// real copy lives in the board image's libopencv-dev. This stub provides the
// minimal cv::Mat type those headers reference. gemma4::LoadImage itself is
// replaced by a double in chat_app_doubles.cpp; nothing here represents
// image decoding or board behavior.
#pragma once

#include <cstddef>
#include <cstdint>

namespace cv {

class Mat {
 public:
  Mat() = default;
  ~Mat() = default;
  Mat(const Mat&) = default;
  Mat& operator=(const Mat&) = default;
};

}  // namespace cv
