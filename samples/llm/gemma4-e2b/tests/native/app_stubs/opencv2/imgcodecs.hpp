// Host compile-check stub for OpenCV's imgcodecs header.
//
// This is NOT OpenCV and decodes nothing. The vision engine source
// (src/gemma4_vision_engine.cpp) defines gemma4::LoadImage, which calls
// cv::imread for real on the board; host compiles that link the engine
// without real image decoding use this stub. The suites that exercise
// image loading provide their own double (chat_app_doubles.cpp) or the
// real OpenCV (tests/native vision_test); nothing here represents image
// decoding or board behavior.
#pragma once

#include <string>

#include "opencv2/core.hpp"

namespace cv {

enum { IMREAD_COLOR = 1 };

// Never called in stub-linked suites; inline so no symbol is required.
inline Mat imread(const std::string&, int) { return Mat{}; }

}  // namespace cv
