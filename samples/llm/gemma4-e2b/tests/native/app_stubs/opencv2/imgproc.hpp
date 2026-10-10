// Host compile-check stub for OpenCV's imgproc header.
//
// This is NOT OpenCV and transforms nothing. The folded vision engine TU
// (src/gemma4_vision_engine.cpp) carries PreprocessImage, whose body calls
// cv::cvtColor / cv::resize for real on the board; host compiles that link
// the engine without real pixel math use this stub so the body compiles.
// Suites that exercise preprocessing provide real OpenCV
// (tests/native vision_test); nothing here represents color conversion,
// resampling or board behavior, and the stub bodies are never executed.
#pragma once

#include "opencv2/core.hpp"

namespace cv {

enum {
  COLOR_BGR2RGB = 4,
  INTER_CUBIC = 2,
};

inline void cvtColor(const Mat&, Mat&, int) {}
inline void resize(const Mat&, Mat&, const Size&, double = 0, double = 0,
                   int = 1) {}

}  // namespace cv
