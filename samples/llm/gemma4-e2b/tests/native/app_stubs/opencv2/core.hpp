// Host compile-check stub for OpenCV's core header.
//
// This is NOT OpenCV and decodes nothing. The production chat-model host
// check compiles the real gemma4.cpp / cli.cpp / main.cpp sources, and the
// sdk-resources check compiles the whole vision engine TU (whose folded
// PreprocessImage body uses cv::Mat geometry/typed access plus the color
// conversion and resize calls from the imgproc stub); the real headers live
// in the board image's libopencv-dev. This stub provides the minimal cv
// surface those sources reference. gemma4::LoadImage itself is replaced by a
// double in chat_app_doubles.cpp; nothing here represents image decoding,
// pixel math or board behavior, and stub bodies are never executed.
#pragma once

#include <cstddef>
#include <cstdint>

// Same values as OpenCV; the real macros are also global, and production
// sources reference them unqualified.
enum {
  CV_8UC1 = 0,
  CV_8UC3 = 16,
  CV_32FC3 = 21,
};

namespace cv {

template <class T, int N> class Vec {
 public:
  T& operator[](int i) { return data[i]; }
  const T& operator[](int i) const { return data[i]; }

 private:
  T data[N]{};
};

using Vec3f = Vec<float, 3>;

class Size {
 public:
  Size(int w, int h) : width(w), height(h) {}
  int width = 0;
  int height = 0;
};

class Mat {
 public:
  Mat() = default;
  ~Mat() = default;
  Mat(const Mat&) = default;
  Mat& operator=(const Mat&) = default;
  Mat(int rows, int cols, int type, void* = nullptr)
      : rows_(rows), cols_(cols), type_(type) {}

  bool empty() const { return rows_ == 0 || cols_ == 0; }
  int dims = 2;
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int type() const { return type_; }

  void convertTo(Mat&, int, double = 1.0) const {}

  template <class T> T& at(int, int) {
    static T cell{};
    return cell;
  }
  template <class T> const T& at(int, int) const {
    static T cell{};
    return cell;
  }

 private:
  int rows_ = 0;
  int cols_ = 0;
  int type_ = 0;
};

}  // namespace cv
