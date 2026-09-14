/**
 * @file yolo26_depth.cpp
 * @brief Implement preprocessing, X5 inference, and depth restoration.
 */

#include "yolo26_depth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "dnn/hb_sys.h"

namespace yolo26_depth {
namespace {

/** @brief Throw a runtime error when a DNN or system call fails. */
void Check(int code, const std::string& message) {
  if (code != 0) {
    throw std::runtime_error(message + ", error code=" + std::to_string(code));
  }
}

/** @brief Manage one cached BPU system-memory allocation. */
class CachedMemory {
 public:
  explicit CachedMemory(int size) {
    Check(hbSysAllocCachedMem(&memory_, size), "hbSysAllocCachedMem failed");
  }

  ~CachedMemory() { hbSysFreeMem(&memory_); }

  CachedMemory(const CachedMemory&) = delete;
  CachedMemory& operator=(const CachedMemory&) = delete;

  hbSysMem& get() { return memory_; }

 private:
  hbSysMem memory_{};
};

/** @brief Release a DNN task handle when leaving scope. */
class TaskHandle {
 public:
  TaskHandle() = default;

  ~TaskHandle() {
    if (handle != nullptr) {
      hbDNNReleaseTask(handle);
    }
  }

  TaskHandle(const TaskHandle&) = delete;
  TaskHandle& operator=(const TaskHandle&) = delete;

  hbDNNTaskHandle_t handle = nullptr;
};

/** @brief Convert an even-sized BGR image to packed NV12 bytes. */
std::vector<std::uint8_t> BgrToNv12(const cv::Mat& image) {
  cv::Mat i420;
  cv::cvtColor(image, i420, cv::COLOR_BGR2YUV_I420);
  const int area = image.rows * image.cols;
  const std::uint8_t* source = i420.ptr<std::uint8_t>();
  const std::uint8_t* u_plane = source + area;
  const std::uint8_t* v_plane = u_plane + area / 4;
  std::vector<std::uint8_t> nv12(area * 3 / 2);
  std::memcpy(nv12.data(), source, area);
  std::uint8_t* uv_plane = nv12.data() + area;
  for (int index = 0; index < area / 4; ++index) {
    uv_plane[index * 2] = u_plane[index];
    uv_plane[index * 2 + 1] = v_plane[index];
  }
  return nv12;
}

/** @brief Apply calibration-compatible letterbox resizing. */
cv::Mat Letterbox(const cv::Mat& image, int size, LetterboxGeometry* geometry) {
  const double ratio = std::min(
      static_cast<double>(size) / image.rows,
      static_cast<double>(size) / image.cols);
  const int resized_width = static_cast<int>(std::round(image.cols * ratio));
  const int resized_height = static_cast<int>(std::round(image.rows * ratio));
  const int pad_width = size - resized_width;
  const int pad_height = size - resized_height;
  geometry->original_height = image.rows;
  geometry->original_width = image.cols;
  geometry->left = static_cast<int>(std::round(pad_width / 2.0 - 0.1));
  geometry->right = static_cast<int>(std::round(pad_width / 2.0 + 0.1));
  geometry->top = static_cast<int>(std::round(pad_height / 2.0 - 0.1));
  geometry->bottom = static_cast<int>(std::round(pad_height / 2.0 + 0.1));

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
  cv::Mat padded;
  cv::copyMakeBorder(
      resized,
      padded,
      geometry->top,
      geometry->bottom,
      geometry->left,
      geometry->right,
      cv::BORDER_CONSTANT,
      cv::Scalar(114, 114, 114));
  return padded;
}

/** @brief Read output height and width from tensor properties. */
std::pair<int, int> OutputHeightWidth(const hbDNNTensorProperties& properties) {
  const hbDNNTensorShape& shape = properties.validShape;
  if (shape.numDimensions != 4) {
    throw std::runtime_error("expected a four-dimensional depth output");
  }
  if (shape.dimensionSize[3] == 1) {
    return {shape.dimensionSize[1], shape.dimensionSize[2]};
  }
  if (shape.dimensionSize[1] == 1) {
    return {shape.dimensionSize[2], shape.dimensionSize[3]};
  }
  throw std::runtime_error("cannot identify the single-channel depth output");
}

}  // namespace

Yolo26Depth::Yolo26Depth(const std::string& model_path) {
  const char* model_file = model_path.c_str();
  Check(hbDNNInitializeFromFiles(&packed_handle_, &model_file, 1),
        "hbDNNInitializeFromFiles failed");
  try {
    const char** names = nullptr;
    int model_count = 0;
    Check(hbDNNGetModelNameList(&names, &model_count, packed_handle_),
          "hbDNNGetModelNameList failed");
    if (model_count != 1) {
      throw std::runtime_error("this sample expects exactly one packed model");
    }
    model_name_ = names[0];
    Check(hbDNNGetModelHandle(&model_handle_, packed_handle_, names[0]),
          "hbDNNGetModelHandle failed");

    int input_count = 0;
    int output_count = 0;
    Check(hbDNNGetInputCount(&input_count, model_handle_), "hbDNNGetInputCount failed");
    Check(hbDNNGetOutputCount(&output_count, model_handle_), "hbDNNGetOutputCount failed");
    if (input_count != 1 || output_count != 1) {
      throw std::runtime_error("this sample expects one input and one output");
    }

    Check(hbDNNGetInputTensorProperties(&input_properties_, model_handle_, 0),
          "hbDNNGetInputTensorProperties failed");
    Check(hbDNNGetOutputTensorProperties(&output_properties_, model_handle_, 0),
          "hbDNNGetOutputTensorProperties failed");
    if (input_properties_.tensorType != HB_DNN_IMG_TYPE_NV12) {
      throw std::runtime_error("model input is not NV12");
    }
    if (input_properties_.validShape.numDimensions != 4 ||
        input_properties_.validShape.dimensionSize[2] !=
            input_properties_.validShape.dimensionSize[3]) {
      throw std::runtime_error("model input is not a square 4D image tensor");
    }
    if (output_properties_.tensorType != HB_DNN_TENSOR_TYPE_F32 ||
        output_properties_.quantiType != NONE) {
      throw std::runtime_error("model output is not dequantized float32");
    }
    input_size_ = input_properties_.validShape.dimensionSize[2];
  } catch (...) {
    hbDNNRelease(packed_handle_);
    packed_handle_ = nullptr;
    throw;
  }
}

Yolo26Depth::~Yolo26Depth() {
  if (packed_handle_ != nullptr) {
    hbDNNRelease(packed_handle_);
  }
}

InferenceResult Yolo26Depth::Infer(const cv::Mat& image) const {
  if (image.empty() || image.type() != CV_8UC3) {
    throw std::invalid_argument("input image must be a non-empty BGR CV_8UC3 matrix");
  }

  InferenceResult result;
  const cv::Mat padded = Letterbox(image, input_size_, &result.geometry);
  const std::vector<std::uint8_t> nv12 = BgrToNv12(padded);

  CachedMemory input_memory(input_properties_.alignedByteSize);
  hbDNNTensor input{};
  input.properties = input_properties_;
  input.sysMem[0] = input_memory.get();
  std::memcpy(input.sysMem[0].virAddr, nv12.data(), nv12.size());
  Check(hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN),
        "hbSysFlushMem input failed");

  CachedMemory output_memory(output_properties_.alignedByteSize);
  hbDNNTensor output{};
  output.properties = output_properties_;
  output.sysMem[0] = output_memory.get();

  TaskHandle task;
  hbDNNInferCtrlParam control;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&control);
  const auto started = std::chrono::steady_clock::now();
  hbDNNTensor* output_pointer = &output;
  const int infer_code =
      hbDNNInfer(&task.handle, &output_pointer, &input, model_handle_, &control);
  if (infer_code == 0) {
    Check(hbDNNWaitTaskDone(task.handle, 0), "hbDNNWaitTaskDone failed");
  }
  const auto stopped = std::chrono::steady_clock::now();
  result.latency_ms =
      std::chrono::duration<double, std::milli>(stopped - started).count();
  Check(infer_code, "hbDNNInfer failed");
  Check(hbSysFlushMem(&output.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE),
        "hbSysFlushMem output failed");

  const auto [output_height, output_width] = OutputHeightWidth(output_properties_);
  const float* output_data = static_cast<const float*>(output.sysMem[0].virAddr);
  result.log_depth = cv::Mat(output_height, output_width, CV_32FC1,
                             const_cast<float*>(output_data)).clone();

  cv::Mat depth_small;
  cv::exp(result.log_depth, depth_small);
  cv::Mat depth_square;
  cv::resize(depth_small, depth_square, cv::Size(input_size_, input_size_),
             0.0, 0.0, cv::INTER_LINEAR);
  const cv::Rect content(
      result.geometry.left,
      result.geometry.top,
      input_size_ - result.geometry.left - result.geometry.right,
      input_size_ - result.geometry.top - result.geometry.bottom);
  const cv::Mat cropped = depth_square(content);
  cv::resize(cropped, result.depth_native,
             cv::Size(result.geometry.original_width, result.geometry.original_height),
             0.0, 0.0, cv::INTER_LINEAR);

  return result;
}

cv::Mat ColorizeDepth(const cv::Mat& depth) {
  if (depth.empty() || depth.type() != CV_32FC1) {
    throw std::invalid_argument("depth must be a non-empty CV_32FC1 matrix");
  }
  std::vector<float> values(depth.begin<float>(), depth.end<float>());
  std::sort(values.begin(), values.end());
  const float low = values[static_cast<std::size_t>(values.size() * 0.02)];
  const float high = values[static_cast<std::size_t>(values.size() * 0.98)];
  cv::Mat normalized;
  depth.convertTo(normalized, CV_32FC1, 1.0 / std::max(high - low, 1e-6f),
                  -low / std::max(high - low, 1e-6f));
  cv::min(normalized, 1.0, normalized);
  cv::max(normalized, 0.0, normalized);
  cv::Mat gray;
  normalized.convertTo(gray, CV_8UC1, -255.0, 255.0);
  cv::Mat color;
  cv::applyColorMap(gray, color, cv::COLORMAP_TURBO);
  return color;
}

}  // namespace yolo26_depth
