/**
 * @file gemma4_vision_engine.hpp
 * @brief Complete Vision path for Gemma4-E2B: preprocessing, the ViT engine
 *        and tensor transport.
 *
 * Loads the Vision HBM and runs the ViT encoder to produce 280 soft image
 * tokens per image, which are injected into the text decoder's input
 * embeddings at image soft-token positions. This header carries the whole
 * vision pipeline in one place: application image IO (LoadImage), the pixel
 * preprocessing stage (PreprocessImage), the fixed-export descriptor
 * contract and strided physical IO (ValidateVisionTensor / WriteVisionInput /
 * ReadVisionOutput), the stage composition (ForwardVision /
 * PostprocessVision / PredictVision), the ViT engine and the optional
 * diagnostics.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "gemma4_config.hpp"
#include "hobot/dnn/hb_dnn.h"

// cv::Mat appears only by reference/return in the declarations below; the
// definition translation unit includes the full OpenCV headers.
namespace cv {
class Mat;
}

namespace gemma4 {

/**
 * @brief Vision ViT inference engine for Gemma4-E2B.
 *
 * Loads the vision HBM model, submits prepared float patches, and returns
 * the ViT output features to be injected into the text decoder.
 */
class VisionEngine {
 public:
  explicit VisionEngine(const std::string& vision_hbm);
  ~VisionEngine();

  VisionEngine(const VisionEngine&) = delete;
  VisionEngine& operator=(const VisionEngine&) = delete;

  std::vector<float> Infer(const std::vector<float>& patches);

  double LoadMs() const { return load_ms_; }

 private:
  void Release() noexcept;
  hbDNNPackedHandle_t packed_ = nullptr;
  hbDNNHandle_t handle_ = nullptr;
  std::vector<hbDNNTensor> inputs_;
  std::vector<hbDNNTensor> outputs_;
  double load_ms_ = 0;
  int64_t input_capacity_ = 0;
  int64_t output_capacity_ = 0;
};

/** @brief Application image IO: decode an image into owned BGR pixels. */
cv::Mat LoadImage(const std::string& path);

// Borrow CV_8UC3 BGR pixels; return an owned [2520,768] RGB float patch
// array. Bicubic resize to 960x672, scale 1/255, 16x16 patches in row-major
// order. No file IO, model load, SDK calls, shared state or mutation of the
// input.
std::vector<float> PreprocessImage(const cv::Mat& bgr);

/** @brief Raw vision runner: prepared patches in, ViT features out. */
using VisionRunner =
    std::function<std::vector<float>(const std::vector<float>&)>;

/**
 * @brief Validate prepared RGB patches and call an explicitly provided raw
 *        runner once.
 */
std::vector<float> ForwardVision(const std::vector<float>& patches,
                                 const VisionRunner& runner);

/** @brief Validate and own the [280,1536] features; no normalization or
 *         rescaling. */
std::vector<float> PostprocessVision(const std::vector<float>& raw);

/**
 * @brief Three-stage vision composition.
 *
 * Image IO and runner construction are caller-owned: decode via LoadImage,
 * preprocess, forward through the runner and validate the output.
 */
std::vector<float> PredictVision(const cv::Mat& bgr,
                                 const VisionRunner& runner);

// ---- Fixed Vision export descriptor contract and strided physical IO ----
// Exact semantic matrix, optional leading singleton axes, nonoverlapping
// byte strides and in-allocation addresses. Capacity is the originally
// allocated size.
void ValidateVisionTensor(const hbDNNTensorProperties& properties, bool input,
                          int64_t capacity = 0);
void WriteVisionInput(hbDNNTensor& tensor, const std::vector<float>& patches,
                      int64_t capacity);
std::vector<float> ReadVisionOutput(const hbDNNTensor& tensor,
                                    int64_t capacity);

/** @brief Optional [DEBUG] statistics dump of a float buffer (stderr). */
inline void LogVisionValues(const char* name,
                            const std::vector<float>& values) {
  if (!RuntimeDebugEnabled() || values.empty())
    return;
  double sum = 0, squares = 0;
  float low = values[0], high = values[0];
  for (float value : values) {
    sum += value;
    squares += static_cast<double>(value) * value;
    low = std::min(low, value);
    high = std::max(high, value);
  }
  const double mean = sum / values.size();
  const double variance = std::max(0.0, squares / values.size() - mean * mean);
  std::cerr << "[DEBUG] " << name << ": size=" << values.size()
            << " min=" << low << " max=" << high << " mean=" << mean
            << " std=" << std::sqrt(variance) << std::endl;
}

/** @brief Optional [DEBUG] dump of a vision tensor descriptor (stderr). */
inline void LogVisionTensor(const char* name, const hbDNNTensorProperties& p) {
  if (!RuntimeDebugEnabled())
    return;
  std::cerr << "[DEBUG] " << name << ": type=" << p.tensorType
            << " ndim=" << p.validShape.numDimensions << " shape=[";
  for (int axis = 0; axis < p.validShape.numDimensions; ++axis) {
    if (axis)
      std::cerr << ",";
    std::cerr << p.validShape.dimensionSize[axis];
  }
  std::cerr << "] aligned_bytes=" << p.alignedByteSize << std::endl;
}

}  // namespace gemma4
