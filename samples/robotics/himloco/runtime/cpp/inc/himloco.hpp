/**
 * @file himloco.hpp
 * @brief Declare the reusable HIMLoco RDK X5 C++ Runtime wrapper.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace himloco {

constexpr int kInputElements = 270;
constexpr int kOutputElements = 12;

/** @brief Describe one Runtime tensor without exposing owned SDK pointers. */
struct TensorMetadata {
  std::string name;
  std::vector<int> valid_shape;
  std::vector<int> aligned_shape;
  int tensor_layout = 0;
  int tensor_type = 0;
  int quanti_type = 0;
  int aligned_byte_size = 0;
};

/** @brief Store one policy action and the synchronous DNN-call latency. */
struct InferenceResult {
  std::vector<float> actions;
  double latency_ms = 0.0;
};

/** @brief Own one fused HIMLoco model and its reusable BPU tensor memory. */
class HimLoco {
 public:
  /**
   * @brief Load and validate one RDK X5 HIMLoco BIN model.
   * @param model_path Path to the Bayes-e ``.bin`` model.
   * @param priority DNN task priority in [0,255], or -1 for Runtime default.
   */
  explicit HimLoco(const std::string& model_path, int priority = -1);

  /** @brief Release tensor memory before releasing the packed model. */
  ~HimLoco();

  HimLoco(const HimLoco&) = delete;
  HimLoco& operator=(const HimLoco&) = delete;
  HimLoco(HimLoco&&) noexcept;
  HimLoco& operator=(HimLoco&&) noexcept;

  /**
   * @brief Execute one policy inference.
   * @param observation Exactly 270 finite float32 values.
   * @return Exactly 12 finite float32 actions and DNN-call latency.
   *
   * This method reuses input/output memory and is not thread-safe.
   */
  InferenceResult Infer(const std::vector<float>& observation);

  /** @return Packed model name reported by libdnn. */
  const std::string& model_name() const;

  /** @return libdnn version reported on the board. */
  const std::string& runtime_version() const;

  /** @return Validated input tensor metadata. */
  const TensorMetadata& input_metadata() const;

  /** @return Validated output tensor metadata. */
  const TensorMetadata& output_metadata() const;

  /** @return Requested task priority, or -1 when the Runtime default is used. */
  int priority() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace himloco
