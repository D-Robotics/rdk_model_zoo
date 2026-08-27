/**
 * @file himloco.hpp
 * @brief Declare the reusable HIMLoco RDK X5 C++ Runtime interface.
 *
 * The interface separates model initialization, preprocessing, BPU inference,
 * and postprocessing. One instance owns one model and is not thread-safe.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace himloco {

constexpr int kInputElements = 270;   ///< Number of float32 observation values.
constexpr int kOutputElements = 12;   ///< Number of float32 policy actions.

/**
 * @brief Configure model initialization and DNN task scheduling.
 *
 * Defaults match the repository layout when execution starts from
 * ``runtime/cpp``.
 */
struct HimLocoConfig {
  std::string model_path =
      "../../model/bayes-e/himloco_go2_bayese_1x270.bin";  ///< X5 model path.
  int priority = -1;  ///< DNN priority in [0,255], or -1 for Runtime default.
};

/**
 * @brief Describe one Runtime tensor without exposing owned SDK pointers.
 *
 * The metadata is populated during ``init()`` and remains valid for the
 * lifetime of the initialized model.
 */
struct TensorMetadata {
  std::string name;               ///< Tensor name reported by libdnn.
  std::vector<int> valid_shape;   ///< Logical Runtime tensor shape.
  std::vector<int> aligned_shape; ///< Physical Runtime tensor shape.
  int tensor_layout = 0;          ///< libdnn tensor layout enum value.
  int tensor_type = 0;            ///< libdnn tensor dtype enum value.
  int quanti_type = 0;            ///< libdnn quantization enum value.
  int aligned_byte_size = 0;      ///< Allocated tensor size in bytes.
};

/**
 * @brief Store one policy action and synchronous DNN-call latency.
 *
 * Actions contain exactly 12 finite float32 values after successful
 * postprocessing.
 */
struct InferenceResult {
  std::vector<float> actions;  ///< Postprocessed policy actions.
  double latency_ms = 0.0;     ///< BPU inference latency in milliseconds.
};

/**
 * @brief Own one fused HIMLoco model and its reusable BPU tensor memory.
 *
 * Construction performs no model loading or tensor allocation. Call
 * ``init()`` explicitly before inference.
 *
 * @note The object is not thread-safe.
 */
class HimLoco {
 public:
  /** @brief Create an uninitialized HIMLoco Runtime wrapper. */
  HimLoco();

  /** @brief Release tensor memory and the packed model when initialized. */
  ~HimLoco();

  HimLoco(const HimLoco&) = delete;
  HimLoco& operator=(const HimLoco&) = delete;

  /** @brief Move one Runtime wrapper without duplicating owned resources. */
  HimLoco(HimLoco&&) noexcept;

  /** @brief Replace owned resources by moving another Runtime wrapper. */
  HimLoco& operator=(HimLoco&&) noexcept;

  /**
   * @brief Load the model, validate metadata, and allocate reusable tensors.
   *
   * @param[in] config Model path and optional DNN task priority.
   * @retval 0 Success.
   * @retval -1 Initialization or validation failed; inspect ``last_error()``.
   */
  int init(const HimLocoConfig& config) noexcept;

  /**
   * @brief Update the DNN task priority used by later inference calls.
   *
   * @param[in] priority Priority in [0,255], or -1 for Runtime default.
   * @retval 0 Success.
   * @retval -1 Priority is outside the supported range.
   */
  int set_scheduling_params(int priority) noexcept;

  /**
   * @brief Validate and copy one observation into a logical input tensor.
   *
   * @param[in] observation Exactly 270 finite float32 observation values.
   * @param[out] input_tensor Validated logical input tensor.
   * @retval 0 Success.
   * @retval -1 Input validation failed.
   */
  int pre_process(const std::vector<float>& observation,
                  std::vector<float>& input_tensor) const noexcept;

  /**
   * @brief Submit one logical input tensor to BPU and read raw output.
   *
   * @param[in] input_tensor Exactly 270 finite float32 input values.
   * @param[out] output_tensor Exactly 12 raw float32 output values.
   * @param[out] latency_ms Synchronous DNN execution latency in milliseconds.
   * @retval 0 Success.
   * @retval -1 Runtime inference failed; inspect ``last_error()``.
   */
  int infer(const std::vector<float>& input_tensor,
            std::vector<float>& output_tensor, double& latency_ms) noexcept;

  /**
   * @brief Convert raw model output into policy actions.
   *
   * @param[in] output_tensor Exactly 12 finite float32 raw actions.
   * @param[in] latency_ms Synchronous DNN execution latency in milliseconds.
   * @param[out] result Postprocessed policy actions and latency.
   * @retval 0 Success.
   * @retval -1 Output validation failed.
   */
  int post_process(const std::vector<float>& output_tensor, double latency_ms,
                   InferenceResult& result) const noexcept;

  /**
   * @brief Run preprocessing, BPU inference, and postprocessing.
   *
   * @param[in] observation Exactly 270 finite float32 observation values.
   * @param[out] result Postprocessed policy actions and latency.
   * @retval 0 Success.
   * @retval -1 A pipeline stage failed; inspect ``last_error()``.
   */
  int predict(const std::vector<float>& observation,
              InferenceResult& result) noexcept;

  /** @return Whether ``init()`` completed successfully. */
  bool initialized() const noexcept;

  /** @return Last public API error, or an empty string after success. */
  const std::string& last_error() const noexcept;

  /** @return Packed model name reported by libdnn. */
  const std::string& model_name() const noexcept;

  /** @return libdnn version reported on the board. */
  const std::string& runtime_version() const noexcept;

  /** @return Validated input tensor metadata. */
  const TensorMetadata& input_metadata() const noexcept;

  /** @return Validated output tensor metadata. */
  const TensorMetadata& output_metadata() const noexcept;

  /** @return Requested task priority, or -1 for Runtime default. */
  int priority() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< Heavy Runtime resources created by init().
  mutable std::string last_error_;  ///< Last validation or Runtime failure.
};

}  // namespace himloco
