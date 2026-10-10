/** @file minicpm5.hpp
 * @brief S600 MiniCPM5 text generation through the supplied OELLM Runtime.
 *
 * Model validation and OELLM runtime configuration handling are part of the
 * model implementation in src/minicpm5.cpp.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "oellm_runtime_basic/oellm_runtime.h"

namespace minicpm5 {

/** @brief Model location and bounded generation settings. */
struct Config {
  std::string model_path = "../../model/s600";  ///< Extracted model directory.
  int32_t max_new_tokens = 128;  ///< Per-request output limit, from 1 to 4096.
};

/** @brief Generated text, termination reason and runtime measurements. */
struct Result {
  std::string text;  ///< Generated UTF-8 text, excluding the EOS marker.
  std::vector<int32_t>
      tokens;             ///< Generated token IDs returned by the runtime.
  int status = 0;         ///< 3: EOS; 4: context limit; 6: output limit.
  double ttft_ms = 0;     ///< Runtime time to first token in milliseconds.
  double decode_tps = 0;  ///< Runtime decode tokens per second.
  double e2e_ms = 0;      ///< Runtime request latency in milliseconds.
};

/** @brief Verify the required model files inside an extracted directory.
 * @param model_directory Candidate extracted S600 model directory.
 * @return The canonical directory path used for runtime configuration.
 * @throws std::runtime_error if the path is missing any required file.
 */
std::filesystem::path validate_model_directory(
    const std::filesystem::path& model_directory);

/** @brief Write the runtime JSON configuration and initialize the runtime.
 *
 * The temporary configuration file is owned by an RAII guard, so it is
 * removed on success, SDK error return and exception paths alike.
 * @param model_directory Extracted S600 model directory.
 * @param runtime Uninitialized OELLM runtime instance.
 * @throws std::runtime_error on temporary-file or SDK initialization failure.
 */
void configure_runtime(const std::filesystem::path& model_directory,
                       oellm::OellmRuntime& runtime);

/** @brief Reject non-finite or negative runtime measurements in
 * post-processing.
 *
 * Zero stays valid: a one-token length-limited request legitimately reports
 * decode_tps == 0. Invalid values are rejected instead of being coerced to
 * zero, so the CLI JSON can never present an invalid measurement as success.
 * @param ttft_ms Time to first token in milliseconds.
 * @param decode_tps Decode tokens per second.
 * @param e2e_ms End-to-end request latency in milliseconds.
 * @throws std::runtime_error naming the first invalid measurement.
 */
void validate_metrics(double ttft_ms, double decode_tps, double e2e_ms);

/** @brief Build one OELLM request; the preprocess stage makes no SDK calls.
 *
 * Tokenization and template rendering happen inside the runtime; the public
 * preprocess boundary is the request structure the runtime accepts.
 * @param prompt Nonempty UTF-8 user prompt.
 * @param new_chat True to clear conversation state for this request.
 * @param max_new_tokens Per-request output limit, from 1 to 4096; enforced
 * here for every caller, independent of the constructor's own check.
 * @param request_id Sequential identifier starting at one.
 * @return Request with conversation_id 1 and a text prompt.
 * @throws std::runtime_error if the prompt is empty or the output limit is
 * out of range.
 */
oellm::OellmRequest preprocess(const std::string& prompt, bool new_chat,
                               int32_t max_new_tokens, int32_t request_id);

/** @brief Run one synchronous inference and validate the response shape.
 * @param runtime Initialized OELLM runtime.
 * @param request Request built by preprocess.
 * @param[out] response Receives exactly one response data payload.
 * @throws std::runtime_error on SDK errors or a malformed response.
 */
void infer(oellm::OellmRuntime& runtime, const oellm::OellmRequest& request,
           oellm::OellmResponse& response);

/** @brief Post-process one validated response into text, status and metrics.
 * @param response Response validated by infer; must hold exactly one payload.
 * @param runtime Initialized runtime, used for the request-metrics API.
 * @return Generated text, tokens, stop reason and validated measurements.
 * @throws std::runtime_error on unexpected generation status or invalid
 * metrics; see validate_metrics for the measurement rules.
 */
Result postprocess(const oellm::OellmResponse& response,
                   oellm::OellmRuntime& runtime);

/** @brief Own a model runtime and one conversation; calls are sequential. */
class MiniCPM5 {
 public:
  /** @brief Validate settings, then load the S600 model on four BPU cores.
   * @param config Model directory and output limit.
   * @throws std::runtime_error if settings, model files or runtime
   * initialization fail; see validate_model_directory/configure_runtime for
   * the file handling.
   */
  explicit MiniCPM5(const Config& config);

  /** @brief Generate text with greedy decoding and thinking disabled by the
   * model template; the predict chain of preprocess, infer and postprocess.
   * @param prompt Nonempty UTF-8 user prompt.
   * @param new_chat Clear conversation state before this request when true.
   * @return Text, tokens, stop reason and measured latency.
   * @throws std::runtime_error on inference, response or metric errors.
   */
  Result predict(const std::string& prompt, bool new_chat = true);

 private:
  Config config_;
  oellm::OellmRuntime runtime_;
  int32_t request_id_ = 0;
};
}  // namespace minicpm5
