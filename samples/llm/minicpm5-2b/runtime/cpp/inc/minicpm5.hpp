/** @file minicpm5.hpp
 * @brief S600 MiniCPM5 text generation through the supplied OELLM Runtime.
 */
#pragma once

#include <cstdint>
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
  std::vector<int32_t> tokens;  ///< Generated token IDs returned by the runtime.
  int status = 0;  ///< 3: EOS; 4: context limit; 6: output limit.
  double ttft_ms = 0;  ///< Runtime time to first token in milliseconds.
  double decode_tps = 0;  ///< Runtime decode tokens per second.
  double e2e_ms = 0;  ///< Runtime request latency in milliseconds.
};

/** @brief Own a model runtime and one conversation; calls are sequential. */
class MiniCPM5 {
 public:
  /** @brief Load the S600 model and configure four BPU cores.
   * @param config Model directory and output limit.
   * @throws std::runtime_error if model files or runtime initialization fail.
   */
  explicit MiniCPM5(const Config& config);

  /** @brief Generate text with greedy decoding and thinking disabled by the model template.
   * @param prompt Nonempty UTF-8 user prompt.
   * @param new_chat Clear conversation state before this request when true.
   * @return Text, tokens, stop reason and measured latency.
   * @throws std::runtime_error on inference, response or metric errors.
   */
  Result Generate(const std::string& prompt, bool new_chat = true);

 private:
  Config config_;
  oellm::OellmRuntime runtime_;
  int32_t request_id_ = 0;
};
}  // namespace minicpm5
