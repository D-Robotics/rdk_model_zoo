/** @file minicpm5.hpp Single-request MiniCPM5 inference with OELLM 1.0.0. */
#pragma once
#include <string>
#include "xlm.h"
/** Paths and prompt for a fixed W8, chunk-256, context-4096 model. */
struct MiniCPM5Config {
  /** Path to the board-specific W8, chunk-256, context-4096 HBM file. */
  std::string model_path;
  /** Directory containing the prepared OELLM 1.0.0 tokenizer metadata. */
  std::string tokenizer_path;
  /** Path to the prepared non-thinking text Jinja template file. */
  std::string template_path;
  /** UTF-8 user text for one new conversation; defaults to self-introduction. */
  std::string prompt = "请用一句话介绍你自己。";
};
/** Owns the SDK handle; each instance serves one synchronous request. */
class MiniCPM5 {
 public:
  /** Store configuration without allocating runtime resources.
   * @param config Model/tokenizer/template paths and the single-request prompt.
   */
  explicit MiniCPM5(MiniCPM5Config config);
  /** Release the runtime handle. */
  ~MiniCPM5();
  MiniCPM5(const MiniCPM5&) = delete;
  MiniCPM5& operator=(const MiniCPM5&) = delete;
  /** Load the model and tokenizer.
   * @throws std::runtime_error If already initialized or SDK initialization fails.
   */
  void init();
  /** Stream one greedy non-thinking response and release the SDK handle.
   * @return Zero for normal EOS and successful cleanup; one for inference,
   * callback, missing-EOS or cleanup failure.
   * @throws std::runtime_error If uninitialized or the template cannot be opened,
   * is empty, or exceeds 65535 bytes.
   */
  int predict();
 private:
  static void callback(xlm_result_t*, xlm_state_t, void*);
  MiniCPM5Config config_;
  xlm_handle_t handle_ = nullptr;
  bool ended_ = false;
  bool failed_ = false;
};
