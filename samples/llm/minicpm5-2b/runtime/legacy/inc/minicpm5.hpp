/** @file minicpm5.hpp Single-request MiniCPM5 inference with OELLM 1.0.0. */
#pragma once
#include <string>
#include "xlm.h"
/** Paths and prompt for a fixed W8, chunk-256, context-4096 model. */
struct MiniCPM5Config {
  std::string model_path;
  std::string tokenizer_path;
  std::string template_path;
  std::string prompt = "请用一句话介绍你自己。";
};
/** Owns the SDK handle; each instance serves one synchronous request. */
class MiniCPM5 {
 public:
  /** Store configuration without allocating runtime resources. */
  explicit MiniCPM5(MiniCPM5Config config);
  /** Release the runtime handle. */
  ~MiniCPM5();
  MiniCPM5(const MiniCPM5&) = delete;
  MiniCPM5& operator=(const MiniCPM5&) = delete;
  /** Load the model and tokenizer. Throws on SDK initialization failure. */
  void init();
  /** Stream a single greedy non-thinking response. Return zero on normal EOS. */
  int predict();
 private:
  static void callback(xlm_result_t*, xlm_state_t, void*);
  MiniCPM5Config config_;
  xlm_handle_t handle_ = nullptr;
  bool ended_ = false;
  bool failed_ = false;
};
