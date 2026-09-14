/**
 * @file gemma4_text_engine.hpp
 * @brief Text LLM engine for Gemma4-E2B (prefill + decode + KV cache).
 *
 * Wraps the Horizon BPU DNN APIs to run the Gemma4-E2B text decoder on
 * supported RDK S targets. Manages prefill (chunked) and decode steps, zero-copy KV cache,
 * and logits→token sampling.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"

#include "gemma4_embeddings.hpp"
#include "gemma4_kv_cache.hpp"

namespace gemma4 {

/**
 * @brief Benchmark timing result for a single prompt run.
 */
struct BenchmarkResult {
  double load_ms = 0;          ///< Model load time (ms)
  double prefill_ms = 0;       ///< Prefill time (ms)
  double decode_ms = 0;        ///< Decode time (ms, sum of all steps)
  int decode_steps = 0;        ///< Number of decode steps
  double tokens_per_sec = 0;   ///< Decode throughput (tokens/sec)
};

/**
 * @brief Hold one exported prefill chunk for conversion/runtime verification.
 */
struct PrefillChunkTensors {
  std::vector<int64_t> input_ids;
  std::vector<int32_t> position_ids;
  std::vector<float> inputs_embeds;
  std::vector<float> full_mask;
  std::vector<float> sliding_mask;
};

/**
 * @brief Own the DNN handle and tensors for one compiled text subgraph.
 */
struct ModelIo {
  hbDNNHandle_t handle = nullptr;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  int seq_len = 0;
};

// Called for each newly generated token id. Return false to stop early.
using TokenCallback = std::function<bool(int64_t token_id)>;

/**
 * @brief Run Gemma4-E2B text generation with reusable KV-cache state.
 *
 * A TextEngine owns the prefill/decode models, embedding table, and one chat
 * session. Callers must serialize access to an instance.
 */
class TextEngine {
 public:
  TextEngine(const std::string& text_hbm, const std::string& embed_path);
  ~TextEngine();

  TextEngine(const TextEngine&) = delete;
  TextEngine& operator=(const TextEngine&) = delete;

  /**
   * @brief Run one-shot greedy text generation for a prompt.
   *
   * @param prompt_ids Token IDs of the prompt.
   * @param max_new_tokens Maximum number of new tokens to generate.
   *
   * @return Generated token IDs (excluding the prompt).
   */
  std::vector<int64_t> Generate(const std::vector<int64_t>& prompt_ids,
                                int max_new_tokens);

  /**
   * @brief Generate text with a per-token streaming callback.
   *
   * @param prompt_ids Token IDs of the prompt.
   * @param max_new_tokens Maximum number of new tokens to generate.
   * @param on_token Callback invoked with each newly generated token ID.
   *        Returning false stops generation early.
   *
   * @return Generated token IDs (excluding the prompt).
   */
  std::vector<int64_t> GenerateStream(const std::vector<int64_t>& prompt_ids,
                                       int max_new_tokens, TokenCallback on_token);

  /**
   * @brief Generate text starting from prebuilt prompt hidden states.
   *
   * @param prompt_ids Token IDs of the prompt.
   * @param prompt_hidden Prebuilt inputs_embeds for the prompt.
   * @param max_new_tokens Maximum number of new tokens to generate.
   *
   * @return Generated token IDs (excluding the prompt).
   */
  std::vector<int64_t> GenerateWithPromptEmbeddings(
      const std::vector<int64_t>& prompt_ids,
      const std::vector<float>& prompt_hidden, int max_new_tokens);

  /**
   * @brief Incremental chat generation reusing the existing KV cache.
   *
   * Prefills only the new suffix of @p full_ids and keeps the previous KV
   * cache intact, enabling multi-turn chat without re-encoding history.
   *
   * @param full_ids Full token sequence including prior context.
   * @param max_new_tokens Maximum number of new tokens to generate.
   * @param full_hidden Optional prebuilt hidden states for the suffix.
   *
   * @return Generated token IDs.
   */
  std::vector<int64_t> ContinueGenerate(
      const std::vector<int64_t>& full_ids, int max_new_tokens,
      const std::vector<float>* full_hidden = nullptr);

  /**
   * @brief Streaming variant of @ref ContinueGenerate.
   *
   * @param full_ids Full token sequence including prior context.
   * @param max_new_tokens Maximum number of new tokens to generate.
   * @param on_token Per-token streaming callback.
   * @param full_hidden Optional prebuilt hidden states for the suffix.
   *
   * @return Generated token IDs.
   */
  std::vector<int64_t> ContinueGenerateStream(
      const std::vector<int64_t>& full_ids, int max_new_tokens,
      TokenCallback on_token,
      const std::vector<float>* full_hidden = nullptr);

  /// Clear all KV-cache and session state.
  void ResetSession();

  /// Number of tokens currently processed and held in the KV cache.
  int ProcessedTokens() const { return processed_tokens_; }

  // Context management for multi-turn chat
  /// Set the number of leading tokens preserved during a context shift.
  void SetKeepTokens(int n) { n_keep_ = n; }
  /// Number of leading tokens preserved during a context shift.
  int KeepTokens() const { return n_keep_; }

  /**
   * @brief Compact the KV cache to make room for new context.
   *
   * Keeps the first @p n_keep tokens, discards the middle, and compacts the
   * KV cache so generation can continue without exceeding capacity.
   *
   * @param n_keep Number of leading tokens to preserve.
   *
   * @return Number of tokens discarded.
   */
  int ContextShift(int n_keep);

  /**
   * @brief Check whether new tokens would exceed KV capacity.
   *
   * Auto-truncates the pending input if needed so generation stays within
   * the fixed cache length.
   *
   * @param new_prompt_tokens Number of prompt tokens about to be added.
   * @param max_new_tokens Requested output length.
   *
   * @return True if truncation occurred.
   */
  bool AutoTruncate(int new_prompt_tokens, int max_new_tokens);

  // Chat history management
  /// Append a full turn (e.g. user+assistant tokens) to chat history.
  void AddToHistory(const std::vector<int64_t>& tokens);
  /// Clear the chat history used for truncation decisions.
  void ClearHistory();
  /// Full chat history accumulated for truncation decisions.
  const std::vector<int64_t>& GetHistory() const { return chat_history_; }

  /**
   * @brief Build prompt hidden states by injecting vision features.
   *
   * @param prompt_ids Token IDs of the text prompt.
   * @param vision_features Vision soft-token features to inject.
   *
   * @return inputs_embeds for the full prompt.
   */
  std::vector<float> BuildPromptHidden(
      const std::vector<int64_t>& prompt_ids,
      const std::vector<float>& vision_features) const;

  /**
   * @brief Export one prefill chunk's tensors without running the BPU.
   *
   * Used for golden mask/KV alignment verification against PC-side data.
   *
   * @param prompt_ids Full prompt token IDs.
   * @param chunk_start Start offset of the chunk within the prompt.
   * @param chunk_valid Number of valid tokens in the chunk.
   *
   * @return The exported chunk tensors.
   */
  PrefillChunkTensors ExportPrefillChunk(const std::vector<int64_t>& prompt_ids,
                                         int chunk_start,
                                         int chunk_valid) const;

  /**
   * @brief Benchmark prefill and decode latency for a prompt.
   *
   * @param prompt_ids Token IDs of the prompt.
   * @param max_new_tokens Number of decode steps to time.
   * @param warmup_decode Decode warmup steps performed before timing.
   *
   * @return Benchmark timing result.
   */
  BenchmarkResult Benchmark(const std::vector<int64_t>& prompt_ids,
                            int max_new_tokens, int warmup_decode = 0);

  /// Model load time in milliseconds.
  double LoadMs() const { return load_ms_; }

 private:
  static bool IsEos(int64_t token_id);

  void SetupZeroCopyKv();
  void FillCommonInputs(ModelIo& io, const std::vector<int64_t>& token_ids,
                        int chunk_start, int chunk_valid, bool decode,
                        const float* prebuilt_hidden = nullptr);
  void FillDecodeInputs(int64_t token_id, int pos);
  void RunPrefillChunk(const std::vector<int64_t>& chunk, int chunk_start,
                       const float* prebuilt_hidden = nullptr);
  void PrefillSuffix(const std::vector<int64_t>& ids, int start,
                     const std::vector<float>* hidden = nullptr);
  int64_t RunDecodeStep(int64_t token_id);
  int64_t ArgmaxLogits(const hbDNNTensor& logits_tensor, int seq_idx);
  static int64_t OutputRowStrideBytes(const hbDNNTensor& tensor);

  static void BuildFullMask(const KvCache& kv, float* mask, int cache_start,
                            int chunk_start, int chunk_valid, int seq_len);
  static void BuildSlidingMask(const KvCache& kv, float* mask, int cache_start,
                               int chunk_start, int chunk_valid, int seq_len);
  static void QuantizeMask(const float* mask_f32, int16_t* mask_i16, int rows,
                           int cols);

  hbDNNPackedHandle_t packed_ = nullptr;
  ModelIo prefill_;
  ModelIo decode_;
  TokenEmbeddings embeddings_;
  KvCache kv_;
  int token_offset_ = 0;
  int processed_tokens_ = 0;
  int n_keep_ = 0;  // tokens to preserve during context shift (system prompt)
  std::vector<int64_t> chat_history_;  // full chat history for truncation
  double load_ms_ = 0;

  std::vector<float> decode_hidden_;
  std::vector<float> decode_mask_;
  std::vector<float> decode_slide_mask_;
  std::vector<int16_t> decode_mask_q_;
  std::vector<int16_t> decode_slide_mask_q_;
};

}  // namespace gemma4
