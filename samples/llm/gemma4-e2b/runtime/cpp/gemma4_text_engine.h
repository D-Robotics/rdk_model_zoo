#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"

#include "gemma4_embeddings.h"
#include "gemma4_kv_cache.h"

namespace gemma4 {

struct BenchmarkResult {
  double load_ms = 0;
  double prefill_ms = 0;
  double decode_ms = 0;
  int decode_steps = 0;
  double tokens_per_sec = 0;
};

struct PrefillChunkTensors {
  std::vector<int64_t> input_ids;
  std::vector<int32_t> position_ids;
  std::vector<float> inputs_embeds;
  std::vector<float> full_mask;
  std::vector<float> sliding_mask;
};

struct ModelIo {
  hbDNNHandle_t handle = nullptr;
  std::vector<hbDNNTensor> inputs;
  std::vector<hbDNNTensor> outputs;
  int seq_len = 0;
};

// Called for each newly generated token id. Return false to stop early.
using TokenCallback = std::function<bool(int64_t token_id)>;

class TextEngine {
 public:
  TextEngine(const std::string& text_hbm, const std::string& embed_path);
  ~TextEngine();

  TextEngine(const TextEngine&) = delete;
  TextEngine& operator=(const TextEngine&) = delete;

  std::vector<int64_t> Generate(const std::vector<int64_t>& prompt_ids,
                                int max_new_tokens);

  // Streaming version: calls on_token for each generated token.
  std::vector<int64_t> GenerateStream(const std::vector<int64_t>& prompt_ids,
                                       int max_new_tokens, TokenCallback on_token);

  std::vector<int64_t> GenerateWithPromptEmbeddings(
      const std::vector<int64_t>& prompt_ids,
      const std::vector<float>& prompt_hidden, int max_new_tokens);

  // Incremental chat: prefill only new suffix, keep KV cache.
  std::vector<int64_t> ContinueGenerate(
      const std::vector<int64_t>& full_ids, int max_new_tokens,
      const std::vector<float>* full_hidden = nullptr);

  // Streaming incremental generation.
  std::vector<int64_t> ContinueGenerateStream(
      const std::vector<int64_t>& full_ids, int max_new_tokens,
      TokenCallback on_token,
      const std::vector<float>* full_hidden = nullptr);

  void ResetSession();

  int ProcessedTokens() const { return processed_tokens_; }

  // Context management for multi-turn chat
  void SetKeepTokens(int n) { n_keep_ = n; }
  int KeepTokens() const { return n_keep_; }

  // Perform context shift: keep first n_keep tokens, discard middle,
  // and compact KV cache. Returns the number of tokens discarded.
  int ContextShift(int n_keep);

  // Check if new tokens would exceed capacity and auto-truncate if needed.
  // Returns true if truncation occurred.
  bool AutoTruncate(int new_prompt_tokens, int max_new_tokens);

  // Chat history management
  void AddToHistory(const std::vector<int64_t>& tokens);
  void ClearHistory();
  const std::vector<int64_t>& GetHistory() const { return chat_history_; }

  std::vector<float> BuildPromptHidden(
      const std::vector<int64_t>& prompt_ids,
      const std::vector<float>& vision_features) const;

  // Build prefill tensors without running BPU (for golden_mask_kv alignment).
  PrefillChunkTensors ExportPrefillChunk(const std::vector<int64_t>& prompt_ids,
                                         int chunk_start,
                                         int chunk_valid) const;

  BenchmarkResult Benchmark(const std::vector<int64_t>& prompt_ids,
                            int max_new_tokens, int warmup_decode = 0);

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
