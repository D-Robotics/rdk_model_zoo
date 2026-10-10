/**
 * @file gemma4_text_engine.hpp
 * @brief Text LLM engine for Gemma4-E2B (prefill + decode + KV cache).
 *
 * One coherent engine owns the whole Text pipeline for the Horizon BPU DNN
 * APIs. Its named pieces cover each stage of one subgraph invocation:
 * input preparation (PrepareBatchInputs / PrepareDecodeInputs and the mask
 * builders), the fixed-export descriptor contract and strided physical IO
 * (ValidateTextTensor / WriteTextInput / TextKvOutputRows /
 * ArgmaxTextLogits), raw SDK transport (InitTextSubgraph / BindKvCache /
 * WriteBatchInputs / RunSubgraphInference / CollectKvOutputs), and the
 * orchestration that decodes logits and updates the KV cache/session for
 * chunked prefill, greedy decode and reusable conversation prefixes.
 * Multi-turn continuation, history and context-shift policy live in
 * gemma4_text_session; the engine only sequences them.
 *
 * The library prints nothing implicitly — install a debug sink with
 * SetDebugSink to receive diagnostics.
 */
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"

#include "gemma4_config.hpp"
#include "gemma4_embeddings.hpp"
#include "gemma4_kv_cache.hpp"
#include "gemma4_model_io.hpp"
#include "gemma4_text_session.hpp"

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

// Called for each newly generated token id. Return false to stop early.
using TokenCallback = std::function<bool(int64_t token_id)>;

/// Receiver for engine diagnostics; never invoked unless installed.
using DebugSink = std::function<void(const std::string& message)>;

// ---- Stage 1: prepared CPU inputs (no SDK types, no printing) ----

/// Prepared per-call context for one subgraph invocation.
struct TextBatchInputs {
  std::vector<int64_t> token_ids;    ///< PLE-substituted ids, seq elements.
  std::vector<float> hidden;         ///< inputs_embeds rows, seq*kHiddenSize.
  std::vector<int32_t> positions;    ///< Position ids, seq elements.
  std::vector<int16_t> full_mask_q;  ///< Quantized full mask, seq*kCacheLen.
  std::vector<int16_t> slide_mask_q; ///< Quantized sliding mask, seq*kCacheLen.
};

/**
 * @brief Build the full-attention mask with the source's right-aligned
 * layout.
 *
 * After a chunk covering [chunk_start, chunk_start+chunk_valid): row r
 * (clamped to the last seen token for pad rows) attends to
 * [cache_start_abs .. query_abs]. Validates the geometry contract below and
 * throws std::invalid_argument before writing when it is violated; @p mask
 * must have room for seq_len*kCacheLen elements.
 */
void BuildFullMask(float* mask, int chunk_start, int chunk_valid, int seq_len);

/**
 * @brief Build the sliding-window mask (source layout).
 *
 * Same geometry contract as @ref BuildFullMask; rows additionally start at
 * query_abs-kSlidingWindow+1.
 */
void BuildSlidingMask(float* mask, int chunk_start, int chunk_valid,
                      int seq_len);

/**
 * @brief Quantize float masks to int16 with the source round+clamp
 * algorithm.
 *
 * Rejects negative @p rows/@p cols and counts that cannot be multiplied
 * without overflow.
 */
void QuantizeMask(const float* mask_f32, int16_t* mask_i16, int rows,
                  int cols);

/**
 * @brief Prepare one prefill chunk's inputs.
 *
 * Geometry contract: the model context is fixed at kCacheLen = 4096 tokens
 * and the right-aligned mask layout indexes a full @p seq_len window per
 * chunk, so every preparation/mask call validates `0 <= chunk_start`,
 * `0 <= chunk_valid <= seq_len`, `1 <= seq_len <= kCacheLen` and
 * `chunk_start + seq_len <= kCacheLen` before touching any buffer
 * (overflow-safe signed arithmetic) and rejects anything outside the
 * supported window with std::invalid_argument. Nothing is clamped: a request
 * that does not fit the fixed context is an error, never a silent attention
 * change.
 *
 * @param embeddings Token embedding table (stage-owned dependency).
 * @param token_ids Valid chunk token ids; must hold exactly @p chunk_valid
 *        ids.
 * @param chunk_start Global offset of the chunk within the prompt.
 * @param chunk_valid Number of valid tokens in the chunk (0..seq_len).
 * @param prebuilt_hidden Optional inputs_embeds for the whole prompt: when
 *        present it must hold at least
 *        `(chunk_start + chunk_valid) * kHiddenSize` floats, because rows
 *        are indexed from the prompt start; the first chunk_valid rows
 *        override the pad embedding at their slots (raw vision features at
 *        image positions). Null keeps the pure embedding lookup.
 * @param seq_len Subgraph sequence length (prefill kChunkSize).
 *
 * Throws std::invalid_argument on any geometry, count or extent violation —
 * before any allocation, lookup or write.
 */
TextBatchInputs PrepareBatchInputs(const TokenEmbeddings& embeddings,
                                   const std::vector<int64_t>& token_ids,
                                   int chunk_start, int chunk_valid,
                                   const std::vector<float>* prebuilt_hidden,
                                   int seq_len);

/**
 * @brief Prepare one decode step's inputs (a one-row batch).
 *
 * @param embeddings Token embedding table.
 * @param token_id Last generated token.
 * @param pos Global position being decoded; must satisfy
 *        `0 <= pos < kCacheLen` (the step occupies one context row).
 *
 * Throws std::invalid_argument outside the fixed context.
 */
TextBatchInputs PrepareDecodeInputs(const TokenEmbeddings& embeddings,
                                    int64_t token_id, int pos);

// ---- Fixed Text export descriptor contract and strided physical IO ----

/// Which fixed-export binding a tensor description belongs to.
enum class TextTensorRole {
  kInputsEmbeds,  ///< inputs[0]: F32 [seq, kHiddenSize]
  kTokenIds,      ///< inputs[1]: S64 [seq] (source graph declares [1, seq])
  kPositionIds,   ///< inputs[2]: S32 [seq]
  kFullMask,      ///< inputs[3]: S16 [seq, kCacheLen]
  kSlidingMask,   ///< inputs[4]: S16 [seq, kCacheLen]
  kLogits,        ///< outputs[0]: S16 [seq, kVocabSize]
  kKvInput,       ///< inputs 5..34: S8 [kCacheLen, kHeadDims[layer]], contiguous
  kKvOutput,      ///< outputs 1..30: S8 [seq, kHeadDims[layer]], row padding ok
};

/**
 * @brief Validate one Text descriptor against the fixed-export contract.
 *
 * Singleton axes are ignored (they do not change the physical layout); after
 * collapsing them the rank and dimensions must match the role exactly.
 * Byte strides must be element aligned, nonoverlapping, and every addressed
 * byte must stay inside the declared allocation and, when @p capacity is
 * positive, inside the original buffer capacity. KV inputs must be fully
 * contiguous because KvCache owns raw contiguous matrices; KV outputs may
 * pad rows (the append path carries an explicit row stride). Quantization
 * metadata is rejected: mask/logit/KV quantization is applied on the CPU by
 * the source algorithm, not through tensor descriptors.
 *
 * @param properties SDK descriptor to check.
 * @param role Fixed-export binding of this tensor.
 * @param seq_len Expected leading dimension (kChunkSize prefill, 1 decode).
 * @param layer KV layer index (0..kNumKvLayers-1); ignored for other roles.
 * @param capacity Original allocation size, or 0 when untracked.
 */
void ValidateTextTensor(const hbDNNTensorProperties& properties,
                        TextTensorRole role, int seq_len, int layer = 0,
                        int64_t capacity = 0);

/**
 * @brief Write one CPU-prepared input into its BPU buffer.
 *
 * Validates the descriptor, zeroes the aligned buffer, and copies the compact
 * source matrix through the descriptor's strides. @p source_elements must
 * equal the descriptor's valid element count, and the innermost stride must
 * be the element width (the copy is compact along the last axis; inter-row
 * padding is supported). KV inputs and outputs are never written by the CPU.
 */
void WriteTextInput(hbDNNTensor& tensor, const void* source,
                    int64_t source_elements, TextTensorRole role, int seq_len,
                    int64_t capacity);

/// Physical location of one KV output's rows: head row address and row stride.
struct TextKvRows {
  const int8_t* data = nullptr;
  int64_t row_stride = 0;
};

/**
 * @brief Validate a KV output after inference and locate its rows.
 *
 * Requires S8 [seq_rows, kHeadDims[layer]] with contiguous columns and row
 * padding allowed, all addresses within the allocation and the original
 * capacity. The returned row stride is what KvCache append consumes.
 */
TextKvRows TextKvOutputRows(const hbDNNTensor& tensor, int layer, int seq_rows,
                            int64_t capacity);

/**
 * @brief Greedy argmax over one logits row, honoring the descriptor.
 *
 * Preserves the source semantics: int16 storage scaled by kLogitScale, the
 * first maximum wins ties, and the row is selected by @p seq_idx. The
 * descriptor must declare S16 [seq_len, kVocabSize]; no other storage type
 * or width is reinterpreted.
 */
int64_t ArgmaxTextLogits(const hbDNNTensor& tensor, int seq_idx, int seq_len,
                         int64_t capacity);

// ---- Stage 2: raw SDK inference and transport ----

/**
 * @brief Validate and allocate one Text subgraph's tensors.
 *
 * Enforces the fixed-export descriptor contract (35 inputs / 31 outputs,
 * pinned dtypes/shapes/strides, prefill seq = kChunkSize, decode seq = 1)
 * before any allocation. Throws on an incompatible export.
 */
ModelIo InitTextSubgraph(hbDNNPackedHandle_t packed, const char* name,
                         int seq_len);

/**
 * @brief Borrow one KvCache's buffers as both subgraphs' KV input slots.
 *
 * Requires the subgraphs to agree on every cache allocation size; records
 * the borrowed capacity. Only call before inference; successful reallocation
 * of the cache invalidates the aliases.
 */
void BindKvCache(ModelIo& prefill, ModelIo& decode, KvCache& cache);

/**
 * @brief Write one prepared batch into the subgraph's BPU input buffers.
 *
 * Strided writes through the accepted descriptor contract; the batch's
 * element counts must match the subgraph's sequence length.
 */
void WriteBatchInputs(ModelIo& io, const TextBatchInputs& batch);

/**
 * @brief Run one inference: flush CPU-modified inputs, submit, wait, refresh
 * all output descriptors.
 *
 * KV rows are rolled into the cache on CPU after every inference, so all 35
 * inputs are flushed before the BPU reads them again (source behavior).
 * Throws on SDK errors; the task is released on every failure path.
 */
void RunSubgraphInference(ModelIo& io);

/// Validated per-layer KV output rows (pointers borrowed from the tensors).
struct TextKvOutputSet {
  std::array<TextKvRows, kNumKvLayers> keys;
  std::array<TextKvRows, kNumKvLayers> values;
};

/**
 * @brief Revalidate the refreshed KV output descriptors against the owned
 * allocations and gather per-layer rows for the cache append.
 *
 * @param io Subgraph whose inference just completed.
 * @param rows Leading rows the caller will append (chunk_valid or 1); the
 *        descriptors themselves must carry a row per subgraph position.
 */
TextKvOutputSet CollectKvOutputs(const ModelIo& io, int rows);

// ---- Stage 3: orchestration (decode + KV update + session) ----

/**
 * @brief Run Gemma4-E2B text generation with reusable KV-cache state.
 *
 * A TextEngine owns the prefill/decode subgraphs, embedding table, KV cache
 * and one chat session. Callers must serialize access to an instance.
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
   * @return The full sequence: @p prompt_ids followed by the generated
   *         tokens (the same vector ContinueGenerate returns).
   */
  std::vector<int64_t> Generate(const std::vector<int64_t>& prompt_ids,
                                int max_new_tokens);

  /**
   * @brief Generate text with a per-token streaming callback.
   *
   * @param prompt_ids Token IDs of the prompt.
   * @param max_new_tokens Maximum number of new tokens to generate.
   * @param on_token Callback invoked with each newly generated token ID.
   *        Returning false stops generation early; the tokens generated so
   *        far are still returned.
   *
   * @return The full sequence: @p prompt_ids followed by the generated
   *         tokens.
   */
  std::vector<int64_t> GenerateStream(const std::vector<int64_t>& prompt_ids,
                                       int max_new_tokens, TokenCallback on_token);

  /**
   * @brief Generate text starting from prebuilt prompt hidden states.
   *
   * @param prompt_ids Token IDs of the prompt.
   * @param prompt_hidden Prebuilt inputs_embeds for the whole prompt —
   *        exactly `prompt_ids.size() * kHiddenSize` floats.
   * @param max_new_tokens Maximum number of new tokens to generate.
   *
   * @return The full sequence: @p prompt_ids followed by the generated
   *         tokens. A @p prompt_hidden size mismatch throws before any
   *         session state changes.
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
   * @param full_hidden Optional prebuilt inputs_embeds covering the WHOLE
   *        @p full_ids sequence — exactly
   *        `full_ids.size() * kHiddenSize` floats, indexed from the prompt
   *        start (not a suffix). A size mismatch throws before any session
   *        state changes; null keeps the pure embedding lookup.
   *
   * @return The full sequence: @p full_ids followed by the generated
   *         tokens.
   */
  std::vector<int64_t> ContinueGenerate(
      const std::vector<int64_t>& full_ids, int max_new_tokens,
      const std::vector<float>* full_hidden = nullptr);

  /**
   * @brief Streaming variant of @ref ContinueGenerate.
   *
   * @param full_ids Full token sequence including prior context.
   * @param max_new_tokens Maximum number of new tokens to generate.
   * @param on_token Per-token streaming callback. Returning false stops
   *        generation early; the tokens generated so far are still returned.
   * @param full_hidden Optional prebuilt inputs_embeds covering the WHOLE
   *        @p full_ids sequence — exactly
   *        `full_ids.size() * kHiddenSize` floats, indexed from the prompt
   *        start (not a suffix). A size mismatch throws before any session
   *        state changes; null keeps the pure embedding lookup.
   *
   * @return The full sequence: @p full_ids followed by the generated
   *         tokens.
   */
  std::vector<int64_t> ContinueGenerateStream(
      const std::vector<int64_t>& full_ids, int max_new_tokens,
      TokenCallback on_token,
      const std::vector<float>* full_hidden = nullptr);

  /// Clear all KV-cache and session state.
  void ResetSession();

  /// Number of tokens currently processed and held in the KV cache.
  int ProcessedTokens() const { return session_.processed_tokens; }

  // Context management for multi-turn chat
  /// Set the number of leading tokens preserved during a context shift.
  void SetKeepTokens(int n) { session_.n_keep = n; }
  /// Number of leading tokens preserved during a context shift.
  int KeepTokens() const { return session_.n_keep; }

  /**
   * @brief Compact the KV cache to make room for new context.
   *
   * Keeps the first @p n_keep tokens, discards the suffix, and compacts the
   * KV cache so generation can continue without exceeding capacity. The
   * caller must re-prefill the discarded tokens.
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
  const std::vector<int64_t>& GetHistory() const { return session_.history; }

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

  /**
   * @brief Install an explicit diagnostics receiver.
   *
   * The engine never prints on its own; without a sink, diagnostics are
   * discarded. Applications may wire this to their console/logging policy.
   */
  void SetDebugSink(DebugSink sink) { debug_sink_ = std::move(sink); }

 private:
  static bool IsEos(int64_t token_id);

  void EmitDebug(const std::string& message);
  void AppendKvChunk(const TextKvOutputSet& rows, int chunk_start,
                     int chunk_valid);
  void AppendKvStep(const TextKvOutputSet& rows, int pos);
  void RunPrefillChunk(const std::vector<int64_t>& chunk, int chunk_start,
                       const std::vector<float>* prebuilt_hidden = nullptr);
  void PrefillSuffix(const std::vector<int64_t>& ids, int start,
                     const std::vector<float>* hidden = nullptr);
  int64_t RunDecodeStep(int64_t token_id);

  hbDNNPackedHandle_t packed_ = nullptr;
  ModelIo prefill_;
  ModelIo decode_;
  TokenEmbeddings embeddings_;
  KvCache kv_;
  TextSessionState session_;
  double load_ms_ = 0;
  DebugSink debug_sink_;
};

}  // namespace gemma4
