// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_inputs.hpp
 * @brief Stage 1 of the Text pipeline: pure CPU input preparation.
 *
 * Builds the prepared per-call context for one prefill chunk or decode step
 * — token ids (with the source's PLE image-token substitution), embedding
 * rows, positions and the quantized attention masks — as plain vectors with
 * no SDK types and no printing. The physical strided write into BPU buffers
 * belongs to stage 2 (gemma4_text_transport).
 *
 * Geometry contract: the model context is fixed at kCacheLen = 4096 tokens
 * and the right-aligned mask layout indexes a full @p seq_len window per
 * chunk, so every preparation/mask call validates
 * `0 <= chunk_start`, `0 <= chunk_valid <= seq_len`, `1 <= seq_len <=
 * kCacheLen` and `chunk_start + seq_len <= kCacheLen` before touching any
 * buffer (overflow-safe signed arithmetic) and rejects anything outside the
 * supported window with std::invalid_argument. Positions follow the same
 * rule (`pos < kCacheLen` for decode). Nothing is clamped: a request that
 * does not fit the fixed context is an error, never a silent attention
 * change.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "gemma4_config.hpp"
#include "gemma4_embeddings.hpp"

namespace gemma4 {

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
 * [cache_start_abs .. query_abs]. Validates the geometry contract above and
 * throws std::invalid_argument before writing when it is violated; @p mask
 * must have room for seq_len*kCacheLen elements.
 */
void BuildFullMask(float *mask, int chunk_start, int chunk_valid, int seq_len);

/**
 * @brief Build the sliding-window mask (source layout).
 *
 * Same geometry contract as @ref BuildFullMask; rows additionally start at
 * query_abs-kSlidingWindow+1.
 */
void BuildSlidingMask(float *mask, int chunk_start, int chunk_valid,
                      int seq_len);

/**
 * @brief Quantize float masks to int16 with the source round+clamp
 * algorithm.
 *
 * Rejects negative @p rows/@p cols and counts that cannot be multiplied
 * without overflow.
 */
void QuantizeMask(const float *mask_f32, int16_t *mask_i16, int rows,
                  int cols);

/**
 * @brief Prepare one prefill chunk's inputs.
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
TextBatchInputs PrepareBatchInputs(const TokenEmbeddings &embeddings,
                                   const std::vector<int64_t> &token_ids,
                                   int chunk_start, int chunk_valid,
                                   const std::vector<float> *prebuilt_hidden,
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
TextBatchInputs PrepareDecodeInputs(const TokenEmbeddings &embeddings,
                                    int64_t token_id, int pos);

}  // namespace gemma4
