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
 * @brief Build the mask matrices with the source's right-aligned layout.
 *
 * After a chunk covering [chunk_start, chunk_start+chunk_valid): row r
 * (clamped to the last seen token for pad rows) attends to
 * [cache_start_abs .. query_abs]; the sliding variant additionally starts at
 * query_abs-kSlidingWindow+1. These functions never read the cache: the
 * layout depends only on the chunk window and constants.
 */
void BuildFullMask(float *mask, int chunk_start, int chunk_valid, int seq_len);
void BuildSlidingMask(float *mask, int chunk_start, int chunk_valid,
                      int seq_len);

/// Quantize float masks to int16 with the source round+clamp algorithm.
void QuantizeMask(const float *mask_f32, int16_t *mask_i16, int rows,
                  int cols);

/**
 * @brief Prepare one prefill chunk's inputs.
 *
 * @param embeddings Token embedding table (stage-owned dependency).
 * @param token_ids Valid chunk token ids (at most seq_len).
 * @param chunk_start Global offset of the chunk within the prompt.
 * @param chunk_valid Number of valid tokens in the chunk.
 * @param prebuilt_hidden Optional inputs_embeds for the whole prompt; when
 *        present the first chunk_valid rows override the pad embedding at
 *        their slots (raw vision features at image positions).
 * @param seq_len Subgraph sequence length (prefill kChunkSize).
 */
TextBatchInputs PrepareBatchInputs(const TokenEmbeddings &embeddings,
                                   const std::vector<int64_t> &token_ids,
                                   int chunk_start, int chunk_valid,
                                   const float *prebuilt_hidden, int seq_len);

/**
 * @brief Prepare one decode step's inputs (a one-row batch).
 *
 * @param embeddings Token embedding table.
 * @param token_id Last generated token.
 * @param pos Global position being decoded.
 */
TextBatchInputs PrepareDecodeInputs(const TokenEmbeddings &embeddings,
                                    int64_t token_id, int pos);

}  // namespace gemma4
