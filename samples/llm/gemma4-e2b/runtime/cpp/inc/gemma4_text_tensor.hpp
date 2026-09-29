// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_tensor.hpp
 * @brief Physical descriptor contract for the fixed Gemma4-E2B Text export.
 *
 * The Text HBM is exported from the fixed source graph with a pinned layout
 * (chunk 256, cache 4096, hidden 1536, vocab 262144, 15 K/V layers). These
 * helpers own descriptor validation and the raw physical reads/writes for
 * that layout; TextEngine keeps only model orchestration. Layouts outside
 * the contract are rejected, never reinterpreted: an unknown or mismatched
 * tensor type cannot fall through to a float or contiguous reinterpretation.
 */
#pragma once

#include "hobot/dnn/hb_dnn.h"

#include <cstdint>

namespace gemma4 {

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
void ValidateTextTensor(const hbDNNTensorProperties &properties,
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
void WriteTextInput(hbDNNTensor &tensor, const void *source,
                    int64_t source_elements, TextTensorRole role, int seq_len,
                    int64_t capacity);

/// Physical location of one KV output's rows: head row address and row stride.
struct TextKvRows {
  const int8_t *data = nullptr;
  int64_t row_stride = 0;
};

/**
 * @brief Validate a KV output after inference and locate its rows.
 *
 * Requires S8 [seq_rows, kHeadDims[layer]] with contiguous columns and row
 * padding allowed, all addresses within the allocation and the original
 * capacity. The returned row stride is what KvCache append consumes.
 */
TextKvRows TextKvOutputRows(const hbDNNTensor &tensor, int layer, int seq_rows,
                            int64_t capacity);

/**
 * @brief Greedy argmax over one logits row, honoring the descriptor.
 *
 * Preserves the source semantics: int16 storage scaled by kLogitScale, the
 * first maximum wins ties, and the row is selected by @p seq_idx. The
 * descriptor must declare S16 [seq_len, kVocabSize]; no other storage type
 * or width is reinterpreted.
 */
int64_t ArgmaxTextLogits(const hbDNNTensor &tensor, int seq_idx, int seq_len,
                         int64_t capacity);

}  // namespace gemma4
