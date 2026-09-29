// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_inputs.cpp
 * @brief Stage 1 implementation: prepared per-call Text context on the CPU.
 *
 * The algorithms are moved verbatim from the former TextEngine input
 * preparation; only the implicit debug print was removed (diagnostics are an
 * engine-level, caller-installed sink concern).
 */

#include "gemma4_text_inputs.hpp"

#include <algorithm>
#include <cmath>

namespace gemma4 {

void BuildFullMask(float *mask, int chunk_start, int chunk_valid,
                   int seq_len) {
  const int total = seq_len * kCacheLen;
  std::fill(mask, mask + total, kMaskValue);

  const int total_seen = std::min(chunk_start + chunk_valid, kCacheLen);
  const int valid_total = std::min(total_seen, kCacheLen);
  const int cache_start_abs = total_seen - valid_total;
  const int current_pad = seq_len - chunk_valid;
  const int cache_col_start = kCacheLen - current_pad - valid_total;

  for (int r = 0; r < seq_len; ++r) {
    int query_abs = chunk_start + r;
    if (query_abs >= total_seen) {
      query_abs = total_seen - 1;
    }
    const int allowed_start = cache_start_abs;
    const int allowed_end = query_abs;
    if (allowed_end < allowed_start) continue;
    const int start_col = cache_col_start + (allowed_start - cache_start_abs);
    const int end_col = cache_col_start + (allowed_end - cache_start_abs);
    float *row = mask + static_cast<size_t>(r) * kCacheLen;
    for (int c = start_col; c <= end_col; ++c) {
      row[c] = 0.f;
    }
  }
}

void BuildSlidingMask(float *mask, int chunk_start, int chunk_valid,
                      int seq_len) {
  const int total = seq_len * kCacheLen;
  std::fill(mask, mask + total, kMaskValue);

  const int total_seen = std::min(chunk_start + chunk_valid, kCacheLen);
  const int valid_total = std::min(total_seen, kCacheLen);
  const int cache_start_abs = total_seen - valid_total;
  const int current_pad = seq_len - chunk_valid;
  const int cache_col_start = kCacheLen - current_pad - valid_total;

  for (int r = 0; r < seq_len; ++r) {
    int query_abs = chunk_start + r;
    if (query_abs >= total_seen) {
      query_abs = total_seen - 1;
    }
    int allowed_start = cache_start_abs;
    allowed_start = std::max(allowed_start, query_abs - kSlidingWindow + 1);
    const int allowed_end = query_abs;
    if (allowed_end < allowed_start) continue;
    const int start_col = cache_col_start + (allowed_start - cache_start_abs);
    const int end_col = cache_col_start + (allowed_end - cache_start_abs);
    float *row = mask + static_cast<size_t>(r) * kCacheLen;
    for (int c = start_col; c <= end_col; ++c) {
      row[c] = 0.f;
    }
  }
}

void QuantizeMask(const float *mask_f32, int16_t *mask_i16, int rows,
                  int cols) {
  for (int i = 0; i < rows * cols; ++i) {
    const float v = std::max(-32768.f, std::min(32767.f, std::round(mask_f32[i])));
    mask_i16[i] = static_cast<int16_t>(v);
  }
}

namespace {

// Image token ids → pad for PLE token-identity (token input and the embed
// lookup base), exactly as the source prepared them.
std::vector<int64_t> PleSubstituted(const std::vector<int64_t> &ids) {
  std::vector<int64_t> substituted = ids;
  for (auto &id : substituted) {
    if (id == kImageTokenId) {
      id = kPadTokenId;
    }
  }
  return substituted;
}

}  // namespace

TextBatchInputs PrepareBatchInputs(const TokenEmbeddings &embeddings,
                                   const std::vector<int64_t> &token_ids,
                                   int chunk_start, int chunk_valid,
                                   const float *prebuilt_hidden, int seq_len) {
  TextBatchInputs batch;
  batch.token_ids.resize(static_cast<size_t>(seq_len), 0);
  batch.hidden.assign(static_cast<size_t>(seq_len) * kHiddenSize, 0.f);
  batch.positions.resize(static_cast<size_t>(seq_len), 0);
  batch.full_mask_q.resize(static_cast<size_t>(seq_len) * kCacheLen);
  batch.slide_mask_q.resize(static_cast<size_t>(seq_len) * kCacheLen);

  // PLE substitution over the padded id row: pad ids keep the pad embedding
  // while the image slots are substituted before the lookup.
  std::vector<int64_t> padded = token_ids;
  padded.resize(static_cast<size_t>(seq_len), 0);
  const std::vector<int64_t> ple_padded = PleSubstituted(padded);

  // Hidden buffer: pad embedding lookup first; with prebuilt hidden the
  // first chunk_valid rows are then overwritten (raw vision at image slots).
  embeddings.Lookup(ple_padded, batch.hidden.data());
  if (prebuilt_hidden != nullptr) {
    for (int i = 0; i < chunk_valid; ++i) {
      const float *src = prebuilt_hidden +
                         static_cast<size_t>(chunk_start + i) * kHiddenSize;
      float *dst = batch.hidden.data() + static_cast<size_t>(i) * kHiddenSize;
      std::copy(src, src + kHiddenSize, dst);
    }
  }

  for (int i = 0; i < seq_len; ++i) {
    batch.token_ids[static_cast<size_t>(i)] = ple_padded[static_cast<size_t>(i)];
    batch.positions[static_cast<size_t>(i)] =
        i < chunk_valid ? chunk_start + i
                        : chunk_start + std::max(chunk_valid - 1, 0);
  }

  std::vector<float> full_mask(static_cast<size_t>(seq_len) * kCacheLen);
  std::vector<float> slide_mask(static_cast<size_t>(seq_len) * kCacheLen);
  BuildFullMask(full_mask.data(), chunk_start, chunk_valid, seq_len);
  BuildSlidingMask(slide_mask.data(), chunk_start, chunk_valid, seq_len);
  QuantizeMask(full_mask.data(), batch.full_mask_q.data(), seq_len, kCacheLen);
  QuantizeMask(slide_mask.data(), batch.slide_mask_q.data(), seq_len,
               kCacheLen);
  return batch;
}

TextBatchInputs PrepareDecodeInputs(const TokenEmbeddings &embeddings,
                                    int64_t token_id, int pos) {
  TextBatchInputs batch;
  batch.token_ids = {token_id};
  batch.hidden.assign(kHiddenSize, 0.f);
  batch.positions = {static_cast<int32_t>(pos)};
  batch.full_mask_q.resize(kCacheLen);
  batch.slide_mask_q.resize(kCacheLen);

  embeddings.Lookup(std::vector<int64_t>{token_id}, batch.hidden.data());

  std::vector<float> full_mask(kCacheLen);
  std::vector<float> slide_mask(kCacheLen);
  BuildFullMask(full_mask.data(), pos, 1, 1);
  BuildSlidingMask(slide_mask.data(), pos, 1, 1);
  QuantizeMask(full_mask.data(), batch.full_mask_q.data(), 1, kCacheLen);
  QuantizeMask(slide_mask.data(), batch.slide_mask_q.data(), 1, kCacheLen);
  return batch;
}

}  // namespace gemma4
