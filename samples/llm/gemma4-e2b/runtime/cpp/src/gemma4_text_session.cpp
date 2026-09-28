// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_session.cpp
 * @brief Session state and pure policy decisions for Text generation.
 *
 * Every decision table is moved verbatim from the former TextEngine methods
 * (ContextShift, AutoTruncate, ContinueGenerateStream alignment, last-chunk
 * row selection).
 */

#include "gemma4_text_session.hpp"

namespace gemma4 {

void TextSessionState::Reset() {
  processed_tokens = 0;
  token_offset = 0;
  history.clear();
}

void TextSessionState::AddHistory(const std::vector<int64_t> &tokens) {
  history.insert(history.end(), tokens.begin(), tokens.end());
}

void TextSessionState::ClearHistory() { history.clear(); }

TextContextShiftPlan PlanContextShift(const TextSessionState &state,
                                      int keep) {
  TextContextShiftPlan plan;
  if (keep < 0 || keep >= state.processed_tokens) {
    return plan;
  }
  plan.discard = state.processed_tokens - keep;
  if (plan.discard <= 0) {
    return plan;
  }
  plan.valid = true;
  plan.keep = keep;
  return plan;
}

TextAutoTruncatePlan PlanAutoTruncate(const TextSessionState &state,
                                      int new_prompt_tokens,
                                      int max_new_tokens) {
  TextAutoTruncatePlan plan;
  const int available = kCacheLen - state.processed_tokens;
  const int needed = new_prompt_tokens + max_new_tokens;

  if (available >= needed) {
    return plan;  // No truncation needed.
  }
  // Need to truncate: keep the preserved prefix and discard old history.
  if (state.n_keep <= 0 || state.n_keep >= state.processed_tokens) {
    return plan;  // Nothing to keep or nothing to discard.
  }
  const int overflow = needed - available;
  const int discardable = state.processed_tokens - state.n_keep;
  if (overflow > discardable) {
    return plan;  // Even discarding everything would not fit.
  }
  plan.valid = true;
  plan.keep = state.n_keep;
  return plan;
}

TextContinuationPlan PlanContinuationAlignment(const TextSessionState &state,
                                               int full_ids) {
  TextContinuationPlan plan;
  if (full_ids > state.processed_tokens &&
      state.processed_tokens % kChunkSize != 0) {
    plan.needs_alignment = true;
    plan.aligned_prefix = (state.processed_tokens / kChunkSize) * kChunkSize;
    plan.replay_tokens = state.processed_tokens - plan.aligned_prefix;
  }
  return plan;
}

int LastChunkRowIndex(int processed_tokens) {
  int last_idx = 0;
  if (processed_tokens > 0) {
    const int last_chunk_start =
        ((processed_tokens - 1) / kChunkSize) * kChunkSize;
    last_idx = processed_tokens - 1 - last_chunk_start;
    if (last_idx < 0 || last_idx >= kChunkSize) {
      last_idx = 0;
    }
  }
  return last_idx;
}

}  // namespace gemma4
