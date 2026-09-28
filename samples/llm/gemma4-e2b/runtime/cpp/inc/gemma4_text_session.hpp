// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: MIT
/**
 * @file gemma4_text_session.hpp
 * @brief Multi-turn session state and pure continuation policy for Text.
 *
 * Holds the mutable session counters and chat history, and answers policy
 * questions (context shift, auto-truncate, continuation alignment, logits
 * row selection) as pure functions over that state. Decisions never touch
 * the KV cache or the SDK: the engine executes a returned plan.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "gemma4_config.hpp"

namespace gemma4 {

/// Mutable state of one serialized chat session.
struct TextSessionState {
  int processed_tokens = 0;  ///< Tokens resident in/covered by the KV cache.
  int token_offset = 0;      ///< Global position the next decode writes.
  int n_keep = 0;            ///< Leading tokens preserved by a context shift.
  std::vector<int64_t> history;  ///< Chat history for truncation decisions.

  /// Clear the cache-linked counters and history; @p n_keep is caller-owned
  /// and preserved (source ResetSession behavior).
  void Reset();
  void AddHistory(const std::vector<int64_t> &tokens);
  void ClearHistory();
};

/// A context shift the engine may execute: retain @p keep leading rows and
/// discard @p discard tokens (which the caller must re-prefill).
struct TextContextShiftPlan {
  bool valid = false;
  int keep = 0;
  int discard = 0;
};

/// Source semantics: invalid unless 0 <= keep < processed_tokens and the
/// discarded suffix is nonempty.
TextContextShiftPlan PlanContextShift(const TextSessionState &state,
                                      int keep);

/// An auto-truncate decision: shift to @p keep and ask the caller to replay
/// the recent history suffix.
struct TextAutoTruncatePlan {
  bool valid = false;
  int keep = 0;
};

/// Source semantics: no truncation when the cache still fits the request,
/// when nothing can be preserved, or when discarding everything would still
/// not fit.
TextAutoTruncatePlan PlanAutoTruncate(const TextSessionState &state,
                                      int new_prompt_tokens,
                                      int max_new_tokens);

/// The continuation alignment the source applies before prefilling a suffix:
/// when the processed prefix is not a multiple of the chunk size, keep only
/// its chunk-aligned part and replay the remainder.
struct TextContinuationPlan {
  bool needs_alignment = false;
  int aligned_prefix = 0;
  int replay_tokens = 0;
};

TextContinuationPlan PlanContinuationAlignment(const TextSessionState &state,
                                               int full_ids);

/// Row of the prefill logits holding the last processed token (with the
/// source's clamp back to row 0).
int LastChunkRowIndex(int processed_tokens);

}  // namespace gemma4
