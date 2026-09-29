// Pure session-policy decisions: no SDK, no cache, no engine.
#include "gemma4_text_session.hpp"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  using gemma4::LastChunkRowIndex;
  using gemma4::PlanAutoTruncate;
  using gemma4::PlanContextShift;
  using gemma4::PlanContinuationAlignment;
  using gemma4::TextSessionState;

  // ---- context shift decision table ----
  {
    TextSessionState state;
    state.processed_tokens = 300;
    state.token_offset = 300;
    // keep < 0, keep == processed, keep > processed: refused.
    assert(!PlanContextShift(state, -1).valid);
    assert(!PlanContextShift(state, 300).valid);
    assert(!PlanContextShift(state, 301).valid);
    const auto plan = PlanContextShift(state, 256);
    assert(plan.valid && plan.keep == 256 && plan.discard == 44);
    const auto keep_all = PlanContextShift(state, 0);
    assert(keep_all.valid && keep_all.keep == 0 && keep_all.discard == 300);
    // Empty session: nothing to shift.
    TextSessionState empty;
    assert(!PlanContextShift(empty, 0).valid);
  }

  // ---- auto-truncate decision table ----
  {
    TextSessionState state;
    state.processed_tokens = 4000;
    state.n_keep = 8;
    // Fits without truncation.
    assert(!PlanAutoTruncate(state, 64, 16).valid);
    // Overflows and a prefix is preserved: shift to n_keep.
    const auto plan = PlanAutoTruncate(state, 128, 32);
    assert(plan.valid && plan.keep == 8);
    // Nothing to keep.
    state.n_keep = 0;
    assert(!PlanAutoTruncate(state, 128, 32).valid);
    // Keep equals processed: nothing to discard.
    state.n_keep = 4000;
    assert(!PlanAutoTruncate(state, 128, 32).valid);
    // Discarding everything still does not fit.
    state.n_keep = 8;
    assert(!PlanAutoTruncate(state, 4096, 128).valid);
    // Fits only after discarding: overflow within the discardable range.
    const auto exact = PlanAutoTruncate(state, 3999, 1);
    assert(exact.valid && exact.keep == 8);
  }

  // ---- continuation alignment (prefix replay) ----
  {
    TextSessionState state;
    state.processed_tokens = 512;
    // Chunk-aligned prefix continues without alignment.
    const auto aligned = PlanContinuationAlignment(state, 600);
    assert(!aligned.needs_alignment && aligned.aligned_prefix == 0);
    // Unaligned prefix (e.g. after generation advanced the count): align and
    // replay the remainder.
    state.processed_tokens = 513;
    const auto unaligned = PlanContinuationAlignment(state, 600);
    assert(unaligned.needs_alignment && unaligned.aligned_prefix == 512 &&
           unaligned.replay_tokens == 1);
    // Extending at an aligned boundary never replays.
    state.processed_tokens = 256;
    const auto boundary = PlanContinuationAlignment(state, 257);
    assert(!boundary.needs_alignment);
    // No new tokens: no alignment.
    assert(!PlanContinuationAlignment(state, 256).needs_alignment);
    // Any unaligned processed prefix aligns — including small ones, which
    // align to zero and replay everything through the shift.
    state.processed_tokens = 5;
    const auto small = PlanContinuationAlignment(state, 10);
    assert(small.needs_alignment && small.aligned_prefix == 0 &&
           small.replay_tokens == 5);
  }

  // ---- last-chunk logits row ----
  assert(LastChunkRowIndex(0) == 0);
  assert(LastChunkRowIndex(1) == 0);
  assert(LastChunkRowIndex(5) == 4);
  assert(LastChunkRowIndex(256) == 255);
  assert(LastChunkRowIndex(257) == 0);   // First row of the second chunk.
  assert(LastChunkRowIndex(513) == 0);

  // ---- session reset keeps the caller-owned keep token ----
  {
    TextSessionState state;
    state.processed_tokens = 100;
    state.token_offset = 100;
    state.n_keep = 8;
    state.history = {1, 2, 3};
    state.Reset();
    assert(state.processed_tokens == 0 && state.token_offset == 0);
    assert(state.history.empty());
    assert(state.n_keep == 8);
    state.AddHistory({4, 5});
    assert(state.history.size() == 2 && state.history[1] == 5);
    state.ClearHistory();
    assert(state.history.empty());
  }

  std::cout << "text session policy checks passed\n";
  return 0;
}
