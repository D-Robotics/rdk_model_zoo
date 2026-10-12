// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// Behavioural host checks for the ByteTrack C++ runtime, compiling the
/// production sources (tracking.cpp, lapjv.cpp, parallel.cpp) — no stubs.
/// Boundary scores, empty frames, occlusion/lost-beyond-buffer lifecycle,
/// equal-cost ties, thread-count invariance, reset/ID monotonicity, invalid
/// input rejection, frame-temporary lifetime, and the parallel runner's
/// exactly-once coverage with real multi-thread engagement.
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "parallel.hpp"
#include "tracking.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

void check_eq(long long got, long long want, const std::string& what) {
  if (got != want) {
    ++g_failures;
    std::printf("FAIL: %s (got %lld, want %lld)\n", what.c_str(), got, want);
  }
}

bytetrack::Detection det(float x1, float y1, float x2, float y2, float score) {
  return bytetrack::Detection{x1, y1, x2, y2, score};
}

/// Deterministic integer-only PRNG so float sequences are identical on every
/// host and board (all arithmetic is IEEE-exact from these integers).
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed ? seed : 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double uniform(double lo, double hi) {  // k/4096 grid, exact in double
    return lo + static_cast<double>(next() % 4096) / 4096.0 * (hi - lo);
  }
};

struct Person {
  double x, y, w, h, vx, vy;
  double base_score;
  int occlude_from = -1, occlude_to = -1;   // detection missing window
  int low_from = -1, low_to = -1;           // low-score dip window
};

/// Synthetic multi-person stream: constant velocity + grid jitter, occlusion
/// gaps, low-score dips and periodic empty frames (the same shape of stress
/// the Python parity inputs use).
std::vector<std::vector<bytetrack::Detection>> synth_stream(int persons, int frames,
                                                            uint64_t seed) {
  Lcg rng(seed);
  std::vector<Person> ps(persons);
  for (Person& p : ps) {
    p.x = rng.uniform(0.0, 1600.0);
    p.y = rng.uniform(0.0, 800.0);
    p.w = rng.uniform(30.0, 90.0);
    p.h = rng.uniform(60.0, 180.0);
    p.vx = rng.uniform(-6.0, 6.0);
    p.vy = rng.uniform(-4.0, 4.0);
    p.base_score = rng.uniform(0.5, 0.95);
    p.occlude_from = 100 + static_cast<int>(rng.next() % 60);
    p.occlude_to = p.occlude_from + 8 + static_cast<int>(rng.next() % 25);
    p.low_from = 200 + static_cast<int>(rng.next() % 40);
    p.low_to = p.low_from + 10 + static_cast<int>(rng.next() % 30);
  }
  std::vector<std::vector<bytetrack::Detection>> out(frames);
  for (int f = 0; f < frames; ++f) {
    if (f % 90 >= 85) continue;  // 5 consecutive empty frames every 90
    for (Person& p : ps) {
      if (f >= p.occlude_from && f < p.occlude_to) continue;
      const double jx = rng.uniform(-1.5, 1.5), jy = rng.uniform(-1.0, 1.0);
      const double x = p.x + p.vx * f + jx, y = p.y + p.vy * f + jy;
      const double sc = (f >= p.low_from && f < p.low_to) ? 0.2 : p.base_score;
      out[f].push_back(det(static_cast<float>(x), static_cast<float>(y),
                           static_cast<float>(x + p.w), static_cast<float>(y + p.h),
                           static_cast<float>(sc)));
      p.w += rng.uniform(-0.2, 0.2);
      p.h += rng.uniform(-0.3, 0.3);
    }
  }
  return out;
}

std::vector<std::vector<bytetrack::Track>> run_stream(
    const std::vector<std::vector<bytetrack::Detection>>& frames, unsigned threads) {
  bytetrack::ByteTracker tracker(bytetrack::TrackingConfig{}, threads);
  std::vector<std::vector<bytetrack::Track>> out(frames.size());
  for (size_t f = 0; f < frames.size(); ++f)
    out[f] = tracker.update(frames[f], 1080, 1920);
  return out;
}

bool tracks_identical(const std::vector<bytetrack::Track>& a,
                      const std::vector<bytetrack::Track>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].track_id != b[i].track_id || a[i].frame_id != b[i].frame_id) return false;
    if (std::memcmp(a[i].tlbr, b[i].tlbr, sizeof a[i].tlbr) != 0) return false;
    if (a[i].score != b[i].score) return false;
  }
  return true;
}

// ---------------------------------------------------------------- checks ----

void check_parallel_runner() {
  for (unsigned threads : {1u, 2u, 4u, 8u}) {
    bytetrack::ParallelRunner runner(threads);
    // Counted coverage: incrementing detects a chunk written twice, which a
    // boolean hit flag could not (1 assigned again is still 1).
    std::vector<unsigned> hits(100000, 0);
    bool parallel_engaged = false;
    for (int round = 0; round < 3; ++round) {
      std::fill(hits.begin(), hits.end(), 0u);
      const bool engaged = runner.run(hits.size(), 1, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) ++hits[i];
      });
      check(engaged == (runner.slots() > 1), "parallel engagement flag matches slots");
      parallel_engaged = parallel_engaged || engaged;
      bool covered = true;
      for (unsigned h : hits) covered = covered && h == 1;
      check(covered, "exactly-once coverage (threads " + std::to_string(threads) + ")");
    }
    // Real multi-threading, not a sequential fallback: with more than one
    // slot, overlapping chunk execution must observe >= 2 distinct threads.
    if (runner.slots() > 1) {
      std::mutex mu;
      std::vector<std::thread::id> ids;
      runner.run(64 * runner.slots(), 1, [&](size_t, size_t) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard<std::mutex> g(mu);
        ids.push_back(std::this_thread::get_id());
      });
      std::sort(ids.begin(), ids.end());
      const size_t distinct = std::unique(ids.begin(), ids.end()) - ids.begin();
      check(distinct >= 2, "chunks ran on >= 2 threads (real parallelism)");
      check(parallel_engaged, "parallel region engaged for threads=" + std::to_string(threads));
    }
    // Below the chunk threshold the work must run inline and still cover.
    {
      std::vector<int> small(10, 0);
      const bool engaged = runner.run(small.size(), 1000, [&](size_t b, size_t e) {
        for (size_t i = b; i < e; ++i) small[i] = 1;
      });
      check(!engaged, "small work runs inline");
      for (int v : small) check_eq(v, 1, "inline fallback covers items");
    }
    // An exception thrown on the CALLER's chunk (index 7 is chunk 0)
    // propagates and leaves the runner usable...
    bool threw = false;
    try {
      runner.run(100000, 1, [](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
          if (i == 7) throw std::runtime_error("caller chunk exception");
      });
    } catch (const std::runtime_error&) {
      threw = true;
    }
    check(threw, "caller-chunk exception propagates");
    // ...and so does one thrown inside a real WORKER chunk: with fixed-slot
    // partitioning the last index lives in the final slot, which the caller
    // never executes when slots > 1.
    threw = false;
    try {
      runner.run(100000, 1, [](size_t begin, size_t end) {
        if (end >= 100000 && begin > 7) throw std::runtime_error("worker chunk exception");
      });
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == "worker chunk exception";
    }
    if (runner.slots() > 1) check(threw, "worker-chunk exception propagates to caller");
    std::vector<unsigned> after(100000, 0);
    runner.run(after.size(), 1, [&after](size_t b, size_t e) {
      for (size_t i = b; i < e; ++i) ++after[i];
    });
    bool covered = true;
    for (unsigned h : after) covered = covered && h == 1;
    check(covered, "runner reusable after exception");
  }
}

void check_score_boundaries() {
  // score == track_thresh (float32(0.3)) is excluded from BOTH pools.
  {
    bytetrack::ByteTracker t;
    const auto out = t.update({det(100, 100, 160, 260, 0.3f)}, 1080, 1920);
    check(out.empty(), "score == track_thresh activates nothing on frame 1");
    check_eq(t.frame_id(), 1, "frame advanced");
  }
  // score < det_thresh (float32 of 0.3+0.1) never activates a new track.
  {
    bytetrack::ByteTracker t;
    const auto out = t.update({det(100, 100, 160, 260, 0.3999f)}, 1080, 1920);
    check(out.empty(), "score just below det_thresh does not activate");
  }
  // score >= det_thresh on frame 1 activates and is reported.
  {
    bytetrack::ByteTracker t;
    const auto out = t.update({det(100, 100, 160, 260, 0.5f)}, 1080, 1920);
    check_eq(out.size(), 1, "frame-1 high-score detection activates");
    check_eq(out[0].track_id, 1, "first ID is 1");
    check_eq(out[0].frame_id, 1, "snapshot frame id");
    check(std::fabs(out[0].tlbr[2] - 160.0) < 1e-9 && std::fabs(out[0].tlbr[3] - 260.0) < 1e-9,
          "frame-1 box echoes the detection (mean initialized from it)");
  }
  // Low-score second pool: 0.1 < score < track_thresh matches an existing
  // track (no new ID) but alone activates nothing.
  {
    bytetrack::ByteTracker t;
    t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    const auto out = t.update({det(101, 101, 161, 261, 0.15f)}, 1080, 1920);
    check_eq(out.size(), 1, "low-score match keeps exactly one track");
    check_eq(out[0].track_id, 1, "low-score match reuses the same ID");
  }
  // score == 0.1f exactly is below the strict low bound: no second pool.
  {
    bytetrack::ByteTracker t;
    t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    const auto out = t.update({det(101, 101, 161, 261, 0.1f)}, 1080, 1920);
    check(out.empty(), "score == 0.1 is excluded from the second pool");
  }
}

void check_empty_and_lifecycle() {
  // Empty frames keep the tracker advancing without output.
  {
    bytetrack::ByteTracker t;
    t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    for (int i = 0; i < 5; ++i) {
      const auto out = t.update({}, 1080, 1920);
      check(out.empty(), "empty frame after loss emits nothing");
    }
    check_eq(t.frame_id(), 6, "empty frames still advance the index");
    // Occlusion reappearance inside the buffer: same ID returns.
    const auto out = t.update({det(102, 102, 162, 262, 0.8f)}, 1080, 1920);
    check_eq(out.size(), 1, "reappearance reactivates the track");
    check_eq(out[0].track_id, 1, "reappearance keeps the ID");
  }
  // Lost beyond the buffer (default 60): the track is marked removed on the
  // 61st empty frame but — exactly like the Python rebuild order — stays in
  // lost_stracks one more frame, so a detection arriving then re-finds the
  // SAME ID; a few frames later the ID is gone for good and a new one starts.
  {
    bytetrack::ByteTracker t;
    t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    for (int i = 0; i < 61; ++i) t.update({}, 1080, 1920);
    const auto lag = t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    check_eq(lag.size(), 1, "one-frame removal lag: detection still re-finds the track");
    check_eq(lag[0].track_id, 1, "one-frame removal lag keeps the ID (Python rebuild order)");
  }
  {
    bytetrack::ByteTracker t;
    t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    for (int i = 0; i < 70; ++i) t.update({}, 1080, 1920);
    const auto fresh = t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    check(fresh.empty(), "brand-new track on frame > 1 is not reported yet");
    const auto out = t.update({det(101, 101, 161, 261, 0.9f)}, 1080, 1920);
    check_eq(out.size(), 1, "new track confirmed on its next frame");
    check_eq(out[0].track_id, 2, "removed track's ID is not reused");
  }
  // Unconfirmed removal: a first-frame track that never matches again is
  // dropped (not activated on frame > 1, unmatched in the unconfirmed stage).
  {
    bytetrack::ByteTracker t;
    t.update({}, 1080, 1920);  // frame 1: nothing (so next frame is not the first)
    const auto first = t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
    check(first.empty(), "new track on frame > 1 is not yet activated");
    const auto out = t.update({}, 1080, 1920);
    check(out.empty(), "unconfirmed unmatched track disappears");
  }
}

void check_equal_cost_ties() {
  // Two detections at identical positions/scores: whichever the solver picks
  // must be stable across thread counts (association is sequential).
  for (unsigned threads : {1u, 2u, 4u, 8u}) {
    bytetrack::ByteTracker t(bytetrack::TrackingConfig{}, threads);
    const auto out1 = t.update({det(200, 200, 260, 360, 0.9f), det(400, 200, 460, 360, 0.9f)},
                               1080, 1920);
    const auto out2 = t.update({det(201, 201, 261, 361, 0.85f), det(401, 201, 461, 361, 0.85f)},
                               1080, 1920);
    check_eq(out1.size(), 2, "two distinct persons tracked");
    check_eq(out2.size(), 2, "both matched again");
  }
  // Identical cost rows (a track equidistant to two detections) resolve to a
  // deterministic assignment; the other detection starts a new track which —
  // like the Python source — is not reported until its second frame.
  {
    bytetrack::ByteTracker t;
    const auto out1 = t.update({det(200, 200, 260, 360, 0.9f)}, 1080, 1920);
    const auto out2 = t.update({det(202, 202, 262, 362, 0.9f), det(198, 198, 258, 358, 0.9f)},
                               1080, 1920);
    const auto out3 = t.update({det(204, 204, 264, 364, 0.9f), det(196, 196, 256, 356, 0.9f)},
                               1080, 1920);
    check_eq(out1.size(), 1, "single seed track");
    check_eq(out2.size(), 1, "near-tie frame: only the matched track is activated");
    check_eq(out3.size(), 2, "near-tie next frame: match plus the new track");
  }
}

void check_thread_invariance() {
  // Small streams keep the pool below the chunk thresholds (IoU rows need 24
  // items, per-track predict 64), so 2/4/8 threads take the inline fallback
  // here — the sequential-identity baseline.
  const auto frames = synth_stream(40, 300, 20261012);
  const auto ref = run_stream(frames, 1);
  for (unsigned threads : {2u, 4u, 8u}) {
    const auto got = run_stream(frames, threads);
    bool same = got.size() == ref.size();
    for (size_t f = 0; same && f < ref.size(); ++f) same = tracks_identical(ref[f], got[f]);
    check(same, "outputs byte-identical for threads=" + std::to_string(threads));
  }
  // Large streams push the pool PAST every threshold, so the parallel paths
  // genuinely engage (300 tracked+lost tracks > 64 predict chunk, > 24 IoU
  // rows) and must still produce byte-identical output.
  const auto big = synth_stream(300, 120, 424242);
  const auto big_ref = run_stream(big, 1);
  for (unsigned threads : {2u, 4u, 8u}) {
    const auto got = run_stream(big, threads);
    bool same = got.size() == big_ref.size();
    for (size_t f = 0; same && f < big_ref.size(); ++f) same = tracks_identical(big_ref[f], got[f]);
    check(same, "large-stream outputs byte-identical with real parallel chunks for threads=" +
                    std::to_string(threads));
  }
}

void check_reset_and_ids() {
  bytetrack::ByteTracker t;
  t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
  t.update({det(400, 100, 460, 260, 0.9f)}, 1080, 1920);
  t.reset();
  check_eq(t.frame_id(), 0, "reset clears the frame index");
  const auto out = t.update({det(100, 100, 160, 260, 0.9f)}, 1080, 1920);
  check_eq(out.size(), 1, "tracker usable after reset");
  check_eq(out[0].track_id, 3, "IDs continue monotonically after reset (no reuse)");
}

void check_invalid_input() {
  bytetrack::ByteTracker t;
  const int fid_before = t.frame_id();
  bool threw = false;
  try {
    t.update({det(0, 0, std::nanf(""), 100, 0.9f)}, 1080, 1920);
  } catch (const bytetrack::TrackingInputError&) {
    threw = true;
  }
  check(threw, "NaN coordinate rejected");
  threw = false;
  try {
    t.update({det(10, 10, 5, 100, 0.9f)}, 1080, 1920);
  } catch (const bytetrack::TrackingInputError&) {
    threw = true;
  }
  check(threw, "negative-area box rejected");
  threw = false;
  try {
    t.update({det(10, 10, 20, 100, std::nanf(""))}, 1080, 1920);
  } catch (const bytetrack::TrackingInputError&) {
    threw = true;
  }
  check(threw, "NaN score rejected");
  threw = false;
  try {
    t.update({}, 0, 1920);
  } catch (const bytetrack::TrackingInputError&) {
    threw = true;
  }
  check(threw, "zero image size rejected");
  check_eq(t.frame_id(), fid_before, "failed updates do not advance the frame index");
  threw = false;
  try {
    bytetrack::TrackingConfig bad;
    bad.match_thresh = 1.5;
    bytetrack::ByteTracker bad_tracker(bad);
  } catch (const bytetrack::TrackingConfigError&) {
    threw = true;
  }
  check(threw, "out-of-range threshold rejected in config");
}

void check_frame_temporaries_lifetime() {
  const auto frames = synth_stream(50, 2000, 987654321);
  bytetrack::ByteTracker t;
  size_t max_seen = 0;
  for (const auto& f : frames) {
    t.update(f, 1080, 1920);
    max_seen = std::max(max_seen, t.persistent_track_count());
  }
  // 50 persons with occlusion churn produce at most a few hundred persistent
  // tracks; 2000 frames * 50 detections would be 100k if temporaries leaked.
  check(max_seen < 2000, "persistent store bounded by lifecycle, not detection count");
  check(max_seen >= 50, "persistent store actually holds the lifecycle lists");
}

void check_replay_file_contract() {
  // The checks binary has no fixture file; exercise the reader's error
  // contract with paths that cannot exist (structural IO is covered by the
  // python contract test, which generates a real file).
  bool threw = false;
  try {
    bytetrack::read_replay_file("/nonexistent/bytetrack-checks.btcpp");
  } catch (const bytetrack::CliError&) {
    threw = true;
  }
  check(threw, "missing detections file reported as CLI error");
}

}  // namespace

int main() {
  std::signal(SIGALRM, [](int) { std::_Exit(124); });
  alarm(300);  // a deadlocked parallel region fails loudly instead of hanging
  check_parallel_runner();
  check_score_boundaries();
  check_empty_and_lifecycle();
  check_equal_cost_ties();
  check_thread_invariance();
  check_reset_and_ids();
  check_invalid_input();
  check_frame_temporaries_lifetime();
  check_replay_file_contract();
  if (g_failures == 0) {
    std::printf("ALL CHECKS PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
