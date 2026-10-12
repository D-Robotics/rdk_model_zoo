// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// Fixed-slot parallel runner for the tracker's embarrassingly parallel
/// sections (per-track Kalman prediction, IoU matrix rows).
///
/// Contract: run() splits [0, items) into exactly `nthreads` contiguous
/// chunks and executes chunk i on slot i (the caller is slot 0), so every
/// item is processed exactly once regardless of scheduling, and results
/// never depend on the thread count. Sections below nthreads * min_chunk run
/// inline on the caller. The first exception from any chunk is rethrown to
/// the caller after the region joins; worker threads never let exceptions
/// escape (no terminate, no deadlock).
#ifndef BYTETRACK_RUNTIME_CPP_PARALLEL_HPP_
#define BYTETRACK_RUNTIME_CPP_PARALLEL_HPP_

#include <cstddef>
#include <functional>
#include <vector>

namespace bytetrack {

class ParallelRunner {
 public:
  /// Spawns threads-1 workers (caller is slot 0); threads is clamped to
  /// [1, hardware_concurrency]. Throws only if thread creation fails, after
  /// joining everything already created.
  explicit ParallelRunner(unsigned threads);
  ~ParallelRunner();

  ParallelRunner(const ParallelRunner&) = delete;
  ParallelRunner& operator=(const ParallelRunner&) = delete;

  /// Number of slots (caller included) — also the number of chunks a
  /// parallel region splits into.
  unsigned slots() const { return slots_; }

  /// Apply fn(begin, end) to each chunk; see the file comment for the exact
  /// partition. Returns false when the work ran inline (below the chunk
  /// threshold), which lets callers and tests tell real parallelism apart
  /// from the sequential fallback.
  bool run(std::size_t items, std::size_t min_chunk,
           const std::function<void(std::size_t, std::size_t)>& fn);

 private:
  void worker_loop(unsigned slot);
  void chunk_call(unsigned slot, std::size_t items,
                  const std::function<void(std::size_t, std::size_t)>& fn);
  void shutdown_join();

  unsigned slots_ = 1;
  std::vector<class Worker*> workers_;  // pimpl-ish; owned, joined in dtor
  void* state_ = nullptr;              // internal synchronization block
};

}  // namespace bytetrack

#endif  // BYTETRACK_RUNTIME_CPP_PARALLEL_HPP_
