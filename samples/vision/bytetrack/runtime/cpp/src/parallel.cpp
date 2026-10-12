// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "parallel.hpp"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace bytetrack {

class Worker {
 public:
  std::thread thread;
  unsigned slot = 0;
};

namespace {

struct SyncState {
  std::mutex mu;
  std::condition_variable cv_start;   // generation bump wakes workers
  std::condition_variable cv_done;    // last worker wakes the caller
  const std::function<void(std::size_t, std::size_t)>* fn = nullptr;
  std::size_t items = 0;
  std::size_t generation = 0;  // incremented per parallel region
  std::size_t done = 0;
  std::exception_ptr error;
  bool stop = false;
};

}  // namespace

ParallelRunner::ParallelRunner(unsigned threads) {
  const unsigned hw = std::thread::hardware_concurrency();
  slots_ = std::max(1u, threads ? std::min(threads, hw ? hw : 1u) : 1u);
  if (slots_ > 1) {
    auto* sync = new SyncState;
    state_ = sync;
    workers_.reserve(slots_ - 1);
    for (unsigned slot = 1; slot < slots_; ++slot) {
      auto* w = new Worker;
      w->slot = slot;
      try {
        w->thread = std::thread([this, slot] { worker_loop(slot); });
      } catch (...) {
        delete w;
        shutdown_join();  // join what already started, then fail loudly
        throw;
      }
      workers_.push_back(w);
    }
  }
}

ParallelRunner::~ParallelRunner() { shutdown_join(); }

void ParallelRunner::shutdown_join() {
  if (state_) {
    {
      auto* sync = static_cast<SyncState*>(state_);
      std::lock_guard<std::mutex> g(sync->mu);
      sync->stop = true;
    }
    auto* sync = static_cast<SyncState*>(state_);
    sync->cv_start.notify_all();
  }
  for (Worker* w : workers_) {
    if (w->thread.joinable()) w->thread.join();
    delete w;
  }
  workers_.clear();
  delete static_cast<SyncState*>(state_);
  state_ = nullptr;
}

void ParallelRunner::chunk_call(unsigned slot, std::size_t items,
                                const std::function<void(std::size_t, std::size_t)>& fn) {
  const std::size_t n = slots_;
  const std::size_t base = items / n;
  const std::size_t rem = items % n;
  const std::size_t begin = slot * base + std::min<std::size_t>(slot, rem);
  const std::size_t len = base + (slot < rem ? 1 : 0);
  try {
    fn(begin, begin + len);
  } catch (...) {
    auto* sync = static_cast<SyncState*>(state_);
    std::lock_guard<std::mutex> g(sync->mu);
    if (!sync->error) sync->error = std::current_exception();
  }
}

bool ParallelRunner::run(std::size_t items, std::size_t min_chunk,
                         const std::function<void(std::size_t, std::size_t)>& fn) {
  if (slots_ == 1 || items < static_cast<std::size_t>(slots_) * min_chunk) {
    fn(0, items);
    return false;
  }
  auto* sync = static_cast<SyncState*>(state_);
  {
    std::lock_guard<std::mutex> g(sync->mu);
    sync->fn = &fn;
    sync->items = items;
    sync->done = 0;
    sync->error = nullptr;
    ++sync->generation;
  }
  sync->cv_start.notify_all();
  chunk_call(0, items, fn);  // caller owns chunk 0
  {
    std::unique_lock<std::mutex> lk(sync->mu);
    sync->cv_done.wait(lk, [&] { return sync->done == workers_.size(); });
    if (sync->error) std::rethrow_exception(sync->error);
  }
  return true;
}

void ParallelRunner::worker_loop(unsigned slot) {
  auto* sync = static_cast<SyncState*>(state_);
  std::size_t seen = 0;
  for (;;) {
    const std::function<void(std::size_t, std::size_t)>* fn = nullptr;
    std::size_t items = 0;
    {
      std::unique_lock<std::mutex> lk(sync->mu);
      sync->cv_start.wait(lk, [&] { return sync->stop || sync->generation != seen; });
      if (sync->stop) return;
      seen = sync->generation;
      fn = sync->fn;
      items = sync->items;
    }
    chunk_call(slot, items, *fn);
    {
      std::lock_guard<std::mutex> g(sync->mu);
      ++sync->done;
      if (sync->done == workers_.size()) sync->cv_done.notify_one();
    }
  }
}

}  // namespace bytetrack
