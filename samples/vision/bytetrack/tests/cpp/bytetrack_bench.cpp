// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
/// Timing harness for the ByteTrack C++ tracker (evidence tooling; the
/// production CLI deliberately carries no benchmark flags).
///
/// Measurement contract (one unified window per repeat, per (threads, lanes)
/// configuration):
///   1. every lane warms up its own tracker independently (untimed);
///   2. lanes park on a start gate; the MAIN thread then records the single
///      window start: process CPU (CLOCK_PROCESS_CPUTIME_ID, all threads of
///      the process, read ONCE) + wall clock, and releases all lanes;
///   3. lanes run their timed frames (per-frame wall latency per lane);
///   4. when every lane has finished its timed frames the main thread records
///      CPU1/wall1 — a done latch, so thread join/destruction stays OUTSIDE
///      the window.
/// The window CPU is counted once for the whole process: per-lane clocks
/// overlap and summing them would double-count. Aggregate throughput is total
/// lane-frames / unified window wall. Per-frame cpu_ns is meaningful only for
/// a single lane (it then includes the tracker's internal workers, which is
/// the honest process cost); multi-lane rows carry 0 and the JSON says so.
/// Warmup, CSV flushing and tracker (re)construction are outside the window.
/// RSS is sampled before warmup and after the repeat. Parallel-region
/// engagement counters (engaged vs inline fallback, pool slots) are reported
/// per repeat: threads>1 with engaged==0 means every region fell below the
/// chunk thresholds and the run was effectively serial.
/// CSV rows: threads,lanes,repeat,lane,frame,n_in,n_out,wall_ns,cpu_ns.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli.hpp"
#include "tracking.hpp"

namespace {

struct Config {
  std::string detections, csv, json;
  int warmup = 100, frames = 1200, repeats = 2;
  int threads = 1, lanes = 1;
};

[[noreturn]] void usage_exit(const char* msg) {
  if (msg) std::fprintf(stderr, "error: %s\n", msg);
  std::fprintf(stderr,
               "usage: bytetrack_bench --detections FILE [--warmup N] [--frames N]\n"
               "                       [--repeats N] [--threads N] [--lanes N]\n"
               "                       [--csv FILE] [--json FILE]\n");
  std::exit(msg ? 2 : 0);
}

int next_int(const char* name, const char* value) {
  char* end = nullptr;
  const long v = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || v < 0 || v > 100000000) usage_exit(name);
  return static_cast<int>(v);
}

/// Process CPU clock (all threads of this process). Linux boards have
/// CLOCK_PROCESS_CPUTIME_ID; on hosts without it the CSV gets 0 and the
/// summary omits the cpu column meaning.
double cpu_clock_ns() {
#if defined(CLOCK_PROCESS_CPUTIME_ID)
  timespec ts{};
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return -1.0;
  return static_cast<double>(ts.tv_sec) * 1e9 + static_cast<double>(ts.tv_nsec);
#else
  return -1.0;
#endif
}

long resident_kb() {
#if defined(__linux__)
  std::ifstream statm("/proc/self/statm");
  long total = 0, resident_pages = 0;
  if (statm >> total >> resident_pages) return resident_pages * (sysconf(_SC_PAGESIZE) / 1024);
  return -1;
#else
  return -1;
#endif
}

struct TimedFrame {
  uint64_t wall_ns;
  uint64_t cpu_ns;  // 0 when per-frame CPU is not provided (lanes > 1)
  int n_out;
};

/// Start-gate / done-latch synchronisation between the lane workers and the
/// main thread (C++17 has no std::barrier). `abort` is the failure path: a
/// lane that throws sets it once; every waiter predicate accepts it, and the
/// main thread joins ALL workers before exiting non-zero. Predicates compare
/// with >= (never ==) so a worker incrementing a counter after abort was
/// raised cannot push it past the expected value and wedge a waiter.
struct WindowSync {
  std::mutex mu;
  std::condition_variable cv_warm, cv_start, cv_done;
  int warmed = 0;    // lanes that finished warmup
  int finished = 0;  // lanes that finished their timed frames
  bool go = false;
  bool abort = false;
};

struct RepeatResult {
  uint64_t window_wall_ns = 0;
  uint64_t window_cpu_ns = 0;  // 0 when the process CPU clock is unavailable
  long rss_start_kb = -1, rss_end_kb = -1;
  std::size_t engaged = 0, inline_regions = 0;
  unsigned slots = 0;
  std::vector<uint64_t> lane_mean_wall_ns, lane_max_wall_ns;
  std::vector<std::vector<TimedFrame>> timings;
};

}  // namespace

int main(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--detections" && has_value) cfg.detections = argv[++i];
    else if (a == "--csv" && has_value) cfg.csv = argv[++i];
    else if (a == "--json" && has_value) cfg.json = argv[++i];
    else if (a == "--warmup" && has_value) cfg.warmup = next_int(a.c_str(), argv[++i]);
    else if (a == "--frames" && has_value) cfg.frames = next_int(a.c_str(), argv[++i]);
    else if (a == "--repeats" && has_value) cfg.repeats = next_int(a.c_str(), argv[++i]);
    else if (a == "--threads" && has_value) cfg.threads = next_int(a.c_str(), argv[++i]);
    else if (a == "--lanes" && has_value) cfg.lanes = next_int(a.c_str(), argv[++i]);
    else if (a == "--help") usage_exit(nullptr);
    else usage_exit(a.c_str());
  }
  if (cfg.detections.empty()) usage_exit("--detections is required");
  if (cfg.threads < 1 || cfg.lanes < 1) usage_exit("--threads/--lanes must be >= 1");
  if (cfg.frames < 1 || cfg.repeats < 1) usage_exit("--frames/--repeats must be >= 1");

  const bytetrack::ReplayFile rf = bytetrack::read_replay_file(cfg.detections);
  if (cfg.warmup + cfg.frames > static_cast<int>(rf.n_frames))
    usage_exit("warmup+frames exceed the replay file's frame count");

  std::FILE* csv = nullptr;
  if (!cfg.csv.empty()) {
    csv = std::fopen(cfg.csv.c_str(), "wb");
    if (!csv) usage_exit("cannot open csv output");
    std::fprintf(csv, "threads,lanes,repeat,lane,frame,n_in,n_out,wall_ns,cpu_ns\n");
  }

  const bool per_frame_cpu = cfg.lanes == 1;  // multi-lane rows: CPU not provided
  std::vector<RepeatResult> results;
  results.reserve(static_cast<size_t>(cfg.repeats));
  for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
    std::vector<std::unique_ptr<bytetrack::ByteTracker>> trackers;
    trackers.reserve(cfg.lanes);
    for (int lane = 0; lane < cfg.lanes; ++lane)
      trackers.push_back(std::make_unique<bytetrack::ByteTracker>(
          bytetrack::TrackingConfig{}, static_cast<unsigned>(cfg.threads)));

    RepeatResult res;
    res.rss_start_kb = resident_kb();
    res.timings.resize(cfg.lanes);

    WindowSync sync;
    std::atomic<int> failed{0};
    std::vector<std::thread> workers;
    workers.reserve(cfg.lanes);
    for (int lane = 0; lane < cfg.lanes; ++lane) {
      workers.emplace_back([&, lane] {
        auto& tracker = trackers[lane];
        auto& out = res.timings[lane];
        out.reserve(static_cast<size_t>(cfg.frames));
        try {
          // Independent untimed warmup per lane.
          for (int f = 0; f < cfg.warmup; ++f)
            tracker->update(rf.frames[f], static_cast<int>(rf.img_h),
                            static_cast<int>(rf.img_w));
          {
            std::lock_guard<std::mutex> g(sync.mu);
            ++sync.warmed;
          }
          sync.cv_warm.notify_one();
          {  // park until the main thread opens the unified window (or aborts);
            // the abort re-check stays INSIDE the lock: a worker's catch sets
            // it under this mutex, and an unlocked read would be a data race.
            std::unique_lock<std::mutex> lk(sync.mu);
            sync.cv_start.wait(lk, [&] { return sync.go || sync.abort; });
            if (sync.abort) return;
          }
          for (int f = 0; f < cfg.frames; ++f) {
            const auto t0 = std::chrono::steady_clock::now();
            const double c0 = per_frame_cpu ? cpu_clock_ns() : -1.0;
            const auto tracks = tracker->update(rf.frames[cfg.warmup + f],
                                                static_cast<int>(rf.img_h),
                                                static_cast<int>(rf.img_w));
            const double c1 = per_frame_cpu ? cpu_clock_ns() : -1.0;
            const auto t1 = std::chrono::steady_clock::now();
            out.push_back(TimedFrame{
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
                c0 < 0 || c1 < 0 ? 0u : static_cast<uint64_t>(c1 - c0),
                static_cast<int>(tracks.size())});
          }
          {
            std::lock_guard<std::mutex> g(sync.mu);
            ++sync.finished;
          }
          sync.cv_done.notify_one();
        } catch (const std::exception&) {
          failed = 1;
          // Abort every waiter (a plain counter fixup could be undone by
          // another lane's ++ and wedge an == waiter forever).
          {
            std::lock_guard<std::mutex> g(sync.mu);
            sync.abort = true;
          }
          sync.cv_warm.notify_all();
          sync.cv_start.notify_all();
          sync.cv_done.notify_all();
        }
      });
    }

    // Unified window: open only after every lane finished warmup and is
    // parked on the start gate; close at the done latch, before any join.
    // Any lane aborting unwinds this path: join every worker first, then
    // fail the run (nothing is reported for an aborted repeat).
    bool aborted = false;
    {
      std::unique_lock<std::mutex> lk(sync.mu);
      sync.cv_warm.wait(lk, [&] { return sync.warmed >= cfg.lanes || sync.abort; });
      aborted = sync.abort;
    }
    if (aborted) {
      for (auto& w : workers) w.join();
      std::fprintf(stderr, "error: tracker raised during repeat %d (warmup)\n", repeat);
      if (csv) std::fclose(csv);
      return 1;
    }
    const double cpu0 = cpu_clock_ns();
    const auto wall0 = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> g(sync.mu);
      sync.go = true;
    }
    sync.cv_start.notify_all();
    {
      std::unique_lock<std::mutex> lk(sync.mu);
      sync.cv_done.wait(lk, [&] { return sync.finished >= cfg.lanes || sync.abort; });
      aborted = sync.abort;
    }
    if (aborted) {
      for (auto& w : workers) w.join();
      std::fprintf(stderr, "error: tracker raised during repeat %d (timed frames)\n", repeat);
      if (csv) std::fclose(csv);
      return 1;
    }
    const double cpu1 = cpu_clock_ns();
    const auto wall1 = std::chrono::steady_clock::now();
    for (auto& w : workers) w.join();  // destruction stays outside the window
    if (failed) {
      std::fprintf(stderr, "error: tracker raised during repeat %d\n", repeat);
      if (csv) std::fclose(csv);
      return 1;
    }
    res.window_wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(wall1 - wall0).count());
    res.window_cpu_ns = cpu0 < 0 || cpu1 < 0 ? 0u
                        : static_cast<uint64_t>(cpu1 - cpu0);
    res.rss_end_kb = resident_kb();
    for (int lane = 0; lane < cfg.lanes; ++lane) {
      const auto& u = trackers[lane]->parallel_usage();
      res.engaged += u.engaged;
      res.inline_regions += u.inline_fallback;
      res.slots = std::max(res.slots, u.slots);
      uint64_t mean = 0, maxv = 0;
      for (const TimedFrame& tf : res.timings[lane]) {
        mean += tf.wall_ns;
        maxv = std::max(maxv, tf.wall_ns);
      }
      res.lane_mean_wall_ns.push_back(res.timings[lane].empty()
                                          ? 0u
                                          : mean / res.timings[lane].size());
      res.lane_max_wall_ns.push_back(maxv);
    }

    // CSV writes happen outside the measurement window.
    for (int lane = 0; lane < cfg.lanes; ++lane)
      for (int f = 0; f < cfg.frames; ++f) {
        const TimedFrame& tf = res.timings[lane][f];
        if (csv)
          std::fprintf(csv, "%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n", cfg.threads, cfg.lanes, repeat,
                       lane, f, static_cast<int>(rf.frames[cfg.warmup + f].size()), tf.n_out,
                       static_cast<unsigned long long>(tf.wall_ns),
                       static_cast<unsigned long long>(tf.cpu_ns));
      }
    if (csv) std::fflush(csv);

    const double lane_frames = static_cast<double>(cfg.frames) * cfg.lanes;
    const double cpu_avail = res.window_cpu_ns > 0;
    std::printf(
        "repeat %d threads %d lanes %d: window wall %.4f ms/lane-frame, throughput %.1f "
        "lane-frame/s, process-cpu %.4f ms/lane-frame, rss %ld->%ld kB, parallel %zu engaged/%zu "
        "inline regions (pool slots %u%s)\n",
        repeat, cfg.threads, cfg.lanes,
        static_cast<double>(res.window_wall_ns) / 1e6 / lane_frames,
        lane_frames / (static_cast<double>(res.window_wall_ns) / 1e9),
        cpu_avail ? static_cast<double>(res.window_cpu_ns) / 1e6 / lane_frames : -1.0,
        res.rss_start_kb, res.rss_end_kb, res.engaged, res.inline_regions, res.slots,
        cfg.threads > 1 && res.engaged == 0 ? ", EFFECTIVELY SERIAL (all regions below chunk "
                                             "thresholds)" : "");
    results.push_back(std::move(res));
  }
  if (csv) std::fclose(csv);
  if (!cfg.json.empty()) {
    std::ofstream j(cfg.json);
    j << "{\n  \"detections\": \"" << cfg.detections << "\",\n"
      << "  \"warmup\": " << cfg.warmup << ", \"frames\": " << cfg.frames
      << ", \"repeats\": " << cfg.repeats << ",\n"
      << "  \"threads\": " << cfg.threads << ", \"lanes\": " << cfg.lanes << ",\n"
      << "  \"n_frames_file\": " << rf.n_frames << ", \"img\": [" << rf.img_h << ", " << rf.img_w
      << "],\n"
      << "  \"cpu_clock\": "
      << (cpu_clock_ns() >= 0 ? "\"CLOCK_PROCESS_CPUTIME_ID\"" : "\"unavailable\"")
      << ",\n"
      << "  \"measurement\": \"unified window: per-lane warmup, start gate opens after main "
         "records processCPU0+wall0, done latch closes before join; window cpu counted once "
         "(overlapping per-lane clocks are never summed)\",\n"
      << "  \"per_frame_cpu_note\": \""
      << (per_frame_cpu
              ? "cpu_ns valid (single lane, includes the tracker's internal workers)"
              : "not provided for lanes>1 rows (cpu_ns column is 0); use window process cpu")
      << "\",\n"
      << "  \"rss_kb_overall\": {\"start\": " << results.front().rss_start_kb
      << ", \"end\": " << results.back().rss_end_kb << "},\n"
      << "  \"repeats\": [\n";
    for (size_t r = 0; r < results.size(); ++r) {
      const RepeatResult& res = results[r];
      j << "    {\"repeat\": " << r
        << ", \"window_wall_ns\": " << res.window_wall_ns
        << ", \"window_cpu_ns\": " << res.window_cpu_ns
        << ", \"lane_frames\": " << cfg.frames * cfg.lanes
        << ", \"wall_ms_per_lane_frame\": "
        << static_cast<double>(res.window_wall_ns) / 1e6 / (cfg.frames * cfg.lanes)
        << ", \"cpu_ms_per_lane_frame\": "
        << static_cast<double>(res.window_cpu_ns) / 1e6 / (cfg.frames * cfg.lanes)
        << ", \"throughput_lane_frame_per_s\": "
        << static_cast<double>(cfg.frames * cfg.lanes) / (static_cast<double>(res.window_wall_ns) / 1e9)
        << ", \"lane_mean_wall_ns\": [";
      for (size_t l = 0; l < res.lane_mean_wall_ns.size(); ++l)
        j << (l ? ", " : "") << res.lane_mean_wall_ns[l];
      j << "], \"lane_max_wall_ns\": [";
      for (size_t l = 0; l < res.lane_max_wall_ns.size(); ++l)
        j << (l ? ", " : "") << res.lane_max_wall_ns[l];
      j << "], \"parallel_engaged_regions\": " << res.engaged
        << ", \"parallel_inline_regions\": " << res.inline_regions
        << ", \"pool_slots\": " << res.slots
        << ", \"effectively_serial\": " << (cfg.threads > 1 && res.engaged == 0 ? "true" : "false")
        << ", \"rss_kb_start\": " << res.rss_start_kb
        << ", \"rss_kb_end\": " << res.rss_end_kb << "}" << (r + 1 < results.size() ? "," : "")
        << "\n";
    }
    j << "  ]\n}\n";
  }
  return 0;
}
