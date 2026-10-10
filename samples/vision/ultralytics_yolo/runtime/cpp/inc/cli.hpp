/*
 * Copyright (c) 2026, D-Robotics.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Command-line surface of the ultralytics_yolo C++ runtime. The option
// surface, reporting API and per-task benchmark entries are declared here;
// the synchronized benchmark round/aggregate machinery is inline so it stays
// host-testable without any library. This header is free of board-SDK and
// OpenCV includes (the parser, image loading, renderers and benchmark
// drivers live in src/cli.cpp).

#ifndef YOLO_RUNTIME_CPP_INC_CLI_HPP_
#define YOLO_RUNTIME_CPP_INC_CLI_HPP_

#include "yolo.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace yolo {

// ---------------------------------------------------------------------------
// Options. Paths are positional: [model] [image] [output]; kebab-case flags
// (--task, --head, --score-thres, --nms-thres, ...) select behavior and
// thresholds. Missing positionals fall back to per-task defaults.
// ---------------------------------------------------------------------------

struct Options {
  std::string task = "detect";  // detect | segment | pose | classify | obb
  std::string model_path;
  std::string image_path;
  std::string output_path;
  std::string json_path;
  std::string head = "auto";  // auto | dfl | ltrb
  int resize_type = 1;        // 0 = resize, 1 = letterbox
  int opencv_threads = 0;     // 0 = all online CPUs
  int pipeline_streams = 1;   // 1 | 2
  int warmup = 20;
  int runs = 200;
  int rounds = 3;
  int topk = 5;
  int classes = 0;  // obb class channels; 0 keeps the task default (15)
  float score_threshold = 0.25f;
  float nms_threshold = 0.45f;
  float kpt_conf_threshold = 0.5f;
  float angle_sign = 1.0f;             // obb angle convention
  float angle_offset_degrees = 0.0f;   // obb angle offset, in degrees
  bool regularize_obb = true;
  bool benchmark = false;
  bool save_result = true;
  // Optional provenance recorded in the benchmark JSON when given.
  std::string runtime_source_sha256;
  std::string executable_sha256;
  // Bookkeeping for which fields the caller overrode; unset fields fall back
  // to the per-task defaults of the released samples.
  bool model_given = false;
  bool image_given = false;
  bool output_given = false;
  bool nms_given = false;
};

void print_usage(const char* program);
Options parse_options(int argc, char** argv);

// CLI-owned image loading. Holds the decoded BGR frame behind a pimpl so no
// OpenCV type leaks into this header.
class SourceImage {
 public:
  explicit SourceImage(const std::string& path);
  ~SourceImage();
  SourceImage(const SourceImage&) = delete;
  SourceImage& operator=(const SourceImage&) = delete;

  bool valid() const;
  int rows() const;
  int cols() const;
  const std::vector<uint8_t>& bgr() const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

// Per-task reporting: validation summary, rendered output image and logs.
// Definitions (and the OpenCV renderers) live in src/cli.cpp.
void report_detect(const Options& options, const YoloDetect::Prediction& run,
                   const SourceImage& source);
void report_segment(const Options& options,
                    const YoloSegment::Prediction& run);
void report_pose(const Options& options, const YoloPose::Prediction& run,
                 const SourceImage& source);
void report_classify(const Options& options,
                     const YoloClassify::Prediction& run);
void report_obb(const Options& options, const YoloObb::Prediction& run,
                const SourceImage& source);

// Bounded end-to-end benchmarks, one entry per task (validation-pinned
// result count, pipeline streams, warmup/runs/rounds). The model main
// constructed is reused as stream 0 and its validation Prediction provides
// the expected outputs per frame, so each process holds exactly
// options.pipeline_streams runtime contexts and performs exactly one
// validation predict. Returns a process exit code. Definitions in src/cli.cpp.
int run_detect_benchmark(const Options& options, const SourceImage& source,
                         YoloDetect* main_model,
                         const YoloDetect::Prediction& validation);
int run_segment_benchmark(const Options& options, const SourceImage& source,
                          YoloSegment* main_model,
                          const YoloSegment::Prediction& validation);
int run_pose_benchmark(const Options& options, const SourceImage& source,
                       YoloPose* main_model,
                       const YoloPose::Prediction& validation);
int run_classify_benchmark(const Options& options, const SourceImage& source,
                           YoloClassify* main_model,
                           const YoloClassify::Prediction& validation);
int run_obb_benchmark(const Options& options, const SourceImage& source,
                      YoloObb* main_model,
                      const YoloObb::Prediction& validation);

// ---------------------------------------------------------------------------
// Benchmark bookkeeping (pure, host-testable). The JSON emitted by
// write_benchmark_json() deliberately mirrors the end_to_end schema
// validated by model_zoo_web/scripts/build_catalog.py (pipeline_streams,
// runtime_submission_threads, timed_frames, aggregate_wall_ms,
// throughput_fps, per-stage mean/p50/p95/min/max).
// ---------------------------------------------------------------------------

struct StageTiming {
  double preprocess_ms = 0.0;
  double runtime_ms = 0.0;
  double postprocess_ms = 0.0;
  double end_to_end_ms = 0.0;
};

struct StageSamples {
  std::vector<double> preprocess;
  std::vector<double> runtime;
  std::vector<double> postprocess;
  std::vector<double> end_to_end;

  void add(const StageTiming& timing) {
    preprocess.push_back(timing.preprocess_ms);
    runtime.push_back(timing.runtime_ms);
    postprocess.push_back(timing.postprocess_ms);
    end_to_end.push_back(timing.end_to_end_ms);
  }

  void append(const StageSamples& other) {
    preprocess.insert(preprocess.end(), other.preprocess.begin(),
                      other.preprocess.end());
    runtime.insert(runtime.end(), other.runtime.begin(), other.runtime.end());
    postprocess.insert(postprocess.end(), other.postprocess.begin(),
                       other.postprocess.end());
    end_to_end.insert(end_to_end.end(), other.end_to_end.begin(),
                      other.end_to_end.end());
  }
};

struct BenchmarkRound {
  StageSamples samples;
  double wall_ms = 0.0;
  size_t completed_frames = 0;
};

// Identity and methodology fields embedded in the benchmark JSON.
struct BenchmarkMeta {
  std::string model_path;
  std::string image_path;
  std::string implementation;
  std::string output_kind = "detections";
  std::string timing_scope = "in_memory_bgr_to_detections";
  std::string cpu_thread_policy = "all_online";
  int pipeline_streams = 1;
  int online_cpu_threads = 0;
  int opencv_threads = 0;
  int resize_type = -1;
  float angle_sign = 1.0f;
  float angle_offset_degrees = 0.0f;
  bool regularize_obb = true;
  std::string runtime_source_sha256;
  std::string executable_sha256;
  int warmup_frames_per_round = 20;
  int runs_per_round = 200;
  int rounds = 3;
  float score_threshold = 0.25f;
  float nms_threshold = 0.7f;
};

// Benchmark controls derived from Options by the benchmark drivers. The
// field set matches the shared task-benchmark vocabulary so the round
// runner and JSON writer stay task-agnostic.
struct BenchmarkOptions {
  bool enabled = false;
  bool save_result = true;
  int warmup_frames = 20;
  int runs_per_round = 200;
  int rounds = 3;
  int pipeline_streams = 1;
  // 0 means all online CPUs; otherwise a positive OpenCV thread count.
  int opencv_threads = 0;
  int resize_type = -1;
  int classes = 0;
  float score_threshold = 0.25f;
  float nms_threshold = 0.7f;
  float angle_sign = 1.0f;
  float angle_offset_degrees = 0.0f;
  bool regularize_obb = true;
  std::string runtime_source_sha256;
  std::string executable_sha256;
  std::string json_path;
};

// A per-stream callback executes one complete in-memory frame pipeline and
// reports a task result count for determinism checks. The callback and its
// model/tensors must be private to that stream.
typedef std::function<bool(StageTiming*, size_t*, std::string*)>
    BenchmarkPipeline;

// `expected_outputs` may be `kAnyOutputCount` to disable result-count
// stability checking in run_benchmark_round below.
static const size_t kAnyOutputCount = std::numeric_limits<size_t>::max();

struct Statistics {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

inline double percentile(const std::vector<double>& values, double fraction) {
  if (values.empty()) return 0.0;
  std::vector<double> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  const double position = (sorted.size() - 1) * fraction;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  if (lower == upper) return sorted[lower];
  const double weight = position - lower;
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

inline Statistics summarize(const std::vector<double>& values) {
  Statistics result;
  if (values.empty()) return result;
  result.mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  result.p50 = percentile(values, 0.50);
  result.p95 = percentile(values, 0.95);
  result.min = *std::min_element(values.begin(), values.end());
  result.max = *std::max_element(values.begin(), values.end());
  return result;
}

namespace benchmark_detail {

inline std::string json_escape(const std::string& value) {
  std::string result;
  for (size_t i = 0; i < value.size(); ++i) {
    const char ch = value[i];
    if (ch == '\\' || ch == '"') result.push_back('\\');
    if (ch == '\n') {
      result += "\\n";
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

inline void write_metric(std::ofstream* stream, const std::string& name,
                         const std::vector<double>& values, bool comma) {
  const Statistics stats = summarize(values);
  *stream << "    \"" << name << "\": {\"mean\": " << stats.mean
          << ", \"p50\": " << stats.p50 << ", \"p95\": " << stats.p95
          << ", \"min\": " << stats.min << ", \"max\": " << stats.max << "}"
          << (comma ? "," : "") << "\n";
}

}  // namespace benchmark_detail

// Writes the aggregate report. Returns false when `path` cannot be opened.
// `outputs_per_frame` is emitted as such and, for the detect output kind,
// additionally as `detections_per_frame`; the angle options are recorded for
// oriented boxes and score/NMS are omitted for Top-K classification.
inline bool write_benchmark_json(const std::string& path,
                                 const BenchmarkMeta& meta,
                                 const StageSamples& samples,
                                 size_t outputs_per_frame,
                                 size_t completed_frames,
                                 double aggregate_wall_ms) {
  std::ofstream stream(path.c_str());
  if (!stream) {
    std::cerr << "[ERROR] Cannot write benchmark JSON: " << path << std::endl;
    return false;
  }
  stream << std::fixed << std::setprecision(6);
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"model\": \"" << benchmark_detail::json_escape(meta.model_path)
         << "\",\n"
         << "  \"image\": \"" << benchmark_detail::json_escape(meta.image_path)
         << "\",\n"
         << "  \"implementation\": \""
         << benchmark_detail::json_escape(meta.implementation) << "\",\n"
         << "  \"output_kind\": \""
         << benchmark_detail::json_escape(meta.output_kind) << "\",\n"
         << "  \"timing_scope\": \""
         << benchmark_detail::json_escape(meta.timing_scope) << "\",\n"
         << "  \"pipeline_streams\": " << meta.pipeline_streams << ",\n"
         << "  \"runtime_submission_threads\": " << meta.pipeline_streams
         << ",\n"
         << "  \"cpu_thread_policy\": \""
         << benchmark_detail::json_escape(meta.cpu_thread_policy) << "\",\n"
         << "  \"online_cpu_threads\": " << meta.online_cpu_threads << ",\n"
         << "  \"opencv_threads\": " << meta.opencv_threads << ",\n"
         << (meta.runtime_source_sha256.empty()
                 ? ""
                 : "  \"runtime_source_sha256\": \"" +
                       benchmark_detail::json_escape(meta.runtime_source_sha256) +
                       "\",\n")
         << (meta.executable_sha256.empty()
                 ? ""
                 : "  \"executable_sha256\": \"" +
                       benchmark_detail::json_escape(meta.executable_sha256) +
                       "\",\n")
         << (meta.resize_type >= 0
                 ? "  \"resize_type\": " + std::to_string(meta.resize_type) +
                       ",\n"
                 : "")
         << (meta.output_kind == "rotated_boxes"
                 ? "  \"angle_sign\": " + std::to_string(meta.angle_sign) +
                       ",\n  \"angle_offset_degrees\": " +
                       std::to_string(meta.angle_offset_degrees) +
                       ",\n  \"regularize_obb\": " +
                       std::string(meta.regularize_obb ? "true" : "false") +
                       ",\n"
                 : "")
         << "  \"warmup_frames_per_round\": " << meta.warmup_frames_per_round
         << ",\n"
         << "  \"runs_per_round\": " << meta.runs_per_round << ",\n"
         << "  \"frames_per_stream_per_round\": " << meta.runs_per_round
         << ",\n"
         << "  \"rounds\": " << meta.rounds << ",\n"
         << "  \"timed_frames\": " << completed_frames << ",\n"
         << "  \"aggregate_wall_ms\": " << aggregate_wall_ms << ",\n"
         << "  \"outputs_per_frame\": " << outputs_per_frame << ",\n"
         << (meta.output_kind == "detections"
                 ? "  \"detections_per_frame\": " +
                       std::to_string(outputs_per_frame) + ",\n"
                 : "")
         << (meta.output_kind == "topk_predictions"
                 ? ""
                 : "  \"score_threshold\": " +
                       std::to_string(meta.score_threshold) + ",\n")
         << (meta.output_kind == "topk_predictions"
                 ? ""
                 : "  \"nms_threshold\": " +
                       std::to_string(meta.nms_threshold) + ",\n")
         << "  \"metrics_ms\": {\n";
  benchmark_detail::write_metric(&stream, "preprocess", samples.preprocess,
                                 true);
  benchmark_detail::write_metric(&stream, "runtime", samples.runtime, true);
  benchmark_detail::write_metric(&stream, "postprocess", samples.postprocess,
                                 true);
  benchmark_detail::write_metric(&stream, "end_to_end", samples.end_to_end,
                                 false);
  stream << "  },\n"
         << "  \"throughput_fps\": "
         << (aggregate_wall_ms > 0.0
                 ? completed_frames * 1000.0 / aggregate_wall_ms
                 : 0.0)
         << "\n}\n";
  return true;
}

// Run one synchronized round over independent task pipelines.
// `frames_per_stream` is the number of frames each pipeline processes;
// throughput uses all completed frames divided by the common wall time.
// All workers pass a start gate so the wall clock covers the full round.
inline bool run_benchmark_round(const std::vector<BenchmarkPipeline>& pipelines,
                                int frames_per_stream,
                                size_t expected_outputs, bool collect_timing,
                                BenchmarkRound* result, std::string* error) {
  if (pipelines.empty() || frames_per_stream < 0 ||
      (collect_timing && result == nullptr)) {
    if (error) *error = "invalid benchmark round arguments";
    return false;
  }

  std::vector<StageSamples> stream_samples(pipelines.size());
  std::vector<std::thread> workers;
  workers.reserve(pipelines.size());
  std::atomic<bool> failed(false);
  std::mutex gate_mutex;
  std::mutex error_mutex;
  std::condition_variable ready_condition;
  std::condition_variable start_condition;
  size_t ready_workers = 0;
  bool start = false;
  std::string error_message;

  const auto fail = [&](const std::string& message) {
    if (!failed.exchange(true)) {
      std::lock_guard<std::mutex> lock(error_mutex);
      error_message = message;
    }
  };

  for (size_t stream = 0; stream < pipelines.size(); ++stream) {
    workers.push_back(std::thread([&, stream]() {
      {
        std::unique_lock<std::mutex> lock(gate_mutex);
        ++ready_workers;
        ready_condition.notify_one();
        start_condition.wait(lock, [&]() { return start; });
      }
      try {
        for (int frame = 0; frame < frames_per_stream && !failed.load(); ++frame) {
          StageTiming timing;
          size_t output_count = 0;
          std::string pipeline_error;
          if (!pipelines[stream](&timing, &output_count, &pipeline_error)) {
            fail(pipeline_error.empty()
                     ? "pipeline stream " + std::to_string(stream) + " failed"
                     : "pipeline stream " + std::to_string(stream) + ": " +
                           pipeline_error);
            break;
          }
          if (expected_outputs != kAnyOutputCount &&
              output_count != expected_outputs) {
            fail("pipeline stream " + std::to_string(stream) +
                 " output count changed: expected " +
                 std::to_string(expected_outputs) + ", got " +
                 std::to_string(output_count));
            break;
          }
          if (collect_timing) stream_samples[stream].add(timing);
        }
      } catch (const std::exception& exception) {
        fail("pipeline stream " + std::to_string(stream) +
             " raised: " + exception.what());
      } catch (...) {
        fail("pipeline stream " + std::to_string(stream) +
             " raised an unknown exception");
      }
    }));
  }

  typedef std::chrono::steady_clock Clock;
  Clock::time_point wall_start;
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    ready_condition.wait(lock, [&]() { return ready_workers == pipelines.size(); });
    wall_start = Clock::now();
    start = true;
  }
  start_condition.notify_all();
  for (size_t i = 0; i < workers.size(); ++i) workers[i].join();
  const Clock::time_point wall_end = Clock::now();

  if (failed.load()) {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (error) *error = error_message;
    return false;
  }
  if (collect_timing) {
    result->wall_ms = std::chrono::duration_cast<
        std::chrono::duration<double, std::milli> >(wall_end - wall_start)
        .count();
    result->samples = StageSamples();
    for (size_t stream = 0; stream < stream_samples.size(); ++stream) {
      result->samples.append(stream_samples[stream]);
    }
    result->completed_frames = result->samples.end_to_end.size();
  }
  return true;
}

// Execute all requested warmup/measured rounds, report stage statistics and
// optionally write the catalog-shaped JSON record. `outputs_per_frame` is
// checked for stability and emitted with the task's `output_kind`.
inline bool run_benchmark(const std::vector<BenchmarkPipeline>& pipelines,
                          const BenchmarkOptions& options,
                          size_t outputs_per_frame, BenchmarkMeta meta) {
  if (pipelines.empty() || pipelines.size() !=
                               static_cast<size_t>(options.pipeline_streams)) {
    std::cerr << "[ERROR] Benchmark pipeline count does not match options"
              << std::endl;
    return false;
  }
  const auto print = [](const char* label, const std::vector<double>& values) {
    const Statistics stats = summarize(values);
    std::cout << std::left << std::setw(13) << label << std::right
              << std::fixed << std::setprecision(3) << " mean=" << std::setw(8)
              << stats.mean << " ms  p50=" << std::setw(8) << stats.p50
              << "  p95=" << std::setw(8) << stats.p95 << "  min="
              << std::setw(8) << stats.min << "  max=" << std::setw(8)
              << stats.max << std::endl;
  };
  StageSamples aggregate;
  double aggregate_wall_ms = 0.0;
  size_t completed_frames = 0;
  for (int round = 0; round < options.rounds; ++round) {
    std::string error;
    if (!run_benchmark_round(pipelines, options.warmup_frames,
                             outputs_per_frame, false, nullptr, &error)) {
      std::cerr << "[ERROR] Warmup failed: " << error << std::endl;
      return false;
    }
    BenchmarkRound measured;
    if (!run_benchmark_round(pipelines, options.runs_per_round,
                             outputs_per_frame, true, &measured, &error)) {
      std::cerr << "[ERROR] Benchmark round failed: " << error << std::endl;
      return false;
    }
    aggregate.append(measured.samples);
    aggregate_wall_ms += measured.wall_ms;
    completed_frames += measured.completed_frames;

    std::cout << "\n[ROUND " << round + 1 << "/" << options.rounds << "]"
              << std::endl;
    print("preprocess", measured.samples.preprocess);
    print("runtime", measured.samples.runtime);
    print("postprocess", measured.samples.postprocess);
    print("end_to_end", measured.samples.end_to_end);
    std::cout << "wall_time     " << std::fixed << std::setprecision(3)
              << measured.wall_ms << " ms for " << measured.completed_frames
              << " frames" << std::endl;
    std::cout << "throughput    " << std::fixed << std::setprecision(3)
              << (measured.wall_ms > 0.0
                      ? measured.completed_frames * 1000.0 / measured.wall_ms
                      : 0.0)
              << " aggregate fps" << std::endl;
  }

  meta.pipeline_streams = options.pipeline_streams;
  meta.warmup_frames_per_round = options.warmup_frames;
  meta.runs_per_round = options.runs_per_round;
  meta.rounds = options.rounds;
  meta.score_threshold = options.score_threshold;
  meta.nms_threshold = options.nms_threshold;
  meta.angle_sign = options.angle_sign;
  meta.angle_offset_degrees = options.angle_offset_degrees;
  meta.regularize_obb = options.regularize_obb;
  meta.runtime_source_sha256 = options.runtime_source_sha256;
  meta.executable_sha256 = options.executable_sha256;
  std::cout << "\n[AGGREGATE " << aggregate.end_to_end.size() << " frames]"
            << std::endl;
  print("preprocess", aggregate.preprocess);
  print("runtime", aggregate.runtime);
  print("postprocess", aggregate.postprocess);
  print("end_to_end", aggregate.end_to_end);
  std::cout << "wall_time     " << std::fixed << std::setprecision(3)
            << aggregate_wall_ms << " ms for " << completed_frames
            << " frames" << std::endl;
  std::cout << "throughput    " << std::fixed << std::setprecision(3)
            << (aggregate_wall_ms > 0.0
                    ? completed_frames * 1000.0 / aggregate_wall_ms
                    : 0.0)
            << " aggregate fps" << std::endl;

  if (!options.json_path.empty() &&
      !write_benchmark_json(options.json_path, meta, aggregate,
                            outputs_per_frame, completed_frames,
                            aggregate_wall_ms)) {
    return false;
  }
  return true;
}

}  // namespace yolo

#endif  // YOLO_RUNTIME_CPP_INC_CLI_HPP_
