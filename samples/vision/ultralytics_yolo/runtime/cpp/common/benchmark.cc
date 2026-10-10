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

#include "benchmark.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace yolo {

namespace {

bool parse_int(const std::string& value, int minimum, int* output) {
  if (value.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (errno != 0 || end == value.c_str() || *end != '\0' ||
      parsed < minimum || parsed > std::numeric_limits<int>::max()) {
    return false;
  }
  *output = static_cast<int>(parsed);
  return true;
}

bool parse_float(const std::string& value, float* output) {
  if (value.empty()) return false;
  errno = 0;
  char* end = nullptr;
  const float parsed = std::strtof(value.c_str(), &end);
  if (errno != 0 || end == value.c_str() || *end != '\0' ||
      !std::isfinite(parsed)) {
    return false;
  }
  *output = parsed;
  return true;
}

bool is_sha256(const std::string& value) {
  if (value.size() != 64) return false;
  for (size_t i = 0; i < value.size(); ++i) {
    if (!std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
  }
  return true;
}

bool take_value(int* index, int argc, char** argv, const std::string& option,
                std::string* value, std::string* error) {
  if (*index + 1 >= argc) {
    if (error) *error = "missing value after " + option;
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

}  // namespace

bool parse_benchmark_options(int argc, char** argv, int first_option,
                             BenchmarkOptions* options,
                             std::string* error) {
  if (options == nullptr || first_option < 1 || first_option > argc) {
    if (error) *error = "invalid benchmark option parser arguments";
    return false;
  }
  for (int i = first_option; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--benchmark") {
      options->enabled = true;
    } else if (arg == "--no-save") {
      options->save_result = false;
    } else if (arg == "--no-regularize") {
      options->regularize_obb = false;
    } else if (arg == "--warmup" || arg == "--runs" || arg == "--rounds" ||
               arg == "--pipeline-streams" || arg == "--opencv-threads" ||
               arg == "--resize-type" || arg == "--classes" ||
               arg == "--angle-sign" || arg == "--angle-offset" ||
               arg == "--score" || arg == "--nms" || arg == "--json" ||
               arg == "--runtime-source-sha256" ||
               arg == "--executable-sha256") {
      std::string value;
      if (!take_value(&i, argc, argv, arg, &value, error)) return false;
      if (arg == "--warmup") {
        if (!parse_int(value, 0, &options->warmup_frames)) {
          if (error) *error = "--warmup must be a non-negative integer";
          return false;
        }
      } else if (arg == "--runs") {
        if (!parse_int(value, 1, &options->runs_per_round)) {
          if (error) *error = "--runs must be a positive integer";
          return false;
        }
      } else if (arg == "--rounds") {
        if (!parse_int(value, 1, &options->rounds)) {
          if (error) *error = "--rounds must be a positive integer";
          return false;
        }
      } else if (arg == "--pipeline-streams") {
        if (!parse_int(value, 1, &options->pipeline_streams) ||
            options->pipeline_streams > 2) {
          if (error) *error = "--pipeline-streams must be 1 or 2";
          return false;
        }
      } else if (arg == "--opencv-threads") {
        if (value == "all") {
          options->opencv_threads = 0;
        } else if (!parse_int(value, 1, &options->opencv_threads)) {
          if (error) *error = "--opencv-threads must be 'all' or positive";
          return false;
        }
      } else if (arg == "--resize-type") {
        if (!parse_int(value, 0, &options->resize_type) ||
            options->resize_type > 1) {
          if (error) *error = "--resize-type must be 0 or 1";
          return false;
        }
      } else if (arg == "--classes") {
        if (!parse_int(value, 1, &options->classes)) {
          if (error) *error = "--classes must be a positive integer";
          return false;
        }
      } else if (arg == "--angle-sign") {
        if (!parse_float(value, &options->angle_sign)) {
          if (error) *error = "--angle-sign must be finite";
          return false;
        }
      } else if (arg == "--angle-offset") {
        if (!parse_float(value, &options->angle_offset_degrees)) {
          if (error) *error = "--angle-offset must be finite degrees";
          return false;
        }
      } else if (arg == "--score") {
        if (!parse_float(value, &options->score_threshold) ||
            options->score_threshold <= 0.0f ||
            options->score_threshold >= 1.0f) {
          if (error) *error = "--score must be between 0 and 1";
          return false;
        }
      } else if (arg == "--nms") {
        if (!parse_float(value, &options->nms_threshold) ||
            options->nms_threshold < 0.0f ||
            options->nms_threshold > 1.0f) {
          if (error) *error = "--nms must be between 0 and 1";
          return false;
        }
      } else if (arg == "--runtime-source-sha256") {
        if (!is_sha256(value)) {
          if (error) *error = "--runtime-source-sha256 must be 64 hex digits";
          return false;
        }
        options->runtime_source_sha256 = value;
      } else if (arg == "--executable-sha256") {
        if (!is_sha256(value)) {
          if (error) *error = "--executable-sha256 must be 64 hex digits";
          return false;
        }
        options->executable_sha256 = value;
      } else {
        options->json_path = value;
        if (options->json_path.empty()) {
          if (error) *error = "--json path cannot be empty";
          return false;
        }
      }
    } else {
      if (error) *error = "unknown option: " + arg;
      return false;
    }
  }
  return true;
}

bool run_benchmark(const std::vector<BenchmarkPipeline>& pipelines,
                   const BenchmarkOptions& options, size_t outputs_per_frame,
                   BenchmarkMeta meta) {
  if (pipelines.empty() || pipelines.size() !=
                               static_cast<size_t>(options.pipeline_streams)) {
    std::cerr << "[ERROR] Benchmark pipeline count does not match options"
              << std::endl;
    return false;
  }
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

    const Statistics pre = summarize(measured.samples.preprocess);
    const Statistics run = summarize(measured.samples.runtime);
    const Statistics post = summarize(measured.samples.postprocess);
    const Statistics e2e = summarize(measured.samples.end_to_end);
    const auto print = [](const char* label, const Statistics& stats) {
      std::cout << std::left << std::setw(13) << label << std::right
                << std::fixed << std::setprecision(3) << " mean="
                << std::setw(8) << stats.mean << " ms  p50=" << std::setw(8)
                << stats.p50 << "  p95=" << std::setw(8) << stats.p95
                << "  min=" << std::setw(8) << stats.min << "  max="
                << std::setw(8) << stats.max << std::endl;
    };
    std::cout << "[ROUND " << round + 1 << "/" << options.rounds << "]"
              << std::endl;
    print("preprocess", pre);
    print("runtime", run);
    print("postprocess", post);
    print("end_to_end", e2e);
    std::cout << "wall_time " << std::fixed << std::setprecision(3)
              << measured.wall_ms << " ms for " << measured.completed_frames
              << " frames; throughput "
              << (measured.wall_ms > 0.0
                      ? measured.completed_frames * 1000.0 / measured.wall_ms
                      : 0.0)
              << " frames/s" << std::endl;
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
  std::cout << "[AGGREGATE] " << completed_frames << " frames; throughput "
            << std::fixed << std::setprecision(3)
            << (aggregate_wall_ms > 0.0
                    ? completed_frames * 1000.0 / aggregate_wall_ms
                    : 0.0)
            << " frames/s" << std::endl;
  if (!options.json_path.empty() &&
      !write_benchmark_json(options.json_path, meta, aggregate,
                            outputs_per_frame, completed_frames,
                            aggregate_wall_ms)) {
    return false;
  }
  return true;
}

bool run_benchmark_round(const std::vector<BenchmarkPipeline>& pipelines,
                         int frames_per_stream, size_t expected_outputs,
                         bool collect_timing, BenchmarkRound* result,
                         std::string* error) {
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
        fail("pipeline stream " + std::to_string(stream) + " raised: " +
             exception.what());
      } catch (...) {
        fail("pipeline stream " + std::to_string(stream) +
             " raised an unknown exception");
      }
    }));
  }

  std::chrono::steady_clock::time_point wall_start;
  {
    std::unique_lock<std::mutex> lock(gate_mutex);
    ready_condition.wait(lock, [&]() { return ready_workers == pipelines.size(); });
    wall_start = std::chrono::steady_clock::now();
    start = true;
  }
  start_condition.notify_all();
  for (size_t i = 0; i < workers.size(); ++i) workers[i].join();
  const std::chrono::steady_clock::time_point wall_end =
      std::chrono::steady_clock::now();

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

double percentile(const std::vector<double>& values, double fraction) {
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

Statistics summarize(const std::vector<double>& values) {
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

namespace {

std::string json_escape(const std::string& value) {
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

void write_metric(std::ofstream* stream, const std::string& name,
                  const std::vector<double>& values, bool comma) {
  const Statistics stats = summarize(values);
  *stream << "    \"" << name << "\": {\"mean\": " << stats.mean
          << ", \"p50\": " << stats.p50 << ", \"p95\": " << stats.p95
          << ", \"min\": " << stats.min << ", \"max\": " << stats.max << "}"
          << (comma ? "," : "") << "\n";
}

}  // namespace

bool write_benchmark_json(const std::string& path, const BenchmarkMeta& meta,
                          const StageSamples& samples, size_t detections,
                          size_t completed_frames, double aggregate_wall_ms) {
  std::ofstream stream(path.c_str());
  if (!stream) {
    std::cerr << "[ERROR] Cannot write benchmark JSON: " << path << std::endl;
    return false;
  }
  stream << std::fixed << std::setprecision(6);
  stream << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"model\": \"" << json_escape(meta.model_path) << "\",\n"
         << "  \"image\": \"" << json_escape(meta.image_path) << "\",\n"
         << "  \"implementation\": \"" << json_escape(meta.implementation)
         << "\",\n"
         << "  \"output_kind\": \"" << json_escape(meta.output_kind)
         << "\",\n"
         << "  \"timing_scope\": \"" << json_escape(meta.timing_scope)
         << "\",\n"
         << "  \"pipeline_streams\": " << meta.pipeline_streams << ",\n"
         << "  \"runtime_submission_threads\": " << meta.pipeline_streams
         << ",\n"
         << "  \"cpu_thread_policy\": \"" << json_escape(meta.cpu_thread_policy)
         << "\",\n"
         << "  \"online_cpu_threads\": " << meta.online_cpu_threads << ",\n"
         << "  \"opencv_threads\": " << meta.opencv_threads << ",\n"
         << (meta.runtime_source_sha256.empty()
                 ? ""
                 : "  \"runtime_source_sha256\": \"" +
                       json_escape(meta.runtime_source_sha256) + "\",\n")
         << (meta.executable_sha256.empty()
                 ? ""
                 : "  \"executable_sha256\": \"" +
                       json_escape(meta.executable_sha256) + "\",\n")
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
         << "  \"outputs_per_frame\": " << detections << ",\n"
         << (meta.output_kind == "detections"
                 ? "  \"detections_per_frame\": " +
                       std::to_string(detections) + ",\n"
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
  write_metric(&stream, "preprocess", samples.preprocess, true);
  write_metric(&stream, "runtime", samples.runtime, true);
  write_metric(&stream, "postprocess", samples.postprocess, true);
  write_metric(&stream, "end_to_end", samples.end_to_end, false);
  stream << "  },\n"
         << "  \"throughput_fps\": "
         << (aggregate_wall_ms > 0.0
                 ? completed_frames * 1000.0 / aggregate_wall_ms
                 : 0.0)
         << "\n}\n";
  return true;
}

}  // namespace yolo
