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

// Pure (host-testable) benchmark bookkeeping shared by the C++ task samples.
// The JSON emitted by write_benchmark_json() deliberately mirrors the
// end_to_end schema validated by model_zoo_web/scripts/build_catalog.py
// (pipeline_streams, runtime_submission_threads, timed_frames,
// aggregate_wall_ms, throughput_fps, per-stage mean/p50/p95/min/max).

#ifndef RUNTIME_CPP_COMMON_BENCHMARK_H_
#define RUNTIME_CPP_COMMON_BENCHMARK_H_

#include <cstddef>
#include <string>
#include <vector>

namespace yolo {

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

struct Statistics {
  double mean = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

double percentile(const std::vector<double>& values, double fraction);
Statistics summarize(const std::vector<double>& values);

// Identity and methodology fields embedded in the benchmark JSON.
struct BenchmarkMeta {
  std::string model_path;
  std::string image_path;
  std::string implementation;
  std::string timing_scope = "in_memory_bgr_to_detections";
  std::string cpu_thread_policy = "all_online";
  int pipeline_streams = 1;
  int online_cpu_threads = 0;
  int opencv_threads = 0;
  int warmup_frames_per_round = 20;
  int runs_per_round = 200;
  int rounds = 3;
  float score_threshold = 0.25f;
  float nms_threshold = 0.7f;
};

// Writes the aggregate report. Returns false when `path` cannot be opened.
bool write_benchmark_json(const std::string& path, const BenchmarkMeta& meta,
                          const StageSamples& samples, size_t detections,
                          size_t completed_frames, double aggregate_wall_ms);

}  // namespace yolo

#endif  // RUNTIME_CPP_COMMON_BENCHMARK_H_
