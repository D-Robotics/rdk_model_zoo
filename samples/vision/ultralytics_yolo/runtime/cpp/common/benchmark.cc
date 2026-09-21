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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace yolo {

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
         << "  \"timing_scope\": \"" << json_escape(meta.timing_scope)
         << "\",\n"
         << "  \"pipeline_streams\": " << meta.pipeline_streams << ",\n"
         << "  \"runtime_submission_threads\": " << meta.pipeline_streams
         << ",\n"
         << "  \"cpu_thread_policy\": \"" << json_escape(meta.cpu_thread_policy)
         << "\",\n"
         << "  \"online_cpu_threads\": " << meta.online_cpu_threads << ",\n"
         << "  \"opencv_threads\": " << meta.opencv_threads << ",\n"
         << "  \"warmup_frames_per_round\": " << meta.warmup_frames_per_round
         << ",\n"
         << "  \"runs_per_round\": " << meta.runs_per_round << ",\n"
         << "  \"frames_per_stream_per_round\": " << meta.runs_per_round
         << ",\n"
         << "  \"rounds\": " << meta.rounds << ",\n"
         << "  \"timed_frames\": " << completed_frames << ",\n"
         << "  \"aggregate_wall_ms\": " << aggregate_wall_ms << ",\n"
         << "  \"detections_per_frame\": " << detections << ",\n"
         << "  \"score_threshold\": " << meta.score_threshold << ",\n"
         << "  \"nms_threshold\": " << meta.nms_threshold << ",\n"
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
