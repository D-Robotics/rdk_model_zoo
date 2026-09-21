/*
 * Copyright (c) 2026, D-Robotics.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host unit tests for the benchmark bookkeeping and the catalog-aligned
// JSON writer (common/benchmark.{h,cc}).

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/benchmark.h"

namespace {

int failures = 0;

void expect_near(const char* what, double actual, double expected, double tol) {
  if (actual < expected - tol || actual > expected + tol) {
    std::printf("FAIL %s: actual=%f expected=%f\n", what, actual, expected);
    ++failures;
  }
}

void expect_contains(const char* what, const std::string& haystack,
                     const std::string& needle) {
  if (haystack.find(needle) == std::string::npos) {
    std::printf("FAIL %s: missing '%s'\n", what, needle.c_str());
    ++failures;
  }
}

}  // namespace

int main() {
  const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
  const yolo::Statistics stats = yolo::summarize(values);
  expect_near("mean", stats.mean, 3.0, 1e-12);
  expect_near("p50", stats.p50, 3.0, 1e-12);
  expect_near("p95", stats.p95, 4.8, 1e-12);
  expect_near("min", stats.min, 1.0, 1e-12);
  expect_near("max", stats.max, 5.0, 1e-12);
  expect_near("percentile empty", yolo::percentile(std::vector<double>(), 0.5),
              0.0, 0.0);

  yolo::StageSamples samples;
  yolo::StageTiming timing;
  timing.preprocess_ms = 4.0;
  timing.runtime_ms = 12.0;
  timing.postprocess_ms = 4.0;
  timing.end_to_end_ms = 20.0;
  samples.add(timing);
  samples.add(timing);

  yolo::BenchmarkMeta meta;
  meta.model_path = "yolo26n_detect_bayese_640x640_nv12.bin";
  meta.image_path = "bus.jpg";
  meta.implementation = "native_cpp_yolo26_ltrb";
  meta.pipeline_streams = 2;
  meta.online_cpu_threads = 8;
  meta.opencv_threads = 8;
  meta.warmup_frames_per_round = 20;
  meta.runs_per_round = 200;
  meta.rounds = 3;

  const std::string path = "/tmp/yolo_test_benchmark.json";
  if (!yolo::write_benchmark_json(path, meta, samples, 5, 10, 100.0)) {
    std::printf("FAIL write_benchmark_json returned false\n");
    return 1;
  }

  std::ifstream stream(path.c_str());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  const std::string json = buffer.str();

  // Catalog schema fields validated by build_catalog.py must all be present.
  expect_contains("json", json, "\"schema_version\": 1");
  expect_contains("json", json, "\"pipeline_streams\": 2");
  expect_contains("json", json, "\"runtime_submission_threads\": 2");
  expect_contains("json", json, "\"cpu_thread_policy\": \"all_online\"");
  expect_contains("json", json, "\"online_cpu_threads\": 8");
  expect_contains("json", json, "\"opencv_threads\": 8");
  expect_contains("json", json, "\"warmup_frames_per_round\": 20");
  expect_contains("json", json, "\"timed_frames\": 10");
  expect_contains("json", json, "\"aggregate_wall_ms\": 100.000000");
  expect_contains("json", json, "\"throughput_fps\": 100.000000");
  expect_contains("json", json, "\"end_to_end\": {\"mean\": 20.000000");
  expect_contains("json", json, "\"runtime\": {\"mean\": 12.000000");
  expect_contains("json", json, "\"implementation\": \"native_cpp_yolo26_ltrb\"");

  if (failures == 0) {
    std::printf("test_benchmark: OK\n");
    return 0;
  }
  std::printf("test_benchmark: %d failure(s)\n", failures);
  return 1;
}
