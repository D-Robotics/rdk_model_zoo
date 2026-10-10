/*
 * Copyright (c) 2026, D-Robotics.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host unit tests for the benchmark bookkeeping: the statistics helpers,
// the catalog-aligned JSON writer and the synchronized round runner (all
// inline in inc/cli.hpp, no library links). The CLI option surface is
// asserted by test_benchmark_streams.cc, which links src/cli.cpp.

#include <atomic>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"

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
  meta.runtime_source_sha256 = std::string(64, 'a');
  meta.executable_sha256 = std::string(64, 'b');
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
  expect_contains("json", json, "\"output_kind\": \"detections\"");
  expect_contains("json", json, "\"outputs_per_frame\": 5");
  expect_contains("json", json, "\"detections_per_frame\": 5");
  expect_contains("json", json, "\"runtime_submission_threads\": 2");
  expect_contains("json", json, "\"cpu_thread_policy\": \"all_online\"");
  expect_contains("json", json, "\"online_cpu_threads\": 8");
  expect_contains("json", json, "\"opencv_threads\": 8");
  expect_contains("json", json,
                  "\"runtime_source_sha256\": \"" + std::string(64, 'a') + "\"");
  expect_contains("json", json,
                  "\"executable_sha256\": \"" + std::string(64, 'b') + "\"");
  expect_contains("json", json, "\"warmup_frames_per_round\": 20");
  expect_contains("json", json, "\"timed_frames\": 10");
  expect_contains("json", json, "\"aggregate_wall_ms\": 100.000000");
  expect_contains("json", json, "\"throughput_fps\": 100.000000");
  expect_contains("json", json, "\"end_to_end\": {\"mean\": 20.000000");
  expect_contains("json", json, "\"runtime\": {\"mean\": 12.000000");
  expect_contains("json", json, "\"implementation\": \"native_cpp_yolo26_ltrb\"");

  // Oriented-box metadata: the angle options are recorded and Top-K tasks
  // omit score/NMS. One probe per conditional field.
  yolo::BenchmarkMeta obb_meta = meta;
  obb_meta.output_kind = "rotated_boxes";
  const std::string obb_path = "/tmp/yolo_test_benchmark_obb.json";
  if (!yolo::write_benchmark_json(obb_path, obb_meta, samples, 2, 4, 40.0)) {
    std::printf("FAIL write_benchmark_json (obb) returned false\n");
    return 1;
  }
  std::ifstream obb_stream(obb_path.c_str());
  std::stringstream obb_buffer;
  obb_buffer << obb_stream.rdbuf();
  const std::string obb_json = obb_buffer.str();
  expect_contains("obb json", obb_json, "\"output_kind\": \"rotated_boxes\"");
  expect_contains("obb json", obb_json, "\"angle_sign\": 1.000000");
  expect_contains("obb json", obb_json, "\"angle_offset_degrees\": 0.000000");
  expect_contains("obb json", obb_json, "\"regularize_obb\": true");
  if (obb_json.find("\"detections_per_frame\"") != std::string::npos) {
    std::printf("FAIL obb json: unexpected detections_per_frame\n");
    ++failures;
  }

  yolo::BenchmarkMeta cls_meta = meta;
  cls_meta.output_kind = "topk_predictions";
  const std::string cls_path = "/tmp/yolo_test_benchmark_cls.json";
  if (!yolo::write_benchmark_json(cls_path, cls_meta, samples, 5, 10, 100.0)) {
    std::printf("FAIL write_benchmark_json (classify) returned false\n");
    return 1;
  }
  std::ifstream cls_stream(cls_path.c_str());
  std::stringstream cls_buffer;
  cls_buffer << cls_stream.rdbuf();
  const std::string cls_json = cls_buffer.str();
  if (cls_json.find("\"score_threshold\"") != std::string::npos ||
      cls_json.find("\"nms_threshold\"") != std::string::npos ||
      cls_json.find("\"detections_per_frame\"") != std::string::npos) {
    std::printf("FAIL classify json: score/NMS must be omitted\n");
    ++failures;
  }

  std::atomic<int> calls0(0);
  std::atomic<int> calls1(0);
  std::vector<yolo::BenchmarkPipeline> pipelines;
  pipelines.push_back([&calls0](yolo::StageTiming* stage, size_t* count,
                                std::string*) {
    ++calls0;
    stage->preprocess_ms = 1.0;
    stage->runtime_ms = 2.0;
    stage->postprocess_ms = 3.0;
    stage->end_to_end_ms = 6.0;
    *count = 1;
    return true;
  });
  pipelines.push_back([&calls1](yolo::StageTiming* stage, size_t* count,
                                std::string*) {
    ++calls1;
    stage->runtime_ms = 2.0;
    stage->end_to_end_ms = 6.0;
    *count = 1;
    return true;
  });
  yolo::BenchmarkRound round;
  std::string round_error;
  if (!yolo::run_benchmark_round(pipelines, 3, 1, true, &round,
                                 &round_error) ||
      round.completed_frames != 6 || calls0 != 3 || calls1 != 3 ||
      round.samples.end_to_end.size() != 6) {
    std::printf("FAIL run benchmark round: %s\n", round_error.c_str());
    ++failures;
  }
  std::remove(path.c_str());
  std::remove(obb_path.c_str());
  std::remove(cls_path.c_str());

  if (failures == 0) {
    std::printf("test_benchmark: OK\n");
    return 0;
  }
  std::printf("test_benchmark: %d failure(s)\n", failures);
  return 1;
}
