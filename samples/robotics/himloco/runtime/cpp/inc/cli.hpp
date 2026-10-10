// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "policy.hpp"
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace himloco {
inline constexpr const char *kAssetId =
    "x5:himloco:himloco_go2_bayese_1x270.bin";
struct Options {
  NativeConfig model;
  std::filesystem::path input =
      "samples/robotics/himloco/test_data/obs_history";
  std::filesystem::path output = "outputs/himloco_cpp", report;
  int warmup = 10;
  bool help = false;
};
struct InputRecord {
  std::int64_t index;
  std::filesystem::path path;
  std::string digest;
};
struct Inputs {
  std::vector<InputRecord> records;
  nlohmann::json manifest = nullptr;
};
Options parse_cli(int argc, char **argv);
std::string cli_help();
Inputs discover_inputs(const std::filesystem::path &path);
std::pair<std::vector<float>, std::string>
load_input(const InputRecord &record);

/// Exclusive run workspace: reserves the output directory and report file,
/// keeps the incremental JSON report, and owns per-run output presentation.
/// Inference itself never writes; main drives it record by record.
class RunWorkspace {
 public:
  RunWorkspace(const Options &options, Inputs inputs,
               std::string model_digest);
  ~RunWorkspace();
  RunWorkspace(const RunWorkspace &) = delete;
  RunWorkspace &operator=(const RunWorkspace &) = delete;
  const Options &options() const;
  const Inputs &inputs() const;
  void note_runtime(const std::string &model_name,
                    const std::string &runtime_version,
                    const TensorMetadata &input, const TensorMetadata &output);
  void begin_warmup();
  void warmup_completed(int completed);
  void flush();
  void begin_record(const InputRecord &record);
  void add_record(const InputRecord &record, const std::string &input_digest,
                  const InferenceResult &result);
  void complete();
  void fail(const std::exception &error);

 private:
  class ReservedReport;
  Options options_;
  Inputs inputs_;
  std::string model_digest_;
  nlohmann::json report_;
  std::vector<double> latencies_;
  std::unique_ptr<ReservedReport> report_file_;
};
}  // namespace himloco
