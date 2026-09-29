// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include "sha256.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <sys/utsname.h>
#include <unistd.h>
namespace fs = std::filesystem;
using Json = nlohmann::json;
namespace himloco {
namespace {
class NewFile {
  int fd_ = -1;

public:
  explicit NewFile(const fs::path &p) {
    fd_ = open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd_ < 0)
      throw std::runtime_error("Cannot reserve new file " + p.string() + ": " +
                               std::strerror(errno));
  }
  ~NewFile() {
    if (fd_ >= 0)
      close(fd_);
  }
  NewFile(const NewFile &) = delete;
  NewFile &operator=(const NewFile &) = delete;
  void Write(const void *data, std::size_t size) {
    if (lseek(fd_, 0, SEEK_SET) < 0)
      throw std::runtime_error("File seek failed");
    auto p = static_cast<const char *>(data);
    std::size_t done = 0;
    while (done < size) {
      auto n = write(fd_, p + done, size - done);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        throw std::runtime_error("File write failed");
      done += n;
    }
    if (ftruncate(fd_, static_cast<off_t>(size)) != 0)
      throw std::runtime_error("File truncate failed");
  }
  void Report(const Json &j) {
    auto bytes = j.dump(2) + "\n";
    Write(bytes.data(), bytes.size());
  }
};
std::string Utc() {
  auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm t{};
  gmtime_r(&now, &t);
  std::ostringstream s;
  s << std::put_time(&t, "%Y-%m-%dT%H:%M:%SZ");
  return s.str();
}
Json Meta(const TensorMetadata &m) {
  return {{"name", m.name},
          {"valid_shape", m.valid_shape},
          {"aligned_shape", m.aligned_shape},
          {"aligned_byte_size", m.aligned_byte_size},
          {"tensor_type", m.tensor_type},
          {"quanti_type", m.quanti_type},
          {"tensor_layout", m.tensor_layout}};
}
double Percentile(const std::vector<double> &v, double p) {
  double position = p * (v.size() - 1);
  auto low = static_cast<std::size_t>(position);
  auto high = std::min(low + 1, v.size() - 1);
  return v[low] + (v[high] - v[low]) * (position - low);
}
} // namespace
void execute(const Options &o) {
  verify_native_model(o.model.model_path);
  auto model_digest = rdk::sha256_file(o.model.model_path);
  auto inputs = discover_inputs(o.input);
  if (fs::exists(o.output) || fs::is_symlink(o.output) ||
      fs::exists(o.report) || fs::is_symlink(o.report))
    throw std::invalid_argument("Output directory and report must be new");
  fs::create_directories(o.output.parent_path());
  if (!fs::create_directory(o.output))
    throw std::runtime_error("Cannot reserve output directory");
  fs::create_directories(o.report.parent_path());
  NewFile report_file(o.report);
  Json report = {{"schema_version", "1.0"},
                 {"status", "running"},
                 {"started_utc", Utc()},
                 {"target", "x5"},
                 {"asset_id", kAssetId},
                 {"model", o.model.model_path},
                 {"model_sha256", model_digest},
                 {"input_manifest", inputs.manifest},
                 {"output_directory", o.output.string()},
                 {"sample_count", 0},
                 {"warmup_runs", o.warmup},
                 {"warmup_completed", 0},
                 {"current_source_index", nullptr},
                 {"records", Json::array()},
                 {"scheduling", {{"priority", o.model.priority}}},
                 {"timing_scope",
                  "hbDNNInfer plus hbDNNWaitTaskDone; excludes cache "
                  "maintenance, copies, file IO and warmup"}};
  struct utsname system_info{};
  if (uname(&system_info) == 0)
    report["environment"]["machine"] = system_info.machine;
  std::ifstream os_file("/etc/version");
  std::string os_version;
  if (std::getline(os_file, os_version))
    report["environment"]["board_os_version"] = os_version;
  report_file.Report(report);
  try {
    SdkRunner runner(o.model);
    HimLoco task([&](const std::vector<float> &v) { return runner.run(v); });
    report["runtime"] = {{"model_name", runner.model_name()},
                         {"version", runner.runtime_version()},
                         {"input", Meta(runner.input_metadata())},
                         {"output", Meta(runner.output_metadata())}};
    report["current_source_index"] = inputs.records.front().index;
    auto first = load_input(inputs.records.front());
    for (int i = 0; i < o.warmup; ++i) {
      task.predict(first.first);
      report["warmup_completed"] = i + 1;
    }
    report_file.Report(report);
    std::vector<double> latencies;
    for (const auto &input : inputs.records) {
      report["current_source_index"] = input.index;
      auto loaded = load_input(input);
      auto result = task.predict(loaded.first);
      std::ostringstream name;
      name << std::setfill('0') << std::setw(6) << input.index << ".bin";
      auto path = o.output / name.str();
      NewFile action(path);
      action.Write(result.actions.data(),
                   result.actions.size() * sizeof(float));
      report["records"].push_back(
          {{"source_index", input.index},
           {"input_file", input.path.string()},
           {"input_sha256", loaded.second},
           {"output_file", path.string()},
           {"output_sha256",
            rdk::sha256_hex(result.actions.data(),
                            result.actions.size() * sizeof(float))},
           {"latency_ms", result.latency_ms}});
      report["sample_count"] = report["records"].size();
      latencies.push_back(result.latency_ms);
      report_file.Report(report);
    }
    if (rdk::sha256_file(o.model.model_path) != model_digest)
      throw std::runtime_error("Model changed during inference");
    if (!inputs.manifest.is_null() &&
        rdk::sha256_file(inputs.manifest.at("path").get<std::string>()) !=
            inputs.manifest.at("sha256"))
      throw std::runtime_error("Input manifest changed during inference");
    std::sort(latencies.begin(), latencies.end());
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) /
                  latencies.size();
    report["latency_ms"] = {{"minimum", latencies.front()},
                            {"mean", mean},
                            {"p50", Percentile(latencies, .5)},
                            {"p95", Percentile(latencies, .95)},
                            {"maximum", latencies.back()}};
    if (mean > 0)
      report["sequential_throughput_fps"] = 1000.0 / mean;
    report["status"] = "completed";
    report["current_source_index"] = nullptr;
    report["finished_utc"] = Utc();
    report_file.Report(report);
  } catch (const std::exception &e) {
    report["status"] = "failed";
    report["error"] = e.what();
    report["finished_utc"] = Utc();
    report.erase("latency_ms");
    report.erase("sequential_throughput_fps");
    report_file.Report(report);
    throw;
  }
}
} // namespace himloco
