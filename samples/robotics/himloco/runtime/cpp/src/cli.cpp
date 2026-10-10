// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include "sha256.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/utsname.h>
#include <unistd.h>
namespace fs = std::filesystem;
namespace himloco {
namespace {
std::int64_t Index(const std::string &s) {
  if (s.empty() || !std::all_of(s.begin(), s.end(),
                                [](char c) { return c >= '0' && c <= '9'; }))
    throw std::invalid_argument("Expected decimal source index");
  std::size_t end = 0;
  auto n = std::stoll(s, &end);
  if (end != s.size() || n < 0)
    throw std::invalid_argument("Invalid index");
  return n;
}
std::string Bytes(const fs::path &p) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot read " + p.string());
  std::string data((std::istreambuf_iterator<char>(f)), {});
  if (f.bad())
    throw std::runtime_error("Read failed " + p.string());
  return data;
}
bool Digest(const std::string &s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
} // namespace
Options parse_cli(int argc, char **argv) {
  Options o;
  std::set<std::string> seen;
  std::string target = "x5", asset = kAssetId;
  bool external = false;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i], value;
    auto equals = key.find('=');
    if (equals != std::string::npos) {
      value = key.substr(equals + 1);
      key.resize(equals);
    }
    if (key == "--model_path")
      key = "--model-path";
    if (key == "--input_path")
      key = "--input-path";
    if (key == "--output_dir")
      key = "--output-dir";
    if (!seen.insert(key).second)
      throw std::invalid_argument("Duplicate option " + key);
    if (key == "--help") {
      if (equals != std::string::npos)
        throw std::invalid_argument("--help takes no value");
      o.help = true;
      continue;
    }
    if (equals == std::string::npos) {
      if (i + 1 >= argc)
        throw std::invalid_argument("Missing value for " + key);
      value = argv[++i];
    }
    if (value.empty())
      throw std::invalid_argument("Empty value for " + key);
    if (key == "--model-path") {
      o.model.model_path = value;
      external = true;
    } else if (key == "--input-path")
      o.input = value;
    else if (key == "--output-dir")
      o.output = value;
    else if (key == "--report")
      o.report = value;
    else if (key == "--target")
      target = value;
    else if (key == "--asset-id")
      asset = value;
    else if (key == "--warmup") {
      auto n = Index(value);
      if (n > 1000000)
        throw std::invalid_argument("warmup too large");
      o.warmup = static_cast<int>(n);
    } else if (key == "--priority") {
      auto n = value == "-1" ? -1 : Index(value);
      if (n > 255)
        throw std::invalid_argument("priority must be -1 or [0,255]");
      o.model.priority = static_cast<int>(n);
    } else
      throw std::invalid_argument("Unknown option " + key);
  }
  if (o.help)
    return o;
  if (target != "x5")
    throw std::invalid_argument("HIMLoco native target must be x5");
  if (asset != kAssetId || (external && !seen.count("--asset-id")))
    throw std::invalid_argument(
        "External model requires exact published --asset-id");
  o.input = fs::absolute(o.input).lexically_normal();
  o.output = fs::absolute(o.output).lexically_normal();
  o.model.model_path =
      fs::absolute(o.model.model_path).lexically_normal().string();
  o.report = o.report.empty() ? o.output / "report.json"
                              : fs::absolute(o.report).lexically_normal();
  if (o.report == o.output)
    throw std::invalid_argument("Report must differ from output directory");
  return o;
}
std::string cli_help() {
  return "HIMLoco native inference; paths relative to caller cwd, defaults "
         "from repository root\n"
         "--target x5 --asset-id x5:himloco:himloco_go2_bayese_1x270.bin\n"
         "--model-path FILE (default "
         "samples/robotics/himloco/model/bayes-e/"
         "himloco_go2_bayese_1x270.bin)\n"
         "--input-path FILE|DIR (default "
         "samples/robotics/himloco/test_data/obs_history)\n"
         "--output-dir DIR (default outputs/himloco_cpp; must be new)\n"
         "--report FILE (default output directory/report.json; must be new)\n"
         "--warmup 10 (0..1000000) --priority -1 (-1 or 0..255) --help\n"
         "Legacy aliases: --model_path --input_path --output_dir\n";
}
Inputs discover_inputs(const fs::path &path) {
  std::map<std::int64_t, fs::path> sorted;
  auto add = [&](const fs::path &p) {
    if (!fs::is_regular_file(p) || p.extension() != ".bin")
      throw std::invalid_argument("Expected observation BIN");
    auto index = Index(p.stem().string());
    if (!sorted.emplace(index, fs::canonical(p)).second)
      throw std::invalid_argument("Duplicate source index");
  };
  if (fs::is_directory(path)) {
    for (const auto &e : fs::directory_iterator(path))
      if (e.path().extension() == ".bin")
        add(e.path());
  } else
    add(path);
  if (sorted.empty())
    throw std::invalid_argument("No observation BIN files");
  auto dir = fs::is_directory(path) ? path : path.parent_path();
  auto manifest_path = dir.parent_path() / "runtime-input-manifest.json";
  Inputs result;
  std::map<fs::path, std::pair<std::int64_t, std::string>> expected;
  if (fs::exists(manifest_path)) {
    auto bytes = Bytes(manifest_path);
    auto m = nlohmann::json::parse(bytes);
    auto contract = m.at("input_contract");
    if (contract.at("name") != "obs_history" ||
        contract.at("shape") != nlohmann::json::array({1, 270}) ||
        contract.at("dtype") != "float32" ||
        contract.at("bytes_per_file") != 1080)
      throw std::invalid_argument("Manifest contract mismatch");
    auto records = m.at("records");
    if (!records.is_array() || records.empty())
      throw std::invalid_argument("Manifest requires records");
    auto base = fs::canonical(manifest_path.parent_path());
    std::set<std::int64_t> indices;
    for (const auto &e : records) {
      if (!e.at("source_index").is_number_integer())
        throw std::invalid_argument("Invalid manifest index");
      auto index = e.at("source_index").get<std::int64_t>();
      auto file = e.at("file").get<std::string>();
      auto digest = e.at("sha256").get<std::string>();
      if (index < 0 || file.empty() || fs::path(file).is_absolute() ||
          !Digest(digest) || e.at("bytes") != 1080 ||
          !indices.insert(index).second)
        throw std::invalid_argument("Invalid manifest record");
      auto full = fs::weakly_canonical(base / file);
      auto relative = full.lexically_relative(base);
      if (relative.empty() || *relative.begin() == ".." ||
          !expected.emplace(full, std::make_pair(index, digest)).second)
        throw std::invalid_argument("Invalid manifest path");
    }
    result.manifest = {{"path", manifest_path.string()},
                       {"sha256", rdk::sha256_hex(bytes.data(), bytes.size())}};
  }
  for (const auto &[index, file] : sorted) {
    std::string digest;
    if (!result.manifest.is_null()) {
      auto entry = expected.find(file);
      if (entry == expected.end() || entry->second.first != index)
        throw std::invalid_argument("Input absent from manifest");
      digest = entry->second.second;
    }
    result.records.push_back({index, file, digest});
  }
  return result;
}
std::pair<std::vector<float>, std::string> load_input(const InputRecord &r) {
  auto bytes = Bytes(r.path);
  if (bytes.size() != 1080)
    throw std::invalid_argument("Observation must contain exactly 1080 bytes");
  auto digest = rdk::sha256_hex(bytes.data(), bytes.size());
  if (!r.digest.empty() && digest != r.digest)
    throw std::invalid_argument("Observation digest mismatch");
  const std::uint32_t marker = 1;
  if (*reinterpret_cast<const unsigned char *>(&marker) != 1)
    throw std::runtime_error("Native raw float IO requires little-endian host");
  std::vector<float> values(270);
  std::memcpy(values.data(), bytes.data(), bytes.size());
  if (!std::all_of(values.begin(), values.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::invalid_argument("Observation contains NaN/Inf");
  return {values, digest};
}

// --- Run workspace: exclusive output reservation and incremental report. ---
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
  void Report(const nlohmann::json &j) {
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
nlohmann::json Meta(const TensorMetadata &m) {
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
class RunWorkspace::ReservedReport : public NewFile {
  using NewFile::NewFile;
};
RunWorkspace::RunWorkspace(const Options &options, Inputs inputs,
                           std::string model_digest)
    : options_(options), inputs_(std::move(inputs)),
      model_digest_(std::move(model_digest)) {
  const auto &o = options_;
  if (fs::exists(o.output) || fs::is_symlink(o.output) ||
      fs::exists(o.report) || fs::is_symlink(o.report))
    throw std::invalid_argument("Output directory and report must be new");
  fs::create_directories(o.output.parent_path());
  if (!fs::create_directory(o.output))
    throw std::runtime_error("Cannot reserve output directory");
  fs::create_directories(o.report.parent_path());
  report_file_ = std::make_unique<ReservedReport>(o.report);
  report_ = {{"schema_version", "1.0"},
             {"status", "running"},
             {"started_utc", Utc()},
             {"target", "x5"},
             {"asset_id", kAssetId},
             {"model", o.model.model_path},
             {"model_sha256", model_digest_},
             {"input_manifest", inputs_.manifest},
             {"output_directory", o.output.string()},
             {"sample_count", 0},
             {"warmup_runs", o.warmup},
             {"warmup_completed", 0},
             {"current_source_index", nullptr},
             {"records", nlohmann::json::array()},
             {"scheduling", {{"priority", o.model.priority}}},
             {"timing_scope",
              "hbDNNInfer plus hbDNNWaitTaskDone; excludes cache "
              "maintenance, copies, file IO and warmup"}};
  struct utsname system_info{};
  if (uname(&system_info) == 0)
    report_["environment"]["machine"] = system_info.machine;
  std::ifstream os_file("/etc/version");
  std::string os_version;
  if (std::getline(os_file, os_version))
    report_["environment"]["board_os_version"] = os_version;
  report_file_->Report(report_);
}
RunWorkspace::~RunWorkspace() = default;
const Options &RunWorkspace::options() const { return options_; }
const Inputs &RunWorkspace::inputs() const { return inputs_; }
void RunWorkspace::note_runtime(const std::string &model_name,
                                const std::string &runtime_version,
                                const TensorMetadata &input,
                                const TensorMetadata &output) {
  report_["runtime"] = {{"model_name", model_name},
                        {"version", runtime_version},
                        {"input", Meta(input)},
                        {"output", Meta(output)}};
}
void RunWorkspace::begin_warmup() {
  report_["current_source_index"] = inputs_.records.front().index;
}
void RunWorkspace::warmup_completed(int completed) {
  report_["warmup_completed"] = completed;
}
void RunWorkspace::flush() { report_file_->Report(report_); }
void RunWorkspace::begin_record(const InputRecord &record) {
  report_["current_source_index"] = record.index;
}
void RunWorkspace::add_record(const InputRecord &record,
                              const std::string &input_digest,
                              const InferenceResult &result) {
  std::ostringstream name;
  name << std::setfill('0') << std::setw(6) << record.index << ".bin";
  auto path = options_.output / name.str();
  NewFile action(path);
  action.Write(result.actions.data(), result.actions.size() * sizeof(float));
  report_["records"].push_back(
      {{"source_index", record.index},
       {"input_file", record.path.string()},
       {"input_sha256", input_digest},
       {"output_file", path.string()},
       {"output_sha256",
        rdk::sha256_hex(result.actions.data(),
                        result.actions.size() * sizeof(float))},
       {"latency_ms", result.latency_ms}});
  report_["sample_count"] = report_["records"].size();
  latencies_.push_back(result.latency_ms);
  report_file_->Report(report_);
}
void RunWorkspace::complete() {
  if (rdk::sha256_file(options_.model.model_path) != model_digest_)
    throw std::runtime_error("Model changed during inference");
  if (!inputs_.manifest.is_null() &&
      rdk::sha256_file(inputs_.manifest.at("path").get<std::string>()) !=
          inputs_.manifest.at("sha256"))
    throw std::runtime_error("Input manifest changed during inference");
  std::sort(latencies_.begin(), latencies_.end());
  double mean = std::accumulate(latencies_.begin(), latencies_.end(), 0.0) /
                latencies_.size();
  report_["latency_ms"] = {{"minimum", latencies_.front()},
                           {"mean", mean},
                           {"p50", Percentile(latencies_, .5)},
                           {"p95", Percentile(latencies_, .95)},
                           {"maximum", latencies_.back()}};
  if (mean > 0)
    report_["sequential_throughput_fps"] = 1000.0 / mean;
  report_["status"] = "completed";
  report_["current_source_index"] = nullptr;
  report_["finished_utc"] = Utc();
  report_file_->Report(report_);
}
void RunWorkspace::fail(const std::exception &error) {
  report_["status"] = "failed";
  report_["error"] = error.what();
  report_["finished_utc"] = Utc();
  report_.erase("latency_ms");
  report_.erase("sequential_throughput_fps");
  report_file_->Report(report_);
}
} // namespace himloco
