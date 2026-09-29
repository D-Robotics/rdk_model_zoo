// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include "sha256.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
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
} // namespace himloco
