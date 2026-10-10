// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include "sha256.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace paraformer {
// --- Argument parsing and vocabulary -------------------------------------------
CliOptions parse_cli(int argc, const char *const *argv) {
  CliOptions options;
  options.models[0].model.stage = Stage::Encoder;
  options.models[1].model.stage = Stage::Predictor;
  options.models[2].model.stage = Stage::Decoder;
  std::string target, limit = "0";
  std::map<std::string, std::string *> fields{
      {"--target", &target},
      {"--manifest", &options.manifest},
      {"--vocab-file", &options.vocabulary},
      {"--output-dir", &options.output},
      {"--max-utts", &limit}};
  const std::array<std::string, 3> names{"encoder", "predictor", "decoder"};
  for (size_t i = 0; i < 3; ++i) {
    fields["--" + names[i] + "-model-path"] = &options.models[i].model.path;
    fields["--" + names[i] + "-asset-id"] = &options.models[i].asset_id;
    fields["--" + names[i] + "-sha256"] = &options.models[i].expected_sha256;
  }
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (!seen.insert(key).second)
      throw std::invalid_argument("Duplicate option: " + key);
    if (key == "--help") {
      options.help = true;
      continue;
    }
    const auto found = fields.find(key);
    if (found == fields.end())
      throw std::invalid_argument("Unknown option: " + key);
    if (i + 1 >= argc || !argv[i + 1][0] ||
        std::string(argv[i + 1]).rfind("--", 0) == 0)
      throw std::invalid_argument("Missing value: " + key);
    *found->second = argv[++i];
  }
  if (options.help)
    return options;
  if (target != "s100")
    throw std::invalid_argument("Native Paraformer requires --target s100");
  if (options.manifest.empty() || options.vocabulary.empty() ||
      options.output.empty())
    throw std::invalid_argument(
        "--manifest, --vocab-file and --output-dir are required");
  if (limit.empty() ||
      !std::all_of(limit.begin(), limit.end(),
                   [](unsigned char c) { return c >= '0' && c <= '9'; }))
    throw std::invalid_argument("--max-utts must be a nonnegative integer");
  options.max_utts = std::stoull(limit);
  for (auto &artifact : options.models) {
    artifact.model.target = target;
    if (artifact.model.path.empty() ||
        artifact.asset_id != expected_asset_id(artifact.model.stage) ||
        artifact.expected_sha256.size() != 64 ||
        !std::all_of(artifact.expected_sha256.begin(),
                     artifact.expected_sha256.end(), [](unsigned char c) {
                       return (c >= '0' && c <= '9') ||
                              (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
                     }))
      throw std::invalid_argument(
          "Each stage requires exact asset ID, model path and SHA-256");
    std::transform(artifact.expected_sha256.begin(),
                   artifact.expected_sha256.end(),
                   artifact.expected_sha256.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }
  return options;
}
std::string cli_help() {
  return "Paraformer native prepared-feature inference\n"
         "Required: --target s100 --manifest FILE --vocab-file FILE "
         "--output-dir NEW_DIR\n"
         "Each encoder/predictor/decoder requires: --<stage>-model-path FILE\n"
         "  --<stage>-asset-id "
         "s:paraformer:s100/paraformer_large_<stage>_<geometry>_s100.hbm\n"
         "  --<stage>-sha256 HEX64 (encoder geometry 400x560; others 400x512)\n"
         "Optional: --max-utts N (0 = all), --help\n"
         "Input: Python --preprocess-only prepared-manifest.json; finite "
         "float32 [1,400,560].\n";
}
std::vector<std::string> load_vocabulary(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("Cannot open vocabulary");
  const std::string bytes(std::istreambuf_iterator<char>(file), {});
  if (file.bad() ||
      rdk::sha256_hex(bytes.data(), bytes.size()) != kVocabularySha256)
    throw std::invalid_argument("Vocabulary SHA-256 mismatch");
  const auto value = nlohmann::json::parse(bytes);
  if (!value.is_array())
    throw std::invalid_argument("Expected ordered vocabulary array");
  auto tokens = value.get<std::vector<std::string>>();
  validate_vocabulary(tokens);
  return tokens;
}

// --- Prepared-feature manifest and NPY features ---------------------------------
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}
std::string sha(std::string value) {
  require(value.size() == 64 && std::all_of(value.begin(), value.end(),
                                            [](unsigned char c) {
                                              return (c >= '0' && c <= '9') ||
                                                     (c >= 'a' && c <= 'f') ||
                                                     (c >= 'A' && c <= 'F');
                                            }),
          "Expected feature SHA-256");
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
int positive_integer(const nlohmann::json &value) {
  require(value.is_number_integer(), "Frame counts must be integers");
  const auto number = value.get<int64_t>();
  require(number > 0 && number <= std::numeric_limits<int>::max(),
          "Frame count out of range");
  return int(number);
}
bool whitespace(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' ||
         c == '\f';
}
// Small data-only parser for the NPY scalar float array header grammar. It
// never evaluates Python and accepts key order/quote style independently.
class Header {
  const std::string &s;
  size_t p = 0;
  void spaces() {
    while (p < s.size() && whitespace(s[p]))
      ++p;
  }
  char peek() {
    spaces();
    return p < s.size() ? s[p] : '\0';
  }
  void take(char c) {
    require(peek() == c, "Malformed NPY header");
    ++p;
  }
  std::string string() {
    const char q = peek();
    require(q == '\'' || q == '"', "Expected NPY string");
    ++p;
    const size_t start = p;
    while (p < s.size() && s[p] != q) {
      require(s[p] != '\\' && s[p] != '\0' && s[p] != '\n',
              "Unsupported NPY string escape");
      ++p;
    }
    require(p < s.size(), "Unclosed NPY string");
    auto out = s.substr(start, p - start);
    ++p;
    return out;
  }
  int integer() {
    spaces();
    const size_t start = p;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9')
      ++p;
    require(p > start && p - start <= 9, "Invalid NPY dimension");
    return std::stoi(s.substr(start, p - start));
  }

public:
  explicit Header(const std::string &text) : s(text) {}
  std::string parse() {
    std::set<std::string> keys;
    std::string dtype;
    std::vector<int> shape;
    take('{');
    while (peek() != '}') {
      const auto key = string();
      require(keys.insert(key).second, "Duplicate NPY key");
      take(':');
      if (key == "descr")
        dtype = string();
      else if (key == "fortran_order") {
        spaces();
        require(s.compare(p, 5, "False") == 0,
                "Only C-order features supported");
        p += 5;
      } else if (key == "shape") {
        take('(');
        while (peek() != ')') {
          shape.push_back(integer());
          require(shape.size() <= 3, "Expected three NPY dimensions");
          if (peek() == ')')
            break;
          take(',');
        }
        take(')');
      } else
        throw std::invalid_argument("Unknown NPY header key");
      if (peek() == '}')
        break;
      take(',');
    }
    take('}');
    spaces();
    require(p == s.size(), "Trailing NPY header syntax");
    require(keys == std::set<std::string>{"descr", "fortran_order", "shape"},
            "Incomplete NPY header");
    require(shape == std::vector<int>{1, 400, 560},
            "Expected feature shape [1,400,560]");
    require(dtype == "<f4" || dtype == ">f4",
            "Expected explicit-endian float32 features");
    return dtype;
  }
};
} // namespace
std::vector<FeatureItem> load_prepared_manifest(const std::string &path,
                                                size_t max_utts) {
  std::ifstream file(path, std::ios::binary);
  require(bool(file), "Cannot open prepared manifest");
  auto entries = nlohmann::json::parse(file);
  require(entries.is_array() && !entries.empty(),
          "Prepared manifest must be a nonempty JSON list");
  std::vector<FeatureItem> result;
  std::set<std::string> ids;
  for (const auto &entry : entries) {
    require(entry.is_object(), "Prepared entry must be an object");
    FeatureItem item;
    item.utt_id = entry.at("utt_id").get<std::string>();
    require(!item.utt_id.empty() && item.utt_id != "." && item.utt_id != ".." &&
                !whitespace(item.utt_id.front()) &&
                !whitespace(item.utt_id.back()) &&
                item.utt_id.find_first_of("/\\") == std::string::npos &&
                item.utt_id.find('\0') == std::string::npos,
            "Invalid utterance ID");
    require(ids.insert(item.utt_id).second, "Duplicate utterance ID");
    item.valid_frames = positive_integer(entry.at("feat_length"));
    item.original_frames = positive_integer(entry.at("original_frames"));
    require(entry.at("truncated").is_boolean(), "truncated must be boolean");
    item.truncated = entry.at("truncated").get<bool>();
    require(item.valid_frames == std::min(item.original_frames, 400) &&
                item.truncated == (item.original_frames > 400),
            "Inconsistent original/valid frames or truncation");
    item.sha256 = sha(entry.at("feature_sha256").get<std::string>());
    auto feature = entry.at("feature_file").get<std::string>();
    require(!feature.empty() && feature.find('\0') == std::string::npos,
            "Invalid feature path");
    item.path = (std::filesystem::absolute(path).parent_path() / feature)
                    .lexically_normal()
                    .string();
    if (entry.contains("text"))
      item.reference_text = entry.at("text").get<std::string>();
    item.original_record_json = entry.dump();
    result.push_back(std::move(item));
  }
  if (max_utts && max_utts < result.size())
    result.resize(max_utts);
  return result;
}
std::vector<float> load_features(const FeatureItem &item) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "Requires IEEE float32");
  require(item.valid_frames >= 1 && item.valid_frames <= 400 &&
              item.original_frames >= item.valid_frames &&
              item.valid_frames == std::min(item.original_frames, 400) &&
              item.truncated == (item.original_frames > 400),
          "Invalid feature frame metadata");
  const auto expected = sha(item.sha256);
  require(std::filesystem::is_regular_file(item.path),
          "Feature file must be regular");
  const auto size = std::filesystem::file_size(item.path);
  constexpr size_t payload = 400 * 560 * 4;
  require(size >= payload + 10 && size <= payload + 65536 + 12,
          "Invalid feature NPY size");
  std::vector<unsigned char> bytes(size);
  std::ifstream file(item.path, std::ios::binary);
  file.read(reinterpret_cast<char *>(bytes.data()),
            std::streamsize(bytes.size()));
  require(file.gcount() == std::streamsize(bytes.size()) &&
              file.peek() == std::ifstream::traits_type::eof() && !file.bad(),
          "Incomplete or changed feature file");
  require(rdk::sha256_hex(bytes.data(), bytes.size()) == expected,
          "Feature SHA-256 mismatch");
  require(std::memcmp(bytes.data(), "\x93NUMPY", 6) == 0, "Invalid NPY magic");
  const int major = bytes[6], minor = bytes[7];
  require((major == 1 || major == 2 || major == 3) && minor == 0,
          "Unsupported NPY version");
  const size_t width = major == 1 ? 2 : 4, begin = 8 + width;
  uint32_t length = 0;
  for (size_t i = 0; i < width; ++i)
    length |= uint32_t(bytes[8 + i]) << (8 * i);
  require(length > 0 && length <= 65536 &&
              begin + length + payload == bytes.size(),
          "Invalid NPY header/payload length");
  const std::string header(reinterpret_cast<const char *>(bytes.data() + begin),
                           length);
  require(header.back() == '\n', "NPY header must end in newline");
  const auto dtype = Header(header).parse();
  const bool little = dtype == "<f4";
  std::vector<float> values(400 * 560);
  const auto *data = bytes.data() + begin + length;
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t bits = 0;
    for (size_t j = 0; j < 4; ++j)
      bits |= uint32_t(data[4 * i + j]) << (8 * (little ? j : 3 - j));
    std::memcpy(&values[i], &bits, 4);
    require(std::isfinite(values[i]), "Features must be finite");
  }
  return values;
}

// --- Run report workspace --------------------------------------------------------
namespace {
std::string utc() {
  const auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::ostringstream out;
  out << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}
void save(const fs::path &path, const Json &report) {
  const auto temporary = path.string() + ".tmp";
  std::ofstream file(temporary, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot create report");
  file << report.dump(2) << '\n';
  file.close();
  if (!file)
    throw std::runtime_error("Cannot write report");
  fs::rename(temporary, path);
}
Json tensors(const std::vector<TensorMetadata> &metadata) {
  Json out = Json::array();
  for (const auto &m : metadata)
    out.push_back({{"name", m.name},
                   {"role", m.role},
                   {"dtype", m.dtype},
                   {"shape", m.shape},
                   {"strides", m.strides},
                   {"allocation_bytes", m.allocation_bytes}});
  return out;
}
} // namespace
std::string file_digest(const std::string &path) {
  auto value = rdk::sha256_file(path);
  if (value.size() != 64)
    throw std::invalid_argument("Cannot hash file: " + path);
  return value;
}
void require_manifest_intact(const std::string &path,
                             const std::string &sha256) {
  if (file_digest(path) != sha256)
    throw std::runtime_error("Manifest changed while reading");
}
struct RunWorkspace::Reserved {
  fs::path output;
  std::string manifest_path, manifest_sha256, result_path;
  Json report = Json::object();
};
RunWorkspace::RunWorkspace(const CliOptions &options, std::string manifest_sha256,
                           std::string backend)
    : reserved_(std::make_unique<Reserved>()) {
  if (fs::exists(options.output) || fs::is_symlink(options.output))
    throw std::invalid_argument("Output directory must be new");
  const fs::path destination = options.output;
  if (!destination.parent_path().empty())
    fs::create_directories(destination.parent_path());
  if (!fs::create_directory(destination))
    throw std::runtime_error("Cannot create new output directory");
  reserved_->output = destination;
  reserved_->result_path = (destination / "result.json").string();
  reserved_->manifest_path = options.manifest;
  reserved_->manifest_sha256 = std::move(manifest_sha256);
  reserved_->report = {{"schema", "rdk-model-zoo/paraformer-native-run/v1"},
                       {"status", "running"},
                       {"execution_backend", backend},
                       {"target", "s100"},
                       {"started_utc", utc()},
                       {"manifest_path",
                        fs::absolute(options.manifest).string()},
                       {"manifest_sha256", reserved_->manifest_sha256},
                       {"vocabulary_sha256", kVocabularySha256},
                       {"inference_attempted", false},
                       {"inference_executed", false},
                       {"models", Json::array()},
                       {"records", Json::array()}};
  const std::array<std::string, 3> stages{"encoder", "predictor", "decoder"};
  for (size_t i = 0; i < 3; ++i) {
    const auto &artifact = options.models[i];
    reserved_->report["models"].push_back(
        {{"stage", stages[i]},
         {"asset_id", artifact.asset_id},
         {"path", fs::absolute(artifact.model.path).string()},
         {"sha256", artifact.expected_sha256}});
  }
}
RunWorkspace::~RunWorkspace() = default;
void RunWorkspace::note_metadata(const SdkMetadata &metadata, size_t index) {
  reserved_->report["models"].at(index)["metadata"] = {
      {"model_name", metadata.model_name},
      {"inputs", tensors(metadata.inputs)},
      {"outputs", tensors(metadata.outputs)}};
}
void RunWorkspace::mark_attempted() {
  reserved_->report["inference_attempted"] = true;
  if (reserved_->report["records"].empty())
    reserved_->report["inference_executed"] = nullptr;
}
void RunWorkspace::add_record(const FeatureItem &item, const Prediction &result) {
  reserved_->report["inference_executed"] = true;
  Json timings = {{"encoder_ms", result.timings.encoder_ms},
                  {"predictor_ms", result.timings.predictor_ms},
                  {"cif_ms", result.timings.cif_ms},
                  {"decoder_ms", result.timings.decoder_ms
                                     ? Json(*result.timings.decoder_ms)
                                     : Json(nullptr)}};
  Json record = {{"utt_id", item.utt_id},
                 {"feature_path", item.path},
                 {"feature_sha256", item.sha256},
                 {"valid_frames", item.valid_frames},
                 {"original_frames", item.original_frames},
                 {"truncated", item.truncated},
                 {"source_record", Json::parse(item.original_record_json)},
                 {"text", result.text},
                 {"token_ids", result.token_ids},
                 {"token_count", result.token_count},
                 {"decoder_executed", result.decoder_executed},
                 {"timings", timings}};
  if (item.reference_text)
    record["reference_text"] = *item.reference_text;
  reserved_->report["records"].push_back(std::move(record));
}
void RunWorkspace::complete(const std::vector<FeatureItem> &items) {
  if (file_digest(reserved_->manifest_path) != reserved_->manifest_sha256)
    throw std::runtime_error("Manifest changed during inference");
  for (const auto &item : items)
    if (file_digest(item.path) != item.sha256)
      throw std::runtime_error("Feature changed during inference: " +
                               item.utt_id);
  reserved_->report["status"] = "completed";
  reserved_->report["finished_utc"] = utc();
  save(reserved_->output / "result.json", reserved_->report);
}
void RunWorkspace::fail(const std::exception &error,
                        const std::string &current_utt_id) {
  try {
    reserved_->report["status"] = "failed";
    reserved_->report["finished_utc"] = utc();
    reserved_->report["error"] = error.what();
    reserved_->report["current_utt_id"] = current_utt_id;
    save(reserved_->output / "failed.json", reserved_->report);
  } catch (const std::exception &failure) {
    std::cerr << "Could not save failure report: " << failure.what() << '\n';
  }
}
const std::string &RunWorkspace::output() const {
  return reserved_->result_path;
}
} // namespace paraformer
