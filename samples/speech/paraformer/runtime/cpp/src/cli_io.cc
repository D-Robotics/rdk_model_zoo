// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include "contract.h"
#include "sha256.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
namespace paraformer {
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
} // namespace paraformer
