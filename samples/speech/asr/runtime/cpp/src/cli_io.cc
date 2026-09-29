// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.h"
#include "contract.h"
#include "preflight.h"
#include "sha256.h"
#include <fstream>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
namespace asr {
CliOptions parse_cli(int argc, const char *const *argv) {
  CliOptions options;
  std::set<std::string> seen;
  std::map<std::string, std::string *> fields = {
      {"--target", &options.model.target},
      {"--asset-id", &options.asset_id},
      {"--model-path", &options.model.path},
      {"--model-sha256", &options.model_sha256},
      {"--audio-file", &options.audio},
      {"--vocab-file", &options.vocabulary},
      {"--output-dir", &options.output},
      {"--decode-mode", &options.decode_mode}};
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
    if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0 ||
        !argv[i + 1][0])
      throw std::invalid_argument("Missing value: " + key);
    *found->second = argv[++i];
  }
  if (options.help)
    return options;
  if (options.model.target != "s100" && options.model.target != "s600")
    throw std::invalid_argument("Native ASR requires --target s100 or s600");
  if (options.asset_id != "s:asr:" + options.model.target + "/asr.hbm")
    throw std::invalid_argument("Exact target-specific --asset-id required");
  if (options.model.path.empty())
    throw std::invalid_argument("--model-path is required");
  (void)make_preflight(options.model_sha256,
                       options.vocabulary); // validate digest syntax only
  std::transform(options.model_sha256.begin(), options.model_sha256.end(),
                 options.model_sha256.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (options.decode_mode != "ctc" && options.decode_mode != "legacy")
    throw std::invalid_argument("--decode-mode must be ctc or legacy");
  return options;
}
std::string cli_help() {
  return "ASR native inference (repository-root default paths)\n"
         "Required: --target s100|s600 --asset-id s:asr:<target>/asr.hbm\n"
         "          --model-path FILE --model-sha256 HEX64\n"
         "Optional: --audio-file FILE "
         "(samples/speech/asr/test_data/chi_sound.wav)\n"
         "          --vocab-file FILE "
         "(samples/speech/asr/test_data/vocab.json)\n"
         "          --output-dir DIR (outputs/asr_cpp/result; must be new)\n"
         "          --decode-mode ctc|legacy (ctc), --help\n"
         "Fixed frontend: 30000 samples at 16000 Hz, independent sinc "
         "windows.\n";
}
std::vector<std::string> load_vocabulary(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::invalid_argument("Cannot read vocabulary");
  const std::string bytes(std::istreambuf_iterator<char>(file), {});
  if (file.bad() ||
      rdk::sha256_hex(bytes.data(), bytes.size()) != kVocabularySha256)
    throw std::invalid_argument("Vocabulary SHA-256 mismatch");
  const auto mapping = nlohmann::json::parse(bytes);
  if (!mapping.is_object() || mapping.size() != 3503)
    throw std::invalid_argument("Expected 3503-token JSON vocabulary");
  std::vector<std::string> tokens(3503);
  for (auto it = mapping.begin(); it != mapping.end(); ++it) {
    if (!it.value().is_number_integer())
      throw std::invalid_argument("Vocabulary ID must be an integer");
    const auto id = it.value().get<int64_t>();
    if (id < 0 || id >= 3503 || !tokens[id].empty() || it.key().empty())
      throw std::invalid_argument("Invalid or duplicate vocabulary ID");
    tokens[id] = it.key();
  }
  validate_vocabulary(tokens);
  return tokens;
}
} // namespace asr
