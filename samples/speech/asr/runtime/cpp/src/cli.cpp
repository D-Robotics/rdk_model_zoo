// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include "sha256.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sndfile.h>
#include <stdexcept>
namespace fs = std::filesystem;
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
         "          --decode-mode ctc|legacy (legacy), --help\n"
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

// --- AudioReader: libsndfile chunking, no DSP -------------------------------
struct AudioReader::Impl {
  struct Close {
    void operator()(SNDFILE *p) const {
      if (p)
        sf_close(p);
    }
  };
  std::unique_ptr<SNDFILE, Close> file;
  SF_INFO info{};
  size_t block = 0, offset = 0, index = 0;
};
AudioReader::AudioReader(const std::string &path)
    : impl_(std::make_unique<Impl>()) {
  impl_->file.reset(sf_open(path.c_str(), SFM_READ, &impl_->info));
  if (!impl_->file)
    throw std::runtime_error("Cannot open audio: " + path);
  const auto &info = impl_->info;
  if (info.samplerate <= 0 || info.channels <= 0 || info.frames <= 0)
    throw std::invalid_argument(
        "Audio must contain frames with valid rate/channels");
  impl_->block = source_chunk_size(info.samplerate);
  if (impl_->block >
          std::numeric_limits<size_t>::max() / size_t(info.channels) ||
      impl_->block >
          static_cast<size_t>(std::numeric_limits<sf_count_t>::max()))
    throw std::invalid_argument("Audio chunk dimensions overflow");
}
AudioReader::~AudioReader() = default;
bool AudioReader::next(AudioChunk &chunk) {
  chunk = AudioChunk{};
  const auto &info = impl_->info;
  std::vector<float> samples(impl_->block * size_t(info.channels));
  const sf_count_t count = sf_readf_float(
      impl_->file.get(), samples.data(), static_cast<sf_count_t>(impl_->block));
  if (sf_error(impl_->file.get()) != SF_ERR_NO_ERROR || count < 0)
    throw std::runtime_error("Audio read failed: " +
                             std::string(sf_strerror(impl_->file.get())));
  if (!count)
    return false;
  if (impl_->offset > std::numeric_limits<size_t>::max() - size_t(count))
    throw std::overflow_error("Audio frame offset overflow");
  samples.resize(size_t(count) * size_t(info.channels));
  chunk = {std::move(samples), info.samplerate, info.channels, impl_->offset,
           impl_->index};
  impl_->offset += size_t(count);
  ++impl_->index;
  return true;
}

// --- Run presentation -------------------------------------------------------
std::string file_digest(const std::string &path) {
  if (!fs::is_regular_file(path) || fs::file_size(path) == 0)
    throw std::invalid_argument("Missing or empty regular input: " + path);
  const auto digest_value = rdk::sha256_file(path);
  if (digest_value.size() != 64)
    throw std::runtime_error("Cannot hash input: " + path);
  return digest_value;
}
fs::path reserve_output(const std::string &output) {
  if (fs::exists(output) || fs::is_symlink(output))
    throw std::invalid_argument("Output directory must be new");
  const fs::path destination = output;
  if (!destination.parent_path().empty())
    fs::create_directories(destination.parent_path());
  if (!fs::create_directory(destination))
    throw std::runtime_error("Cannot create new output directory");
  return destination;
}
void save_report(const fs::path &path, const nlohmann::json &report) {
  const fs::path temporary = path.string() + ".tmp";
  std::ofstream out(temporary, std::ios::binary);
  if (!out)
    throw std::runtime_error("Cannot create report: " + temporary.string());
  out << report.dump(2) << '\n';
  out.close();
  if (!out)
    throw std::runtime_error("Cannot write report: " + temporary.string());
  fs::rename(temporary, path);
}
nlohmann::json initial_report(const CliOptions &options,
                              const std::string &audio_sha256,
                              const char *backend) {
  return {{"schema", "rdk-model-zoo/asr-native-run/v1"},
          {"status", "running"},
          {"execution_backend", backend},
          {"target", options.model.target},
          {"asset_id", options.asset_id},
          {"model_sha256", options.model_sha256},
          {"audio_sha256", audio_sha256},
          {"vocabulary_sha256", asr::kVocabularySha256},
          {"decode_mode", options.decode_mode},
          {"frontend", "libsamplerate-sinc-best; independent windows; "
                       "var+1e-5 before padding"},
          {"config", {{"audio_maxlen", 30000}, {"new_rate", 16000}}},
          {"chunks", nlohmann::json::array()}};
}
nlohmann::json metadata_record(const SdkMetadata &metadata) {
  return {{"model_name", metadata.model_name},
          {"input_shape", {1, 30000}},
          {"output_shape", {1, metadata.steps, 3503}},
          {"dtype", "float32"},
          {"input_strides", metadata.input_strides},
          {"output_strides", metadata.output_strides},
          {"input_bytes", metadata.input_bytes},
          {"output_bytes", metadata.output_bytes}};
}
nlohmann::json chunk_record(const AudioChunk &chunk, const Prediction &result) {
  return {{"index", chunk.index},
          {"source_start", chunk.source_start},
          {"source_frames", chunk.samples.size() / size_t(chunk.channels)},
          {"source_rate", chunk.sample_rate},
          {"valid_target_samples", result.valid_samples},
          {"text", result.text}};
}

// --- RunWorkspace: report lifecycle ------------------------------------------
struct RunWorkspace::Reserved {
  CliOptions options;
  std::string audio_sha256;
  fs::path output;
  nlohmann::json report;
  std::string text;
  size_t chunks = 0;
};
RunWorkspace::RunWorkspace(const CliOptions &options,
                           const std::string &audio_sha256, const char *backend)
    : reserved_(std::make_unique<Reserved>()) {
  reserved_->options = options;
  reserved_->audio_sha256 = audio_sha256;
  reserved_->output = reserve_output(options.output);
  reserved_->report = initial_report(options, audio_sha256, backend);
}
RunWorkspace::~RunWorkspace() = default;
void RunWorkspace::note_metadata(const SdkMetadata &metadata) {
  reserved_->report["metadata"] = metadata_record(metadata);
}
void RunWorkspace::add_chunk(const AudioChunk &chunk, const Prediction &result) {
  reserved_->text += result.text;
  ++reserved_->chunks;
  reserved_->report["chunks"].push_back(chunk_record(chunk, result));
}
size_t RunWorkspace::chunk_count() const { return reserved_->chunks; }
void RunWorkspace::complete() {
  if (file_digest(reserved_->options.audio) != reserved_->audio_sha256 ||
      file_digest(reserved_->options.model.path) !=
          reserved_->options.model_sha256 ||
      file_digest(reserved_->options.vocabulary) != kVocabularySha256)
    throw std::runtime_error(
        "Input/model/vocabulary changed during inference");
  reserved_->report["status"] = "completed";
  reserved_->report["text"] = reserved_->text;
  save_report(reserved_->output / "result.json", reserved_->report);
}
void RunWorkspace::fail(const std::exception &error) {
  try {
    reserved_->report["status"] = "failed";
    reserved_->report["error"] = error.what();
    save_report(reserved_->output / "failed.json", reserved_->report);
  } catch (const std::exception &report_error) {
    std::cerr << "Could not save failure report: " << report_error.what()
              << '\n';
  }
}
const fs::path &RunWorkspace::output() const { return reserved_->output; }
const std::string &RunWorkspace::text() const { return reserved_->text; }
} // namespace asr
