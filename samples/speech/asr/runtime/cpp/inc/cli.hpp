// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "asr.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
namespace asr {
struct CliOptions {
  SdkModel model;
  std::string asset_id, model_sha256;
  std::string audio = "samples/speech/asr/test_data/chi_sound.wav";
  std::string vocabulary = "samples/speech/asr/test_data/vocab.json";
  std::string output = "outputs/asr_cpp/result", decode_mode = "legacy";
  bool help = false;
};
CliOptions parse_cli(int argc, const char *const *argv);
std::string cli_help();
std::vector<std::string> load_vocabulary(const std::string &path);

// File reading only: no channel mixing, resampling, normalization or SDK calls.
class AudioReader {
public:
  explicit AudioReader(const std::string &path);
  ~AudioReader();
  AudioReader(const AudioReader &) = delete;
  AudioReader &operator=(const AudioReader &) = delete;
  bool next(AudioChunk &chunk); // false at clean EOF; throws on read errors
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Run presentation helpers: digests, exclusive output reservation, reports.
std::string file_digest(const std::string &path);
std::filesystem::path reserve_output(const std::string &output);
void save_report(const std::filesystem::path &path,
                 const nlohmann::json &report);
nlohmann::json initial_report(const CliOptions &options,
                              const std::string &audio_sha256,
                              const char *backend);
nlohmann::json metadata_record(const SdkMetadata &metadata);
nlohmann::json chunk_record(const AudioChunk &chunk, const Prediction &result);

// Owns the run report lifecycle: reserves the new output directory, collects
// metadata and per-chunk records from predict results, re-verifies input
// digests at completion and writes result.json/failed.json. main drives the
// flow; no JSON assembly lives there.
class RunWorkspace {
public:
  RunWorkspace(const CliOptions &options, const std::string &audio_sha256,
               const char *backend);
  ~RunWorkspace();
  RunWorkspace(const RunWorkspace &) = delete;
  RunWorkspace &operator=(const RunWorkspace &) = delete;
  void note_metadata(const SdkMetadata &metadata);
  void add_chunk(const AudioChunk &chunk, const Prediction &result);
  size_t chunk_count() const;
  void complete(); // re-hash inputs, mark completed, save result.json
  void fail(const std::exception &error); // save failed.json; never throws
  const std::filesystem::path &output() const;
  const std::string &text() const;

private:
  struct Reserved;
  std::unique_ptr<Reserved> reserved_;
};
} // namespace asr
