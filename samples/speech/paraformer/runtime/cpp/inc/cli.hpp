// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "pipeline.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>
namespace paraformer {

// --- Argument parsing and vocabulary ------------------------------------------
struct CliOptions {
  ModelGroup models;
  std::string manifest, vocabulary, output;
  size_t max_utts = 0;
  bool help = false;
};
CliOptions parse_cli(int argc, const char *const *argv);
std::string cli_help();
std::vector<std::string> load_vocabulary(const std::string &path);

// --- Prepared-feature manifest and NPY features ---------------------------------
struct FeatureItem {
  std::string utt_id, path, sha256, original_record_json;
  int valid_frames = 0, original_frames = 0;
  bool truncated = false;
  std::optional<std::string> reference_text;
};
// Validate every record structurally before selecting the optional prefix.
// Relative feature paths resolve against the manifest parent, not process cwd.
std::vector<FeatureItem> load_prepared_manifest(const std::string &path,
                                                size_t max_utts = 0);
// Reads and hashes the same owned byte buffer. Only finite C-order float32
// [1,400,560] NPY arrays are accepted; returned values are native float32.
std::vector<float> load_features(const FeatureItem &item);

// --- Run report workspace -------------------------------------------------------
std::string file_digest(const std::string &path);
void require_manifest_intact(const std::string &path,
                             const std::string &sha256);
// Owns the report lifecycle main drives per utterance: reserves a new output
// directory, records observed model metadata and per-utterance predictions,
// re-verifies manifest/feature digests on completion and writes
// result.json/failed.json atomically. main never assembles JSON itself.
class RunWorkspace {
public:
  RunWorkspace(const CliOptions &options, std::string manifest_sha256,
               std::string backend);
  ~RunWorkspace();
  RunWorkspace(const RunWorkspace &) = delete;
  RunWorkspace &operator=(const RunWorkspace &) = delete;
  void note_metadata(const SdkMetadata &metadata, size_t index);
  void mark_attempted();
  void add_record(const FeatureItem &item, const Prediction &result);
  void complete(const std::vector<FeatureItem> &items);
  void fail(const std::exception &error, const std::string &current_utt_id);
  const std::string &output() const;

private:
  struct Reserved;
  std::unique_ptr<Reserved> reserved_;
};
} // namespace paraformer
