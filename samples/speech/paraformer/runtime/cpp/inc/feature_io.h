// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <optional>
#include <string>
#include <vector>
namespace paraformer {
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
} // namespace paraformer
