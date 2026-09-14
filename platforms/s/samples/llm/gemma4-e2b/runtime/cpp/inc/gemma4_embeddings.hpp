/**
 * @file gemma4_embeddings.hpp
 * @brief External token embedding lookup table for Gemma4-E2B.
 *
 * Loads `tok_embeddings.bin` from disk (f32 or f16 format auto-detected)
 * and provides per-token-id row lookup. Used by the text engine to feed
 * `inputs_embeds` at prefill / decode time, with optional vision-feature
 * injection at image-token positions.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gemma4_config.hpp"

namespace gemma4 {

/**
 * @brief Own and query the external Gemma4-E2B token embedding table.
 *
 * The table is loaded once at construction and can build text-only or
 * vision-injected hidden states for prefill.
 */
class TokenEmbeddings {
 public:
  explicit TokenEmbeddings(const std::string& path);

  void Lookup(const std::vector<int64_t>& ids, float* out) const;

  // Lookup a single token embedding row.
  const float* GetRow(int64_t id) const;

  std::vector<float> BuildPromptHidden(
      const std::vector<int64_t>& ids,
      const std::vector<float>& vision_features) const;

 private:
  std::vector<float> table_;
};

}  // namespace gemma4
