#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gemma4_config.h"

namespace gemma4 {

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
