#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gemma4_native_tokenizer.h"

namespace gemma4 {

// Bridge that keeps the original TokenizerBridge API (EncodeMessagesJson /
// DecodeIds) but is backed by the native C++ tokenizer (tokenizers-cpp),
// matching the OpenExplorer_LLM-s600 reference implementation. No Python.
class TokenizerBridge {
 public:
  explicit TokenizerBridge(const std::string& tokenizer_dir = "");

  std::vector<int64_t> EncodeMessagesJson(const std::string& messages_json,
                                          bool expand_images = true) const;

  std::string DecodeIds(const std::vector<int64_t>& ids) const;

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
};

}  // namespace gemma4
