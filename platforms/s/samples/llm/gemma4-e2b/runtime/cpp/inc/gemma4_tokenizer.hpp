/**
 * @file gemma4_tokenizer.hpp
 * @brief Tokenizer bridge backed by native C++ tokenizers-cpp.
 *
 * Keeps the original TokenizerBridge API (EncodeMessagesJson / DecodeIds)
 * but is backed by the native C++ tokenizer (tokenizers-cpp), matching the
 * OpenExplorer_LLM-s600 reference implementation. No Python required.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gemma4_native_tokenizer.hpp"

namespace gemma4 {

/**
 * @brief Native C++ tokenizer bridge for Gemma4-E2B chat.
 *
 * Encodes chat messages (JSON format) into token IDs and decodes token IDs
 * back to text, using the vendored tokenizers-cpp HuggingFace binding.
 */
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
