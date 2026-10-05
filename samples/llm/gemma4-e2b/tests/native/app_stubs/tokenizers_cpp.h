// Host compile-check stub for the tokenizers-cpp third-party header.
//
// This is NOT the vendored tokenizers-cpp library and performs no
// tokenization. The production chat-app host check compiles the real
// gemma4_chat_app.cpp / main.cpp sources; their headers include
// "tokenizers_cpp.h", whose real copy is only prepared on the build host by
// third_party/install_tokenizers_cpp.sh. This stub provides exactly the
// surface those headers reference so the production sources can be
// syntax/type checked on a host without the third-party dependency. Engine
// behavior is supplied by the chat_app_doubles.cpp SDK doubles; nothing here
// represents board or tokenizer behavior.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {

class Tokenizer {
 public:
  virtual ~Tokenizer() = default;
  size_t GetVocabSize() const { return 0; }
};

}  // namespace tokenizers
