// Host SDK/engine doubles for the interactive chat application check.
//
// These doubles are NOT the vendor SDK and NOT the Gemma4 engines: no HBM
// files are read, no BPU inference runs and no real tokenization happens.
// They exist so the production application source (src/gemma4_chat_app.cpp)
// can be compiled and its session logic — history budgeting, image turns,
// streaming display, reset/context commands — executed on a host. Engine
// numerical behavior remains board-only and is explicitly not-run here.
//
// Double behavior, kept deterministic for the scenario driver:
//  * TokenizerBridge::EncodeMessagesJson — one token id per JSON byte.
//  * TokenizerBridge::DecodeIds — "W" per id.
//  * TextEngine::ContinueGenerateStream — echoes the prompt ids, then
//    generates {777, 888} (streamed) plus one turn-end token (not streamed),
//    mirroring the real engine's full-vector + stop-token contract.
//  * PredictVision — validates the image path through LoadImage, calls the
//    runner exactly once and returns its [280,1536] result.

#include "chat_app_doubles.hpp"

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "gemma4_chat_app.hpp"
#include "gemma4_image_io.hpp"
#include "gemma4_vision_task.hpp"

namespace chat_test {

DoubleState& Doubles() {
  static DoubleState state;
  return state;
}

}  // namespace chat_test

// ModelIo::Clear() (inline in gemma4_model_io.hpp) releases UCP buffers; the
// doubles never allocate any, so this is a satisfied no-op declaration.
int hbUCPFree(hbUCPSysMem* mem) {
  (void)mem;
  return 0;
}

namespace gemma4 {

// ---- TokenEmbeddings / KvCache construction referenced by TextEngine ----

TokenEmbeddings::TokenEmbeddings(const std::string& path) {
  (void)path;
}

KvCache::KvCache() = default;
KvCache::~KvCache() = default;

Tokenizer::~Tokenizer() = default;

// ---- VisionEngine ----

VisionEngine::VisionEngine(const std::string& vision_hbm) {
  (void)vision_hbm;
  ++chat_test::Doubles().vision_ctors;
  load_ms_ = 7;
}

VisionEngine::~VisionEngine() { ++chat_test::Doubles().vision_dtors; }

std::vector<float> VisionEngine::Infer(const std::vector<float>& patches) {
  (void)patches;
  ++chat_test::Doubles().vision_infers;
  return std::vector<float>(280 * 1536, 0.25f);
}

// ---- TextEngine ----

TextEngine::TextEngine(const std::string& text_hbm,
                       const std::string& embed_path)
    : embeddings_(embed_path) {
  (void)text_hbm;
  ++chat_test::Doubles().text_ctors;
  load_ms_ = 42;
}

TextEngine::~TextEngine() { ++chat_test::Doubles().text_dtors; }

std::vector<float> TextEngine::BuildPromptHidden(
    const std::vector<int64_t>& prompt_ids,
    const std::vector<float>& vision_features) const {
  (void)vision_features;
  ++chat_test::Doubles().prompt_hiddens;
  return std::vector<float>(prompt_ids.size() * 16, 0.5f);
}

std::vector<int64_t> TextEngine::ContinueGenerateStream(
    const std::vector<int64_t>& full_ids, int max_new_tokens,
    TokenCallback on_token, const std::vector<float>* full_hidden) {
  ++chat_test::Doubles().continues;
  auto& state = chat_test::Doubles();
  state.last_max_tokens.push_back(max_new_tokens);
  state.hidden_flags.push_back(full_hidden != nullptr ? 1 : 0);
  state.last_full_ids = full_ids;
  state.full_id_sizes.push_back(full_ids.size());
  session_.processed_tokens = static_cast<int>(full_ids.size());
  std::vector<int64_t> out = full_ids;
  for (const int64_t id : {int64_t{777}, int64_t{888}}) {
    out.push_back(id);
    if (on_token) {
      on_token(id);
    }
  }
  out.push_back(gemma4::kTurnEndTokenId);
  return out;
}

void TextEngine::ResetSession() {
  ++chat_test::Doubles().resets;
  session_.processed_tokens = 0;
  session_.token_offset = 0;
}

// ---- TokenizerBridge ----

TokenizerBridge::TokenizerBridge(const std::string& tokenizer_dir) {
  (void)tokenizer_dir;
  ++chat_test::Doubles().tokenizer_ctors;
}

std::vector<int64_t> TokenizerBridge::EncodeMessagesJson(
    const std::string& messages_json, bool expand_images) const {
  (void)expand_images;
  // Deterministic byte-count encoding: one token id per JSON byte.
  std::vector<int64_t> ids;
  ids.reserve(messages_json.size());
  for (size_t index = 0; index < messages_json.size(); ++index) {
    ids.push_back(1000 + static_cast<int64_t>(index % 500));
  }
  return ids;
}

std::string TokenizerBridge::DecodeIds(
    const std::vector<int64_t>& ids) const {
  ++chat_test::Doubles().tokenizer_decodes;
  return std::string(ids.size(), 'W');
}

// ---- Image IO and vision composition ----

cv::Mat LoadImage(const std::string& path) {
  ++chat_test::Doubles().load_images;
  (void)path;
  return cv::Mat{};
}

std::vector<float> PredictVision(const cv::Mat& bgr,
                                 const VisionRunner& runner) {
  (void)bgr;
  ++chat_test::Doubles().predict_visions;
  return runner(std::vector<float>(2520 * 768, 0.5f));
}

}  // namespace gemma4
