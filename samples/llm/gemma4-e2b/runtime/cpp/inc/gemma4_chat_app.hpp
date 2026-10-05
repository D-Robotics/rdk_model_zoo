/**
 * @file gemma4_chat_app.hpp
 * @brief Application facade for the interactive Gemma4-E2B VLM chat.
 *
 * InteractiveChatApp is the *application* layer of the `main` executable: it
 * owns the console session (banner, help, REPL prompts, streaming echo) and
 * the conversation bookkeeping (history, token budgeting, KV-cache reuse,
 * image turns). It is deliberately NOT a model class: all model work is
 * delegated to the real runtime classes — gemma4::VisionEngine,
 * gemma4::TextEngine (Generate / ContinueGenerateStream / BuildPromptHidden /
 * ResetSession) and gemma4::TokenizerBridge — which never perform console
 * IO themselves. Keeping this boundary in one named class lets `main.cpp`
 * stay a thin entry: parse flags, resolve paths, construct the app, run it.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "gemma4_text_engine.hpp"
#include "gemma4_tokenizer.hpp"
#include "gemma4_vision_engine.hpp"

namespace gemma4::chat {

/** @brief Resolved model/tokenizer locations (no environment lookups here). */
struct ChatPaths {
  std::string text_hbm;        ///< Compiled text LLM *.hbm.
  std::string vision_hbm;      ///< Compiled vision ViT *.hbm.
  std::string tok_embeddings;  ///< External token embedding table.
  std::string tokenizer_json;  ///< HF tokenizer.json.
};

/** @brief Generation budget settings for one chat session. */
struct ChatSettings {
  int max_tokens = 0;  ///< New tokens per turn; 0 uses all remaining KV capacity.
  int min_response_tokens = 256;  ///< Reply capacity kept while trimming history.
  bool rebuild_context_each_turn = false;  ///< Re-prefill the full context per turn.
};

/** @brief One chat turn: role, text and whether it carries the active image. */
struct Message {
  std::string role;
  std::string content;
  bool has_image = false;
};

/**
 * @brief Own the engines plus one interactive console session.
 *
 * The constructor loads the Vision and Text engines and the tokenizer in the
 * source lifecycle order (Vision before Text) and prints the same load
 * progress as the historical entry. Run() executes the read-eval-print loop
 * on std::cin/std::cout until /quit or EOF and returns the process exit code
 * (0 on a clean exit, 1 on a propagated engine exception).
 *
 * Console IO belongs to this class only; the engines print nothing
 * implicitly. Instances are single-session and not thread-safe; callers must
 * serialize access exactly as with the underlying TextEngine.
 */
class InteractiveChatApp {
 public:
  InteractiveChatApp(const ChatPaths& paths, const ChatSettings& settings);
  ~InteractiveChatApp() = default;

  InteractiveChatApp(const InteractiveChatApp&) = delete;
  InteractiveChatApp& operator=(const InteractiveChatApp&) = delete;

  /** @brief Run the interactive loop; @return the process exit code. */
  int Run();

 private:
  ChatSettings settings_;
  int response_reserve_;  ///< min(min_response_tokens, kCacheLen - 1).

  // Constructed in the constructor body so the load-progress messages keep
  // the historical order (banner, "Loading vision...", then the engines).
  std::unique_ptr<VisionEngine> vision_;
  std::unique_ptr<TextEngine> engine_;
  std::unique_ptr<TokenizerBridge> tokenizer_;

  std::vector<Message> history_;
  std::vector<int64_t> session_ids_;
  int turn_count_ = 0;

  // One active image is supported per conversation. Its compact vision
  // features are retained so follow-up turns can still reference the image.
  std::string pending_image_;
  std::vector<float> pending_vision_features_;
  bool has_pending_image_ = false;
};

}  // namespace gemma4::chat
