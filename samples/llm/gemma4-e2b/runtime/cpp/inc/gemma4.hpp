/**
 * @file gemma4.hpp
 * @brief Named Gemma4-E2B model owning the engines and one chat session.
 *
 * Gemma4 is the model layer of the `main` executable: its constructor loads
 * the Vision engine, then the Text engine and the tokenizer, and one predict
 * call executes one chat request —
 * conversation history, token budgeting, KV-cache reuse decisions, vision
 * feature preparation and autoregressive generation with optional streaming.
 * The class performs no console IO: load progress and context notices are
 * emitted through injected sinks, and generated fragments are handed to a
 * caller-owned token sink, so presentation (banner, prompts, echo) stays
 * with the CLI in cli.hpp/.cpp. Model execution is delegated to the
 * real runtime classes — gemma4::VisionEngine, gemma4::TextEngine
 * (ContinueGenerateStream / BuildPromptHidden / ResetSession) and
 * gemma4::TokenizerBridge — which never perform console IO themselves.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "gemma4_text_engine.hpp"
#include "gemma4_tokenizer.hpp"
#include "gemma4_vision_engine.hpp"

namespace gemma4 {

/** @brief Sink for one complete status/debug line (no trailing newline). */
using ChatEventSink = std::function<void(const std::string&)>;

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

/** @brief KV-cache usage snapshot for the /context report. */
struct ChatContextUsage {
  size_t used_tokens = 0;       ///< Tokens currently in the session.
  size_t remaining_tokens = 0;  ///< Free capacity before the KV limit.
  int turns = 0;                ///< Completed chat turns.
};

/** @brief Owned result of one predict call, carrying the report facts. */
struct ChatTurnResult {
  bool prompt_too_long = false;  ///< Prompt alone exceeded the KV cache; the
                                 ///< history was restored and nothing ran.
  size_t prompt_tokens = 0;      ///< Encoded prompt size of this turn.
  int output_budget = 0;         ///< Generation limit applied this turn.
  bool history_trimmed = false;  ///< Oldest turns were dropped to fit.
  std::vector<int64_t> generated_ids;  ///< Generated ids without stop tokens.
  std::string reply;             ///< Decoded reply text.
  int streamed_tokens = 0;       ///< Tokens passed to the token sink.
  double elapsed_ms = 0;         ///< Generation wall time.
  double tokens_per_sec = 0;     ///< Streamed tokens per second.
};

/**
 * @brief Own the engines plus one interactive conversation session.
 *
 * The constructor loads the Vision and Text engines and the tokenizer and
 * reports load progress through the status sink (silently when the sink is
 * null). LoadImage prepares one active image for the following turns,
 * predict executes one request per call and returns an owned result, Reset
 * clears the conversation and ContextUsage reports KV usage.
 *
 * Instances are single-session and not thread-safe; callers must serialize
 * access exactly as with the underlying TextEngine.
 */
class Gemma4 {
 public:
  /**
   * @brief Load the engines/tokenizer and prepare the session budget.
   *
   * @param paths   Model and tokenizer locations (validated by the engines).
   * @param settings Generation budget for every turn.
   * @param status  Optional sink for load progress and context notices.
   * @param debug   Optional sink for the context-reuse debug line.
   */
  Gemma4(const ChatPaths& paths, const ChatSettings& settings,
         ChatEventSink status = nullptr, ChatEventSink debug = nullptr);
  ~Gemma4() = default;

  Gemma4(const Gemma4&) = delete;
  Gemma4& operator=(const Gemma4&) = delete;
  Gemma4(Gemma4&&) = delete;
  Gemma4& operator=(Gemma4&&) = delete;

  /**
   * @brief Make @p path the active image for the next turns.
   *
   * Starts a fresh conversation when one is already in progress (history,
   * KV cache and any previous image are dropped). Runs the full vision
   * path — decode, preprocess, ViT inference — and retains the soft image
   * tokens for the next predict call.
   *
   * @return Size of the retained vision feature vector.
   * @throws std::runtime_error when the image cannot be decoded.
   */
  size_t LoadImage(const std::string& path);

  /**
   * @brief Execute one chat request on the conversation state.
   *
   * Appends the user message, budgets the prompt against the KV cache
   * (trimming the oldest turns when needed), maintains prefix-reuse or
   * resets the engine session, generates with optional per-token streaming
   * and appends the decoded reply to the history. On an oversized prompt
   * the state is restored and a result with prompt_too_long is returned
   * without generation.
   *
   * @param text     User message text (UTF-8).
   * @param on_token Optional sink receiving each streamed fragment.
   */
  ChatTurnResult predict(const std::string& text,
                         ChatEventSink on_token = nullptr);

  /** @brief Clear history, KV cache and any active image. */
  void Reset();

  /** @brief Current KV-cache usage of the session. */
  ChatContextUsage ContextUsage() const;

 private:
  void Emit(const std::string& line) const;

  ChatSettings settings_;
  int response_reserve_;  ///< min(min_response_tokens, kCacheLen - 1).
  ChatEventSink status_;
  ChatEventSink debug_;

  // Constructed in the constructor body so the load-progress messages match
  // the load order ("Loading vision...", then the text engine).
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

}  // namespace gemma4
