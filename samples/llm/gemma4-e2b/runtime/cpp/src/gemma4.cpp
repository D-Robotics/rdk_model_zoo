/**
 * @file gemma4.cpp
 * @brief Session logic for the Gemma4-E2B interactive chat model.
 *
 * Everything here is conversation bookkeeping moved from the historical
 * application facade without algorithm changes: chat-history JSON for the
 * tokenizer, oldest-turn trimming against the 4096-token KV budget,
 * prefix-mismatch resets and the streamed generation turn. Model execution
 * stays in the engines: PredictVision + VisionEngine::Infer for image turns,
 * TextEngine::ContinueGenerateStream (with the optional BuildPromptHidden
 * injection) for generation. All status/debug emission goes through the
 * injected sinks; this file performs no console IO.
 */

#include "gemma4.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "nlohmann/json.hpp"

#include "gemma4_config.hpp"

namespace gemma4 {
namespace {

std::string BuildMessagesJson(const std::vector<Message>& history) {
  nlohmann::json messages = nlohmann::json::array();
  for (const auto& history_item : history) {
    nlohmann::json message;
    message["role"] = history_item.role;
    if (history_item.has_image) {
      message["content"] = nlohmann::json::array(
          {{{"type", "image"}},
           {{"type", "text"}, {"text", history_item.content}}});
    } else {
      message["content"] = history_item.content;
    }
    messages.push_back(std::move(message));
  }
  return messages.dump();
}

bool DropOldestTurn(std::vector<Message>* history, bool* removed_image) {
  if (history == nullptr || history->size() <= 1) {
    return false;
  }

  size_t erase_count = 1;
  if (history->size() >= 2 && (*history)[0].role == "user" &&
      (*history)[1].role == "assistant") {
    erase_count = 2;
  }

  if (removed_image != nullptr) {
    *removed_image = false;
    for (size_t index = 0; index < erase_count; ++index) {
      *removed_image = *removed_image || (*history)[index].has_image;
    }
  }
  history->erase(history->begin(), history->begin() + erase_count);
  return true;
}

std::string FormatWhole(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(0) << value;
  return out.str();
}

}  // namespace

Gemma4::Gemma4(const ChatPaths& paths, const ChatSettings& settings,
               ChatEventSink status, ChatEventSink debug)
    : settings_(settings),
      response_reserve_(std::min(settings.min_response_tokens,
                                 kCacheLen - 1)),
      status_(std::move(status)),
      debug_(std::move(debug)) {
  Emit("Loading vision model...");
  vision_ = std::make_unique<VisionEngine>(paths.vision_hbm);
  Emit("Vision model loaded in " + FormatWhole(vision_->LoadMs()) + " ms");

  Emit("Loading text model...");
  engine_ = std::make_unique<TextEngine>(paths.text_hbm,
                                         paths.tok_embeddings);
  Emit("Text model loaded in " + FormatWhole(engine_->LoadMs()) + " ms");
  Emit("KV cache: " + std::to_string(kCacheLen) + " tokens; max output: " +
       (settings_.max_tokens == 0
            ? std::string("auto (all remaining tokens)")
            : std::to_string(settings_.max_tokens)));

  tokenizer_ = std::make_unique<TokenizerBridge>(paths.tokenizer_json);
}

void Gemma4::Emit(const std::string& line) const {
  if (status_) {
    status_(line);
  }
}

size_t Gemma4::LoadImage(const std::string& path) {
  if (!history_.empty() || !pending_vision_features_.empty()) {
    engine_->ResetSession();
    history_.clear();
    session_ids_.clear();
    turn_count_ = 0;
    pending_vision_features_.clear();
    Emit("Starting a new conversation for the image.");
  }

  pending_vision_features_ = PredictVision(
      gemma4::LoadImage(path), [this](const std::vector<float>& patches) {
        return vision_->Infer(patches);
      });
  pending_image_ = path;
  has_pending_image_ = true;
  return pending_vision_features_.size();
}

ChatTurnResult Gemma4::predict(const std::string& text,
                               ChatEventSink on_token) {
  ChatTurnResult result;

  // Preserve the original image turn and explicitly condition the latest
  // follow-up on the same image. Older repeated placeholders are removed,
  // so a multimodal prompt contains at most two 280-token image runs.
  const bool starts_image_conversation = has_pending_image_;
  if (starts_image_conversation) {
    history_.clear();
    session_ids_.clear();
    engine_->ResetSession();
    turn_count_ = 0;
  }
  const bool uses_active_image =
      starts_image_conversation || !pending_vision_features_.empty();
  if (uses_active_image) {
    bool kept_original_image = false;
    for (auto& history_item : history_) {
      if (!history_item.has_image) {
        continue;
      }
      if (!kept_original_image) {
        kept_original_image = true;
      } else {
        history_item.has_image = false;
      }
    }
  }
  history_.push_back({"user", text, uses_active_image});

  const std::vector<int64_t> prev_ids = session_ids_;
  const int prev_processed = engine_->ProcessedTokens();

  const int desired_reserve = settings_.max_tokens == 0
      ? response_reserve_
      : std::min(settings_.max_tokens, kCacheLen - 1);
  bool history_trimmed = false;
  while (true) {
    session_ids_ = tokenizer_->EncodeMessagesJson(BuildMessagesJson(history_), true);
    if (session_ids_.size() < static_cast<size_t>(kCacheLen) &&
        (session_ids_.size() + static_cast<size_t>(desired_reserve) <=
             static_cast<size_t>(kCacheLen) ||
         history_.size() <= 1)) {
      break;
    }

    bool removed_image = false;
    if (!DropOldestTurn(&history_, &removed_image)) {
      break;
    }
    if (turn_count_ > 0) {
      --turn_count_;
    }
    history_trimmed = true;
    const bool history_still_uses_image = std::any_of(
        history_.begin(), history_.end(),
        [](const Message& history_item) { return history_item.has_image; });
    if (removed_image && !history_still_uses_image) {
      pending_vision_features_.clear();
      pending_image_.clear();
      has_pending_image_ = false;
    }
  }

  if (session_ids_.size() >= static_cast<size_t>(kCacheLen)) {
    result.prompt_too_long = true;
    result.prompt_tokens = session_ids_.size();
    history_.pop_back();
    session_ids_ = prev_ids;
    return result;
  }

  if (history_trimmed) {
    engine_->ResetSession();
    Emit("[context] Oldest chat turns were removed to stay within " +
         std::to_string(kCacheLen) + " tokens.");
  }

  const int available_tokens =
      kCacheLen - static_cast<int>(session_ids_.size());
  const int turn_max_tokens = settings_.max_tokens == 0
      ? available_tokens
      : std::min(settings_.max_tokens, available_tokens);
  result.prompt_tokens = session_ids_.size();
  result.output_budget = turn_max_tokens;
  result.history_trimmed = history_trimmed;
  Emit("[context] prompt=" + std::to_string(session_ids_.size()) +
       ", output_budget=" + std::to_string(turn_max_tokens) +
       ", capacity=" + std::to_string(kCacheLen));

  // Check if we need to reset due to prefix mismatch.
  bool prefix_ok = true;
  if (settings_.rebuild_context_each_turn) {
    engine_->ResetSession();
    prefix_ok = false;
  } else if (!history_trimmed && prev_processed > 0) {
    if (static_cast<int>(session_ids_.size()) < prev_processed ||
        static_cast<int>(prev_ids.size()) < prev_processed) {
      prefix_ok = false;
    } else {
      for (int j = 0; j < prev_processed; ++j) {
        if (session_ids_[static_cast<size_t>(j)] !=
            prev_ids[static_cast<size_t>(j)]) {
          prefix_ok = false;
          break;
        }
      }
    }
    if (!prefix_ok) {
      engine_->ResetSession();
    }
  }
  if (RuntimeDebugEnabled() && debug_) {
    std::ostringstream debug_line;
    debug_line << "[DEBUG] context reuse: prev_processed=" << prev_processed
               << " prev_ids=" << prev_ids.size()
               << " prompt_ids=" << session_ids_.size()
               << " prefix_ok=" << (prefix_ok ? "true" : "false")
               << " rebuild="
               << (settings_.rebuild_context_each_turn ? "true" : "false");
    debug_(debug_line.str());
  }

  auto t_start = std::chrono::steady_clock::now();
  int token_count = 0;

  auto stream_callback = [&](int64_t token_id) {
    if (token_id == kEosTokenId || token_id == kTurnEndTokenId) {
      return true;
    }
    const std::string token_text = tokenizer_->DecodeIds({token_id});
    if (on_token) {
      on_token(token_text);
    }
    ++token_count;
    return true;
  };

  std::vector<int64_t> out;
  if (!pending_vision_features_.empty()) {
    // Retain one image placeholder and inject the same active features when
    // rebuilding the multimodal prompt for follow-up turns.
    auto prompt_hidden = engine_->BuildPromptHidden(
        session_ids_, pending_vision_features_);
    engine_->ResetSession();
    out = engine_->ContinueGenerateStream(
        session_ids_, turn_max_tokens, stream_callback, &prompt_hidden);
    // The image is no longer pending, but retain its features so follow-up
    // turns can rebuild the multimodal prompt correctly.
    has_pending_image_ = false;
    pending_image_.clear();
  } else {
    out = engine_->ContinueGenerateStream(
        session_ids_, turn_max_tokens, stream_callback);
  }

  auto t_end = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  size_t generation_end = out.size();
  while (generation_end > session_ids_.size() &&
         (out[generation_end - 1] == kEosTokenId ||
          out[generation_end - 1] == kTurnEndTokenId)) {
    --generation_end;
  }
  const std::vector<int64_t> gen(
      out.begin() + session_ids_.size(), out.begin() + generation_end);
  const std::string reply = tokenizer_->DecodeIds(gen);
  history_.push_back({"assistant", reply, false});
  session_ids_ = out;
  ++turn_count_;

  result.generated_ids = gen;
  result.reply = reply;
  result.streamed_tokens = token_count;
  result.elapsed_ms = elapsed_ms;
  result.tokens_per_sec = (elapsed_ms > 0) ? (token_count / (elapsed_ms / 1000.0)) : 0;
  return result;
}

void Gemma4::Reset() {
  engine_->ResetSession();
  history_.clear();
  session_ids_.clear();
  turn_count_ = 0;
  has_pending_image_ = false;
  pending_image_.clear();
  pending_vision_features_.clear();
}

ChatContextUsage Gemma4::ContextUsage() const {
  ChatContextUsage usage;
  usage.used_tokens = session_ids_.size();
  usage.remaining_tokens =
      usage.used_tokens < static_cast<size_t>(kCacheLen)
          ? static_cast<size_t>(kCacheLen) - usage.used_tokens
          : 0;
  usage.turns = turn_count_;
  return usage;
}

}  // namespace gemma4
