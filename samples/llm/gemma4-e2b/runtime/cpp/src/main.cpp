/**
 * @file main.cpp
 * @brief Interactive VLM chat entry point for Gemma4-E2B on RDK S series.
 *
 * Provides a streaming, multi-turn chat loop that supports both pure-text
 * prompts and image+text (VLM) queries. Loads pre-compiled HBM models via
 * the Text and Vision engines, reuses the KV cache across turns, and
 * tokenizes with the native C++ tokenizer (no Python).
 *
 * @note Primary executable of this Model Zoo sample; built as `main`.
 */
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <iconv.h>

#include "gflags/gflags.h"
#include "nlohmann/json.hpp"

#include "gemma4_tokenizer.hpp"
#include "gemma4_config.hpp"
#include "gemma4_text_engine.hpp"
#include "gemma4_vision_engine.hpp"

// -------------------- Command-line flags --------------------
// Empty default => resolved at runtime from $GEMMA4_HOME.
DEFINE_string(text_hbm, "",
              "Path to text LLM *.hbm. Default: $GEMMA4_HOME/model/"
              "gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm");
DEFINE_string(vision_hbm, "",
              "Path to vision ViT *.hbm. Default: $GEMMA4_HOME/model/"
              "gemma4-e2b_vit_ptq.hbm");
DEFINE_string(tok_embeddings, "",
              "Path to tok_embeddings.bin (external token embedding table). "
              "Default: $GEMMA4_HOME/model/tok_embeddings.bin");
DEFINE_string(tokenizer_path, "",
              "Path to tokenizer.json. Default: $GEMMA4_HOME/tokenizer/tokenizer.json");
DEFINE_int32(max_tokens, 0,
             "Maximum new tokens per turn. 0 uses all KV capacity remaining "
             "after the prompt.");
DEFINE_int32(min_response_tokens, 256,
             "Minimum response capacity preserved when old chat turns are trimmed.");
DEFINE_bool(rebuild_context_each_turn, false,
            "Rebuild the full prompt and KV cache before every response.");

namespace {

struct Message {
  std::string role;
  std::string content;
  bool has_image = false;
};

void PrintHelp() {
  std::cerr
      << "Gemma4 interactive chat (streaming, KV cache reuse)\n\n"
      << "Commands:\n"
      << "  /help              Show this help\n"
      << "  /reset             Clear conversation history + KV cache\n"
      << "  /context           Show KV-cache usage\n"
      << "  /image <path>      Load image for next message\n"
      << "  /quit              Exit\n\n"
      << "Type a message and press Enter to chat.\n"
      << "Use /image before typing to ask about an image.\n"
      << std::endl;
}

bool IsValidUtf8(const std::string& value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(value.data());
  size_t offset = 0;
  while (offset < value.size()) {
    const uint8_t first = bytes[offset];
    if (first <= 0x7f) {
      ++offset;
      continue;
    }

    size_t length = 0;
    uint32_t code_point = 0;
    uint32_t minimum = 0;
    if ((first & 0xe0) == 0xc0) {
      length = 2;
      code_point = first & 0x1f;
      minimum = 0x80;
    } else if ((first & 0xf0) == 0xe0) {
      length = 3;
      code_point = first & 0x0f;
      minimum = 0x800;
    } else if ((first & 0xf8) == 0xf0) {
      length = 4;
      code_point = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (offset + length > value.size()) {
      return false;
    }
    for (size_t index = 1; index < length; ++index) {
      const uint8_t continuation = bytes[offset + index];
      if ((continuation & 0xc0) != 0x80) {
        return false;
      }
      code_point = (code_point << 6) | (continuation & 0x3f);
    }
    if (code_point < minimum || code_point > 0x10ffff ||
        (code_point >= 0xd800 && code_point <= 0xdfff)) {
      return false;
    }
    offset += length;
  }
  return true;
}

bool ConvertGb18030ToUtf8(const std::string& input, std::string* output) {
  if (output == nullptr) {
    return false;
  }
  iconv_t converter = iconv_open("UTF-8", "GB18030");
  if (converter == reinterpret_cast<iconv_t>(-1)) {
    return false;
  }

  char* input_data = const_cast<char*>(input.data());
  size_t input_remaining = input.size();
  std::string converted(std::max<size_t>(64, input.size() * 2 + 16), '\0');
  size_t output_used = 0;
  while (true) {
    char* output_data = converted.data() + output_used;
    size_t output_remaining = converted.size() - output_used;
    const size_t result = iconv(converter, &input_data, &input_remaining,
                                &output_data, &output_remaining);
    output_used = converted.size() - output_remaining;
    if (result != static_cast<size_t>(-1)) {
      break;
    }
    if (errno != E2BIG) {
      iconv_close(converter);
      return false;
    }
    converted.resize(converted.size() * 2);
  }
  iconv_close(converter);
  converted.resize(output_used);
  if (input_remaining != 0 || !IsValidUtf8(converted)) {
    return false;
  }
  *output = std::move(converted);
  return true;
}

bool NormalizeTerminalInput(std::string* input, bool* converted_from_gb18030) {
  if (input == nullptr) {
    return false;
  }
  if (!input->empty() && input->back() == '\r') {
    input->pop_back();
  }
  if (converted_from_gb18030 != nullptr) {
    *converted_from_gb18030 = false;
  }
  if (IsValidUtf8(*input)) {
    return true;
  }

  std::string converted;
  if (!ConvertGb18030ToUtf8(*input, &converted)) {
    return false;
  }
  *input = std::move(converted);
  if (converted_from_gb18030 != nullptr) {
    *converted_from_gb18030 = true;
  }
  return true;
}

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

void PrintBanner() {
  const char* RST = "\033[0m";
  const char* BLD = "\033[1m";
  const char* DIM = "\033[2m";
  const char* c[] = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m",
    "\033[38;5;214m", "\033[38;5;220m", "\033[38;5;226m",
    "\033[38;5;46m",  "\033[38;5;51m",  "\033[38;5;39m",
    "\033[38;5;33m",  "\033[38;5;99m",  "\033[38;5;201m",
  };

#if defined(SOC_S600)
  const char* title = "        Gemma on RDK S600";
#elif defined(SOC_S100P)
  const char* title = "        Gemma on RDK S100P";
#elif defined(SOC_S100)
  const char* title = "        Gemma on RDK S100";
#else
  const char* title = "        Gemma on RDK S Series";
#endif

  std::cout << "\n" << DIM
            << "================================================================\n" << RST
            << "\n" << BLD;
  int ci = 0;
  for (int i = 0; title[i]; ++i) {
    if (title[i] != ' ') {
      std::cout << c[ci % 12] << title[i];
      ++ci;
    } else {
      std::cout << title[i];
    }
  }
  std::cout << RST << "\n\n" << DIM
            << "            Vision-Language Model | D-Robotics\n"
            << "================================================================\n" << RST
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  // Parse gflags first (consumes recognized --flag args, leaves the rest).
  gflags::SetUsageMessage(
      "Interactive VLM chat for Gemma4-E2B on RDK S series.\n"
      "Usage: ./main [--text_hbm PATH] [--vision_hbm PATH] "
      "[--tok_embeddings PATH] [--tokenizer_path PATH] [--max_tokens N] "
      "[--min_response_tokens N]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Resolve default paths from $GEMMA4_HOME when the corresponding flag is empty.
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  const std::string text_hbm = FLAGS_text_hbm.empty()
      ? home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm"
      : FLAGS_text_hbm;
  const std::string vision_hbm = FLAGS_vision_hbm.empty()
      ? home + "/model/gemma4-e2b_vit_ptq.hbm"
      : FLAGS_vision_hbm;
  const std::string embed = FLAGS_tok_embeddings.empty()
      ? home + "/model/tok_embeddings.bin"
      : FLAGS_tok_embeddings;
  const std::string tokenizer_json = FLAGS_tokenizer_path.empty()
      ? home + "/tokenizer/tokenizer.json"
      : FLAGS_tokenizer_path;

  if (FLAGS_max_tokens < 0) {
    std::cerr << "--max_tokens must be zero or positive" << std::endl;
    return 2;
  }
  if (FLAGS_min_response_tokens <= 0) {
    std::cerr << "--min_response_tokens must be positive" << std::endl;
    return 2;
  }
  const int requested_max_tokens = FLAGS_max_tokens;
  const int response_reserve = std::min(
      FLAGS_min_response_tokens, gemma4::kCacheLen - 1);

  try {
    PrintBanner();

    std::cout << "Loading vision model..." << std::endl;
    gemma4::VisionEngine vision(vision_hbm);
    std::cout << "Vision model loaded in " << std::fixed << std::setprecision(0)
              << vision.LoadMs() << " ms\n";

    std::cout << "Loading text model..." << std::endl;
    gemma4::TextEngine engine(text_hbm, embed);
    std::cout << "Text model loaded in " << std::fixed << std::setprecision(0)
              << engine.LoadMs() << " ms\n";
    std::cout << "KV cache: " << gemma4::kCacheLen
              << " tokens; max output: "
              << (requested_max_tokens == 0
                      ? std::string("auto (all remaining tokens)")
                      : std::to_string(requested_max_tokens))
              << std::endl;

    gemma4::TokenizerBridge tokenizer(tokenizer_json);

    std::vector<Message> history;
    std::vector<int64_t> session_ids;
    int turn_count = 0;

    // One active image is supported per conversation. Its compact vision
    // features are retained so follow-up turns can still reference the image.
    std::string pending_image;
    std::vector<float> pending_vision_features;
    bool has_pending_image = false;

    PrintHelp();
    std::cout << "gemma4> " << std::flush;

    std::string line;
    while (std::getline(std::cin, line)) {
      bool converted_from_gb18030 = false;
      if (!NormalizeTerminalInput(&line, &converted_from_gb18030)) {
        std::cerr << "Input error: terminal text is neither valid UTF-8 nor "
                     "GB18030. Configure the terminal for UTF-8 and retry.\n";
        std::cout << "gemma4> " << std::flush;
        continue;
      }
      if (converted_from_gb18030) {
        std::cerr << "[input] Converted GB18030 terminal bytes to UTF-8.\n";
      }
      if (line == "/quit" || line == "/exit") break;

      if (line == "/help") {
        PrintHelp();
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      if (line == "/reset") {
        engine.ResetSession();
        history.clear();
        session_ids.clear();
        turn_count = 0;
        has_pending_image = false;
        pending_image.clear();
        pending_vision_features.clear();
        std::cout << "Session reset.\n";
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      if (line == "/context") {
        const size_t used_tokens = session_ids.size();
        const size_t remaining_tokens = used_tokens < gemma4::kCacheLen
            ? static_cast<size_t>(gemma4::kCacheLen) - used_tokens
            : 0;
        std::cout << "Context: " << used_tokens << "/" << gemma4::kCacheLen
                  << " tokens, remaining=" << remaining_tokens
                  << ", turns=" << turn_count << std::endl;
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      // Handle /image command
      if (line.substr(0, 7) == "/image ") {
        std::string img_path = line.substr(7);
        // Trim whitespace
        size_t start = img_path.find_first_not_of(" \t");
        size_t end = img_path.find_last_not_of(" \t");
        if (start == std::string::npos) {
          std::cout << "Error: /image requires a file path\n";
          std::cout << "gemma4> " << std::flush;
          continue;
        }
        img_path = img_path.substr(start, end - start + 1);

        // Check if file exists
        std::ifstream test_file(img_path);
        if (!test_file.good()) {
          std::cout << "Error: cannot open image file: " << img_path << "\n";
          std::cout << "gemma4> " << std::flush;
          continue;
        }

        if (!history.empty() || !pending_vision_features.empty()) {
          engine.ResetSession();
          history.clear();
          session_ids.clear();
          turn_count = 0;
          pending_vision_features.clear();
          std::cout << "Starting a new conversation for the image."
                    << std::endl;
        }

        std::cout << "Processing image: " << img_path << "..." << std::endl;
        pending_vision_features = vision.Infer(img_path);
        pending_image = img_path;
        has_pending_image = true;
        std::cout << "Image loaded (" << pending_vision_features.size() << " features).\n";
        std::cout << "Now type your question about the image.\n";
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      if (line.empty()) {
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      // Preserve the original image turn and explicitly condition the latest
      // follow-up on the same image. Older repeated placeholders are removed,
      // so a multimodal prompt contains at most two 280-token image runs.
      const bool starts_image_conversation = has_pending_image;
      if (starts_image_conversation) {
        history.clear();
        session_ids.clear();
        engine.ResetSession();
        turn_count = 0;
      }
      const bool uses_active_image =
          starts_image_conversation || !pending_vision_features.empty();
      if (uses_active_image) {
        bool kept_original_image = false;
        for (auto& history_item : history) {
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
      history.push_back({"user", line, uses_active_image});

      const std::vector<int64_t> prev_ids = session_ids;
      const int prev_processed = engine.ProcessedTokens();

      const int desired_reserve = requested_max_tokens == 0
          ? response_reserve
          : std::min(requested_max_tokens, gemma4::kCacheLen - 1);
      bool history_trimmed = false;
      while (true) {
        session_ids = tokenizer.EncodeMessagesJson(BuildMessagesJson(history), true);
        if (session_ids.size() < static_cast<size_t>(gemma4::kCacheLen) &&
            (session_ids.size() + static_cast<size_t>(desired_reserve) <=
                 static_cast<size_t>(gemma4::kCacheLen) ||
             history.size() <= 1)) {
          break;
        }

        bool removed_image = false;
        if (!DropOldestTurn(&history, &removed_image)) {
          break;
        }
        if (turn_count > 0) {
          --turn_count;
        }
        history_trimmed = true;
        const bool history_still_uses_image = std::any_of(
            history.begin(), history.end(),
            [](const Message& history_item) { return history_item.has_image; });
        if (removed_image && !history_still_uses_image) {
          pending_vision_features.clear();
          pending_image.clear();
          has_pending_image = false;
        }
      }

      if (session_ids.size() >= static_cast<size_t>(gemma4::kCacheLen)) {
        std::cout << "Error: the current prompt uses " << session_ids.size()
                  << " tokens, exceeding the " << gemma4::kCacheLen
                  << "-token KV cache." << std::endl;
        history.pop_back();
        session_ids = prev_ids;
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      if (history_trimmed) {
        engine.ResetSession();
        std::cout << "[context] Oldest chat turns were removed to stay within "
                  << gemma4::kCacheLen << " tokens." << std::endl;
      }

      const int available_tokens =
          gemma4::kCacheLen - static_cast<int>(session_ids.size());
      const int turn_max_tokens = requested_max_tokens == 0
          ? available_tokens
          : std::min(requested_max_tokens, available_tokens);
      std::cout << "[context] prompt=" << session_ids.size()
                << ", output_budget=" << turn_max_tokens
                << ", capacity=" << gemma4::kCacheLen << std::endl;

      // Check if we need to reset due to prefix mismatch.
      bool prefix_ok = true;
      if (FLAGS_rebuild_context_each_turn) {
        engine.ResetSession();
        prefix_ok = false;
      } else if (!history_trimmed && prev_processed > 0) {
        if (static_cast<int>(session_ids.size()) < prev_processed ||
            static_cast<int>(prev_ids.size()) < prev_processed) {
          prefix_ok = false;
        } else {
          for (int j = 0; j < prev_processed; ++j) {
            if (session_ids[static_cast<size_t>(j)] !=
                prev_ids[static_cast<size_t>(j)]) {
              prefix_ok = false;
              break;
            }
          }
        }
        if (!prefix_ok) {
          engine.ResetSession();
        }
      }
      if (gemma4::RuntimeDebugEnabled()) {
        std::cerr << "[DEBUG] context reuse: prev_processed=" << prev_processed
                  << " prev_ids=" << prev_ids.size()
                  << " prompt_ids=" << session_ids.size()
                  << " prefix_ok=" << (prefix_ok ? "true" : "false")
                  << " rebuild="
                  << (FLAGS_rebuild_context_each_turn ? "true" : "false")
                  << std::endl;
      }

      gemma4::TextEngine& text_engine = engine;
      auto t_start = std::chrono::steady_clock::now();
      int token_count = 0;

      auto stream_callback = [&](int64_t token_id) {
        if (token_id == gemma4::kEosTokenId ||
            token_id == gemma4::kTurnEndTokenId) {
          return true;
        }
        const std::string token_text = tokenizer.DecodeIds({token_id});
        std::cout << token_text << std::flush;
        ++token_count;
        return true;
      };

      std::vector<int64_t> out;
      if (!pending_vision_features.empty()) {
        // Retain one image placeholder and inject the same active features when
        // rebuilding the multimodal prompt for follow-up turns.
        auto prompt_hidden = text_engine.BuildPromptHidden(
            session_ids, pending_vision_features);
        text_engine.ResetSession();
        out = text_engine.ContinueGenerateStream(
            session_ids, turn_max_tokens, stream_callback, &prompt_hidden);
        // The image is no longer pending, but retain its features so follow-up
        // turns can rebuild the multimodal prompt correctly.
        has_pending_image = false;
        pending_image.clear();
      } else {
        out = text_engine.ContinueGenerateStream(
            session_ids, turn_max_tokens, stream_callback);
      }

      auto t_end = std::chrono::steady_clock::now();
      double elapsed_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();

      size_t generation_end = out.size();
      while (generation_end > session_ids.size() &&
             (out[generation_end - 1] == gemma4::kEosTokenId ||
              out[generation_end - 1] == gemma4::kTurnEndTokenId)) {
        --generation_end;
      }
      const std::vector<int64_t> gen(
          out.begin() + session_ids.size(), out.begin() + generation_end);
      const std::string reply = tokenizer.DecodeIds(gen);
      history.push_back({"assistant", reply, false});
      session_ids = out;
      ++turn_count;

      double tps = (elapsed_ms > 0) ? (token_count / (elapsed_ms / 1000.0)) : 0;

      std::cout << "\n"
                << "[" << std::fixed << std::setprecision(1) << elapsed_ms
                << " ms, " << token_count << " tokens, "
                << std::setprecision(1) << tps << " tok/s]\n\n";
      std::cout << "gemma4> " << std::flush;
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
