#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gemma4_tokenizer.h"
#include "gemma4_config.h"
#include "gemma4_text_engine.h"
#include "gemma4_vision_engine.h"

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
      << "  /image <path>      Load image for next message\n"
      << "  /quit              Exit\n\n"
      << "Type a message and press Enter to chat.\n"
      << "Use /image before typing to ask about an image.\n"
      << std::endl;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\': out += "\\\\"; break;
      case '"':  out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:   out += c; break;
    }
  }
  return out;
}

std::string BuildMessagesJson(const std::vector<Message>& history) {
  std::ostringstream oss;
  oss << '[';
  for (size_t i = 0; i < history.size(); ++i) {
    if (i) oss << ',';
    if (history[i].has_image) {
      // VLM format: content is array with image + text
      oss << "{\"role\":\"" << history[i].role
          << "\",\"content\":[{\"type\":\"image\"},{\"type\":\"text\",\"text\":\""
          << JsonEscape(history[i].content) << "\"}]}";
    } else {
      oss << "{\"role\":\"" << history[i].role
          << "\",\"content\":\"" << JsonEscape(history[i].content) << "\"}";
    }
  }
  oss << ']';
  return oss.str();
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

  const char* title = "        Gemma on RDK S100";

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
  // Default model paths: $GEMMA4_HOME/model/ and $GEMMA4_HOME/tokenizer/
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  std::string text_hbm = home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm";
  std::string vision_hbm = home + "/model/gemma4-e2b_vit_ptq.hbm";
  std::string embed = home + "/model/tok_embeddings.bin";
  std::string tokenizer_json = home + "/tokenizer/tokenizer.json";
  int max_tokens = gemma4::kCacheLen;  // bounded by KV cache capacity

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--text-hbm" || arg == "--vision-hbm" || arg == "--embed" ||
         arg == "--tokenizer") &&
        i + 1 < argc) {
      const std::string val = argv[++i];
      if (arg == "--text-hbm")      text_hbm = val;
      else if (arg == "--vision-hbm") vision_hbm = val;
      else if (arg == "--embed")    embed = val;
      else if (arg == "--tokenizer") tokenizer_json = val;
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintHelp();
      return 0;
    }
  }

  try {
    PrintBanner();

    std::cout << "Loading text model..." << std::endl;
    gemma4::TextEngine engine(text_hbm, embed);
    std::cout << "Text model loaded in " << std::fixed << std::setprecision(0)
              << engine.LoadMs() << " ms\n";

    std::unique_ptr<gemma4::VisionEngine> vision;
    std::cout << "Loading vision model..." << std::endl;
    vision = std::make_unique<gemma4::VisionEngine>(vision_hbm);
    std::cout << "Vision model loaded in " << std::fixed << std::setprecision(0)
              << vision->LoadMs() << " ms\n";

    gemma4::TokenizerBridge tokenizer;

    std::vector<Message> history;
    std::vector<int64_t> session_ids;
    int turn_count = 0;

    // Pending image path for next message
    std::string pending_image;
    std::vector<float> pending_vision_features;
    bool has_pending_image = false;

    PrintHelp();
    std::cout << "gemma4> " << std::flush;

    std::string line;
    while (std::getline(std::cin, line)) {
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

        std::cout << "Processing image: " << img_path << "..." << std::endl;
        pending_vision_features = vision->Infer(img_path);
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

      // Add message to history
      Message msg{"user", line, has_pending_image};
      history.push_back(msg);

      const std::vector<int64_t> prev_ids = session_ids;
      const int prev_processed = engine.ProcessedTokens();

      const std::string messages_json = BuildMessagesJson(history);
      session_ids = tokenizer.EncodeMessagesJson(messages_json, true);

      // Check if we need to reset due to prefix mismatch
      bool prefix_ok = true;
      if (prev_processed > 0) {
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

      auto t_start = std::chrono::steady_clock::now();
      int token_count = 0;

      auto stream_callback = [&](int64_t token_id) {
        const std::string token_text = tokenizer.DecodeIds({token_id});
        std::cout << token_text << std::flush;
        ++token_count;
        return true;
      };

      std::vector<int64_t> out;
      if (has_pending_image && !pending_vision_features.empty()) {
        // Build prompt hidden with vision features.
        // IMPORTANT: Reset history and session first. The current architecture
        // only supports a single image per conversation (pending_vision_features
        // holds features for one image). If old history contains image tokens
        // from previous turns, BuildPromptHidden would find more image token
        // placeholders than features available, causing incorrect injection.
        history.clear();
        session_ids.clear();
        turn_count = 0;
        // Re-add only the current user message with image
        history.push_back(msg);
        const std::string fresh_json = BuildMessagesJson(history);
        session_ids = tokenizer.EncodeMessagesJson(fresh_json, true);

        auto prompt_hidden = engine.BuildPromptHidden(session_ids, pending_vision_features);
        engine.ResetSession();
        out = engine.ContinueGenerateStream(session_ids, max_tokens, stream_callback, &prompt_hidden);
        // Clear pending image after use
        has_pending_image = false;
        pending_image.clear();
        pending_vision_features.clear();
      } else {
        out = engine.ContinueGenerateStream(session_ids, max_tokens, stream_callback);
      }

      auto t_end = std::chrono::steady_clock::now();
      double elapsed_ms =
          std::chrono::duration<double, std::milli>(t_end - t_start).count();

      const std::vector<int64_t> gen(out.begin() + session_ids.size(), out.end());
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
