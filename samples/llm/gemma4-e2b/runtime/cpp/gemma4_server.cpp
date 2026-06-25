#include <iostream>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "gemma4_text_engine.h"
#include "gemma4_tokenizer.h"
#include "gemma4_vision_engine.h"

namespace {

void PrintHelp() {
  std::cerr
      << "Gemma4 chat server (models loaded once, KV cache reused)\n\n"
      << "Commands:\n"
      << "  /help              Show help\n"
      << "  /reset             Clear conversation + KV cache\n"
      << "  /status            Show session stats\n"
      << "  /image PATH        Attach image for next VLM turn\n"
      << "  /quit              Exit\n\n"
      << "Type a message and press Enter to chat.\n"
      << std::endl;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

std::string BuildMessagesJson(
    const std::vector<std::pair<std::string, std::string>>& turns) {
  std::ostringstream oss;
  oss << '[';
  for (size_t i = 0; i < turns.size(); ++i) {
    if (i) {
      oss << ',';
    }
    oss << "{\"role\":\"" << turns[i].first << "\",\"content\":\""
        << JsonEscape(turns[i].second) << "\"}";
  }
  oss << ']';
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  std::string text_hbm = home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm";
  std::string vision_hbm = home + "/model/gemma4-e2b_vit_ptq.hbm";
  std::string embed = home + "/model/tok_embeddings.bin";
  int max_tokens = 128;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--text-hbm" || arg == "--vision-hbm" || arg == "--embed") &&
        i + 1 < argc) {
      const std::string val = argv[++i];
      if (arg == "--text-hbm") {
        text_hbm = val;
      } else if (arg == "--vision-hbm") {
        vision_hbm = val;
      } else if (arg == "--embed") {
        embed = val;
      }
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    }
  }

  try {
    std::cout << "Loading Text HBM (one-time, ~60s) ..." << std::endl;
    gemma4::TextEngine text(text_hbm, embed);
    std::cout << "Text ready (" << text.LoadMs() << " ms)\n";

    std::unique_ptr<gemma4::VisionEngine> vision;
    gemma4::TokenizerBridge tokenizer;

    std::vector<std::pair<std::string, std::string>> turns;
    std::optional<std::string> pending_image;
    std::vector<int64_t> session_ids;
    std::vector<float> session_hidden;

    auto build_messages_json = [&]() -> std::string {
      std::ostringstream oss;
      oss << '[';
      for (size_t i = 0; i < turns.size(); ++i) {
        if (i) {
          oss << ',';
        }
        oss << "{\"role\":\"" << turns[i].first << "\",\"content\":\""
            << JsonEscape(turns[i].second) << "\"}";
      }
      oss << ']';
      return oss.str();
    };

    auto prefix_match = [](const std::vector<int64_t>& a,
                           const std::vector<int64_t>& b, int len) {
      if (static_cast<int>(a.size()) < len || static_cast<int>(b.size()) < len) {
        return false;
      }
      for (int i = 0; i < len; ++i) {
        if (a[static_cast<size_t>(i)] != b[static_cast<size_t>(i)]) {
          return false;
        }
      }
      return true;
    };

    PrintHelp();
    std::cout << "gemma4> " << std::flush;

    std::string line;
    while (std::getline(std::cin, line)) {
      if (line == "/quit" || line == "/exit") {
        break;
      }
      if (line == "/help") {
        PrintHelp();
        std::cout << "gemma4> " << std::flush;
        continue;
      }
      if (line == "/reset") {
        text.ResetSession();
        turns.clear();
        session_ids.clear();
        session_hidden.clear();
        pending_image.reset();
        std::cout << "Session reset.\n";
        std::cout << "gemma4> " << std::flush;
        continue;
      }
      if (line == "/status") {
        std::cout << "turns=" << turns.size()
                  << " cached_tokens=" << text.ProcessedTokens()
                  << " session_ids=" << session_ids.size() << "\n";
        std::cout << "gemma4> " << std::flush;
        continue;
      }
      if (line.rfind("/image ", 0) == 0) {
        pending_image = line.substr(7);
        std::cout << "Image queued: " << *pending_image << "\n";
        std::cout << "gemma4> " << std::flush;
        continue;
      }
      if (line.empty()) {
        std::cout << "gemma4> " << std::flush;
        continue;
      }

      turns.emplace_back("user", line);

      const std::vector<int64_t> prev_ids = session_ids;
      const int prev_processed = text.ProcessedTokens();

      if (pending_image.has_value()) {
        std::ostringstream oss;
        oss << '[';
        for (size_t i = 0; i + 1 < turns.size(); ++i) {
          if (i) {
            oss << ',';
          }
          oss << "{\"role\":\"" << turns[i].first << "\",\"content\":\""
              << JsonEscape(turns[i].second) << "\"}";
        }
        if (turns.size() > 1) {
          oss << ',';
        }
        oss << "{\"role\":\"user\",\"content\":[{\"type\":\"image\"},{\"type\":\"text\",\"text\":\""
            << JsonEscape(line) << "\"}]}]";
        session_ids = tokenizer.EncodeMessagesJson(oss.str());

        if (!vision) {
          std::cout << "Loading Vision HBM ..." << std::endl;
          vision = std::make_unique<gemma4::VisionEngine>(vision_hbm);
          std::cout << "Vision ready (" << vision->LoadMs() << " ms)\n";
        }
        const auto vision_features = vision->Infer(*pending_image);
        session_hidden = text.BuildPromptHidden(session_ids, vision_features);
        pending_image.reset();
      } else {
        session_ids = tokenizer.EncodeMessagesJson(build_messages_json());
        session_hidden.clear();
      }

      if (prev_processed > 0 &&
          !prefix_match(session_ids, prev_ids, prev_processed)) {
        text.ResetSession();
      }

      const std::vector<float>* hidden_ptr =
          session_hidden.empty() ? nullptr : &session_hidden;
      const auto out = text.ContinueGenerate(session_ids, max_tokens, hidden_ptr);

      const std::vector<int64_t> gen(out.begin() + session_ids.size(), out.end());
      const std::string reply = tokenizer.DecodeIds(gen);
      turns.emplace_back("assistant", reply);
      session_ids = out;

      std::cout << "\nassistant> " << reply << "\n\n";
      std::cout << "gemma4> " << std::flush;
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
