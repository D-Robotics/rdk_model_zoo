#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "gemma4_config.h"
#include "gemma4_text_engine.h"
#include "gemma4_tokenizer.h"
#include "gemma4_tokenizer.h"
#include "gemma4_vision_engine.h"

namespace {

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " text --prompt \"...\" [options]\n"
      << "  " << prog << " vlm --image PATH --prompt \"...\" [options]\n\n"
      << "Options:\n"
      << "  --text-hbm PATH     Text HBM path\n"
      << "  --vision-hbm PATH   Vision HBM path\n"
      << "  --embed PATH        tok_embeddings.bin path\n"
      << "  --max-tokens N      new tokens to generate (default: 32)\n"
      << std::endl;
}

std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
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

std::string BuildTextMessagesJson(const std::string& prompt) {
  return "[{\"role\":\"user\",\"content\":\"" + JsonEscape(prompt) + "\"}]";
}

std::string BuildVlmMessagesJson(const std::string& prompt) {
  return "[{\"role\":\"user\",\"content\":[{\"type\":\"image\"},{\"type\":\"text\",\"text\":\""
         + JsonEscape(prompt) + "\"}]}]";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string mode = argv[1];
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  std::string text_hbm = home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm";
  std::string vision_hbm = home + "/model/gemma4-e2b_vit_ptq.hbm";
  std::string embed = home + "/model/tok_embeddings.bin";
  std::string prompt;
  std::string image_path;
  int max_tokens = 32;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--text-hbm" || arg == "--vision-hbm" || arg == "--embed" ||
         arg == "--prompt" || arg == "--image") &&
        i + 1 < argc) {
      const std::string val = argv[++i];
      if (arg == "--text-hbm") {
        text_hbm = val;
      } else if (arg == "--vision-hbm") {
        vision_hbm = val;
      } else if (arg == "--embed") {
        embed = val;
      } else if (arg == "--prompt") {
        prompt = val;
      } else if (arg == "--image") {
        image_path = val;
      }
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  if (prompt.empty()) {
    std::cerr << "ERROR: --prompt is required\n";
    return 1;
  }
  if (mode == "vlm" && image_path.empty()) {
    std::cerr << "ERROR: --image is required for vlm mode\n";
    return 1;
  }

  try {
    gemma4::TokenizerBridge tokenizer;

    if (mode == "text") {
      const std::string messages_json = BuildTextMessagesJson(prompt);
      const auto prompt_ids = tokenizer.EncodeMessagesJson(messages_json);

      std::cout << "Tokenizing prompt (" << prompt_ids.size() << " tokens)..."
                << std::endl;
      std::cout << "Loading Text HBM (one-time) ..." << std::endl;
      gemma4::TextEngine engine(text_hbm, embed);
      std::cout << "Model load: " << engine.LoadMs() << " ms" << std::endl;

      std::cout << "\n=== Generated ===\n";
      
      // Streaming callback: decode and print each token immediately using C++ tokenizer
      auto stream_callback = [&](int64_t token_id) {
        const std::string token_text = tokenizer.DecodeIds({token_id});
        std::cout << token_text << std::flush;
        return true;  // continue generating
      };

      engine.GenerateStream(prompt_ids, max_tokens, stream_callback);
      std::cout << std::endl;
      return 0;
    }

    if (mode == "vlm") {
      const std::string messages_json = BuildVlmMessagesJson(prompt);
      const auto prompt_ids = tokenizer.EncodeMessagesJson(messages_json);

      std::cout << "Tokenizing VLM prompt (" << prompt_ids.size()
                << " tokens)..." << std::endl;
      std::cout << "Loading Vision HBM ..." << std::endl;
      gemma4::VisionEngine vision(vision_hbm);
      std::cout << "Vision load: " << vision.LoadMs() << " ms" << std::endl;

      std::cout << "Running vision infer on " << image_path << " ..."
                << std::endl;
      const auto vision_features = vision.Infer(image_path);
      std::cout << "Vision output: " << vision_features.size() << " floats"
                << std::endl;

      std::cout << "Loading Text HBM (one-time) ..." << std::endl;
      gemma4::TextEngine engine(text_hbm, embed);
      std::cout << "Text load: " << engine.LoadMs() << " ms" << std::endl;

      const auto hidden =
          engine.BuildPromptHidden(prompt_ids, vision_features);
      const auto out =
          engine.GenerateWithPromptEmbeddings(prompt_ids, hidden, max_tokens);
      const std::vector<int64_t> gen(out.begin() + prompt_ids.size(), out.end());
      const std::string text = tokenizer.DecodeIds(gen);
      std::cout << "\n=== Generated ===\n" << text << std::endl;
      return 0;
    }

    PrintUsage(argv[0]);
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
