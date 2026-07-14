#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "gemma4_config.hpp"
#include "gemma4_text_engine.hpp"
#include "gemma4_tokenizer.hpp"
#include "gemma4_vision_engine.hpp"

namespace {

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " text --prompt \"...\" [options]\n"
      << "  " << prog << " vlm --image_path PATH --prompt \"...\" [options]\n\n"
      << "Options:\n"
      << "  --text_hbm PATH       Text HBM path\n"
      << "  --vision_hbm PATH     Vision HBM path\n"
      << "  --tok_embeddings PATH tok_embeddings.bin path\n"
      << "  --max_tokens N        new tokens to generate (default: 32)\n"
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

// -------------------- Command-line flags --------------------
DEFINE_string(text_hbm, "",
              "Path to text LLM *.hbm. Default: $GEMMA4_HOME/model/"
              "gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm");
DEFINE_string(vision_hbm, "",
              "Path to vision ViT *.hbm. Default: $GEMMA4_HOME/model/"
              "gemma4-e2b_vit_ptq.hbm");
DEFINE_string(tok_embeddings, "",
              "Path to tok_embeddings.bin. Default: $GEMMA4_HOME/model/tok_embeddings.bin");
DEFINE_string(prompt, "", "User prompt text (required).");
DEFINE_string(image_path, "", "Image path (required when mode is 'vlm').");
DEFINE_int32(max_tokens, 32, "Maximum new tokens to generate.");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Gemma4-E2B single-shot demo.\n"
      "Usage:\n"
      "  ./gemma4_demo text --prompt \"...\" [--text_hbm PATH] [--tok_embeddings PATH] [--max_tokens N]\n"
      "  ./gemma4_demo vlm  --image_path PATH --prompt \"...\" [--vision_hbm PATH] [--text_hbm PATH] [--max_tokens N]");
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string mode = argv[1];
  // Strip mode from argv before gflags parses the rest.
  argv[1] = argv[0];
  ++argv;
  --argc;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

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
  const std::string& prompt = FLAGS_prompt;
  const std::string& image_path = FLAGS_image_path;
  const int max_tokens = FLAGS_max_tokens;

  if (prompt.empty()) {
    std::cerr << "ERROR: --prompt is required\n";
    return 1;
  }
  if (mode == "vlm" && image_path.empty()) {
    std::cerr << "ERROR: --image_path is required for vlm mode\n";
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
