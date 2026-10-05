/**
 * @file main.cpp
 * @brief Entry point for the interactive Gemma4-E2B VLM chat.
 *
 * This file stays deliberately thin: parse the gflags command line, resolve
 * default paths from $GEMMA4_HOME, validate the generation settings, then
 * construct the application session and run it. All console interaction and
 * conversation bookkeeping live in gemma4_chat_app.hpp/.cpp
 * (InteractiveChatApp); all model execution lives in the real engines
 * (TextEngine / VisionEngine / TokenizerBridge).
 *
 * @note Primary executable of this Model Zoo sample; built as `main`.
 */
#include <cstdlib>
#include <iostream>
#include <string>

#include "gflags/gflags.h"

#include "gemma4_chat_app.hpp"
#include "gemma4_config.hpp"

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
  gemma4::chat::ChatPaths paths;
  paths.text_hbm = FLAGS_text_hbm.empty()
      ? home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm"
      : FLAGS_text_hbm;
  paths.vision_hbm = FLAGS_vision_hbm.empty()
      ? home + "/model/gemma4-e2b_vit_ptq.hbm"
      : FLAGS_vision_hbm;
  paths.tok_embeddings = FLAGS_tok_embeddings.empty()
      ? home + "/model/tok_embeddings.bin"
      : FLAGS_tok_embeddings;
  paths.tokenizer_json = FLAGS_tokenizer_path.empty()
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

  gemma4::chat::ChatSettings settings;
  settings.max_tokens = FLAGS_max_tokens;
  settings.min_response_tokens = FLAGS_min_response_tokens;
  settings.rebuild_context_each_turn = FLAGS_rebuild_context_each_turn;
  // The reserve is capped by the KV capacity inside InteractiveChatApp.

  try {
    gemma4::chat::InteractiveChatApp app(paths, settings);
    return app.Run();
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
