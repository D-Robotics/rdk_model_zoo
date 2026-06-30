#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gflags/gflags.h"

#include "gemma4_config.hpp"
#include "gemma4_text_engine.hpp"

namespace {

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " bench --text_hbm PATH --tok_embeddings PATH [options]\n"
      << "  " << prog << " generate --text_hbm PATH --tok_embeddings PATH [options]\n\n"
      << "Options:\n"
      << "  --token_ids 9259,1234   prompt token ids (default: 9259 = \"Hello\")\n"
      << "  --max_tokens N          new tokens to generate (default: 8)\n"
      << "  --warmup N              decode warmup steps before timing (default: 2)\n"
      << std::endl;
}

std::vector<int64_t> ParseTokenIds(const std::string& s) {
  std::vector<int64_t> ids;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      ids.push_back(std::stoll(item));
    }
  }
  return ids;
}

}  // namespace

// -------------------- Command-line flags --------------------
DEFINE_string(text_hbm, "",
              "Path to text LLM *.hbm. Default: $GEMMA4_HOME/model/"
              "gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm");
DEFINE_string(tok_embeddings, "",
              "Path to tok_embeddings.bin. Default: $GEMMA4_HOME/model/tok_embeddings.bin");
DEFINE_string(token_ids, "9259",
              "Prompt token ids, comma-separated (default: 9259 = \"Hello\").");
DEFINE_int32(max_tokens, 8, "Maximum new tokens to generate.");
DEFINE_int32(warmup, 2, "Decode warmup steps before timing.");

int main(int argc, char** argv) {
  gflags::SetUsageMessage(
      "Gemma4-E2B text-only benchmark / generate.\n"
      "Usage: ./gemma4_text_bench {bench|generate} [--text_hbm PATH] "
      "[--tok_embeddings PATH] [--token_ids 1,2,3] [--max_tokens N] [--warmup N]");

  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string mode = argv[1];
  // Strip mode before gflags parses the rest.
  argv[1] = argv[0];
  ++argv;
  --argc;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  const std::string hbm = FLAGS_text_hbm.empty()
      ? home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm"
      : FLAGS_text_hbm;
  const std::string embed = FLAGS_tok_embeddings.empty()
      ? home + "/model/tok_embeddings.bin"
      : FLAGS_tok_embeddings;
  const std::string& token_ids_str = FLAGS_token_ids;
  const int max_tokens = FLAGS_max_tokens;
  const int warmup = FLAGS_warmup;

  try {
    const std::vector<int64_t> ids = ParseTokenIds(token_ids_str);

    std::cout << "Loading Text HBM (one-time) ..." << std::endl;
    gemma4::TextEngine engine(hbm, embed);
    std::cout << "Model load: " << engine.LoadMs() << " ms" << std::endl;
    std::cout << "Prompt token ids (" << ids.size() << "): ";
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i) {
        std::cout << ',';
      }
      std::cout << ids[i];
    }
    std::cout << std::endl;

    if (mode == "generate") {
      const auto out = engine.Generate(ids, max_tokens);
      std::cout << "Generated ids (" << out.size() - ids.size() << "): ";
      for (size_t i = ids.size(); i < out.size(); ++i) {
        if (i > ids.size()) {
          std::cout << ',';
        }
        std::cout << out[i];
      }
      std::cout << std::endl;
      return 0;
    }

    if (mode == "bench") {
      const auto result = engine.Benchmark(ids, max_tokens, warmup);
      std::cout << "\n=== C++ hbDNN benchmark (warm model) ===" << std::endl;
      std::cout << "Load (once):     " << result.load_ms << " ms" << std::endl;
      std::cout << "Prefill:         " << result.prefill_ms << " ms" << std::endl;
      std::cout << "Decode (" << result.decode_steps << " tok): "
                << result.decode_ms << " ms" << std::endl;
      if (result.decode_steps > 0) {
        std::cout << "Decode tok/s:    " << result.tokens_per_sec << std::endl;
        std::cout << "Decode ms/tok:   "
                  << result.decode_ms / result.decode_steps << std::endl;
      }
      return 0;
    }

    PrintUsage(argv[0]);
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
}
