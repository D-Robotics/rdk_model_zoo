#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gemma4_config.h"
#include "gemma4_text_engine.h"

namespace {

void PrintUsage(const char* prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " bench --hbm PATH --embed PATH [options]\n"
      << "  " << prog << " generate --hbm PATH --embed PATH [options]\n\n"
      << "Options:\n"
      << "  --token-ids 9259,1234   prompt token ids (default: 9259 = \"Hello\")\n"
      << "  --max-tokens N          new tokens to generate (default: 8)\n"
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

int main(int argc, char** argv) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  const std::string mode = argv[1];
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  std::string hbm = home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm";
  std::string embed = home + "/model/tok_embeddings.bin";
  std::string token_ids_str = "9259";
  int max_tokens = 8;
  int warmup = 2;

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--hbm" || arg == "--embed" || arg == "--token-ids") &&
        i + 1 < argc) {
      const std::string val = argv[++i];
      if (arg == "--hbm") {
        hbm = val;
      } else if (arg == "--embed") {
        embed = val;
      } else if (arg == "--token-ids") {
        token_ids_str = val;
      }
    } else if (arg == "--max-tokens" && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }
  }

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
