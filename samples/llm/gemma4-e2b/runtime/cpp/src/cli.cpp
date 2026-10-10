/**
 * @file cli.cpp
 * @brief Command-line and console layer for the interactive Gemma4-E2B chat.
 *
 * Everything here is entry-shaped or presentation plumbing moved from the
 * historical application facade without string changes: gflags parsing with
 * $GEMMA4_HOME default-path resolution and setting validation, REPL command
 * classification, UTF-8/GB18030 terminal normalization (iconv), the banner
 * and help text, the REPL prompt and the result/image/context report lines.
 * No inference or session state lives in this file; the Gemma4 model owns
 * those and emits its notices through the sinks defined here.
 */

#include "cli.hpp"

#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <iconv.h>

#include "gflags/gflags.h"

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

namespace gemma4::cli {
namespace {

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

}  // namespace

bool ParseOptions(int argc, char** argv, ChatOptions* options) {
  if (options == nullptr) {
    return false;
  }
  gflags::SetUsageMessage(
      "Interactive VLM chat for Gemma4-E2B on RDK S series.\n"
      "Usage: ./main [--text_hbm PATH] [--vision_hbm PATH] "
      "[--tok_embeddings PATH] [--tokenizer_path PATH] [--max_tokens N] "
      "[--min_response_tokens N]");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_max_tokens < 0) {
    std::cerr << "--max_tokens must be zero or positive" << std::endl;
    return false;
  }
  if (FLAGS_min_response_tokens <= 0) {
    std::cerr << "--min_response_tokens must be positive" << std::endl;
    return false;
  }

  // Resolve default paths from $GEMMA4_HOME when the corresponding flag is
  // empty.
  const char* env_home = std::getenv("GEMMA4_HOME");
  const std::string home = (env_home && *env_home) ? env_home : ".";
  options->paths.text_hbm = FLAGS_text_hbm.empty()
      ? home + "/model/gemma4-e2b_lm_chunk_256_cache_4096_ptq.hbm"
      : FLAGS_text_hbm;
  options->paths.vision_hbm = FLAGS_vision_hbm.empty()
      ? home + "/model/gemma4-e2b_vit_ptq.hbm"
      : FLAGS_vision_hbm;
  options->paths.tok_embeddings = FLAGS_tok_embeddings.empty()
      ? home + "/model/tok_embeddings.bin"
      : FLAGS_tok_embeddings;
  options->paths.tokenizer_json = FLAGS_tokenizer_path.empty()
      ? home + "/tokenizer/tokenizer.json"
      : FLAGS_tokenizer_path;

  options->settings.max_tokens = FLAGS_max_tokens;
  options->settings.min_response_tokens = FLAGS_min_response_tokens;
  options->settings.rebuild_context_each_turn = FLAGS_rebuild_context_each_turn;
  // The reserve is capped by the KV capacity inside the Gemma4 model.
  return true;
}

TerminalCommand ParseCommand(const std::string& line) {
  TerminalCommand parsed;
  parsed.text = line;
  if (line == "/quit" || line == "/exit") {
    parsed.command = Command::kQuit;
  } else if (line == "/help") {
    parsed.command = Command::kHelp;
  } else if (line == "/reset") {
    parsed.command = Command::kReset;
  } else if (line == "/context") {
    parsed.command = Command::kContext;
  } else if (line.substr(0, 7) == "/image ") {
    parsed.command = Command::kImage;
    parsed.text = line.substr(7);
  } else if (line.empty()) {
    parsed.command = Command::kEmpty;
  } else {
    parsed.command = Command::kChat;
  }
  return parsed;
}

ChatEventSink StatusLineSink() {
  return [](const std::string& line) { std::cout << line << std::endl; };
}

ChatEventSink DebugLineSink() {
  return [](const std::string& line) { std::cerr << line << std::endl; };
}

ChatEventSink StreamSink() {
  return [](const std::string& fragment) {
    std::cout << fragment << std::flush;
  };
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

void PrintPrompt() { std::cout << "gemma4> " << std::flush; }

ReadStatus ReadLine(std::string* line) {
  if (line == nullptr || !std::getline(std::cin, *line)) {
    return ReadStatus::kEof;
  }
  bool converted_from_gb18030 = false;
  if (!NormalizeTerminalInput(line, &converted_from_gb18030)) {
    std::cerr << "Input error: terminal text is neither valid UTF-8 nor "
                 "GB18030. Configure the terminal for UTF-8 and retry.\n";
    return ReadStatus::kInvalid;
  }
  if (converted_from_gb18030) {
    std::cerr << "[input] Converted GB18030 terminal bytes to UTF-8.\n";
  }
  return ReadStatus::kOk;
}

bool ParseImageArgument(const std::string& argument, std::string* path) {
  // Trim whitespace
  const size_t start = argument.find_first_not_of(" \t");
  const size_t end = argument.find_last_not_of(" \t");
  if (start == std::string::npos) {
    std::cout << "Error: /image requires a file path\n";
    return false;
  }
  *path = argument.substr(start, end - start + 1);

  // Check if file exists
  std::ifstream test_file(*path);
  if (!test_file.good()) {
    std::cout << "Error: cannot open image file: " << *path << "\n";
    return false;
  }
  return true;
}

void ReportImageProcessing(const std::string& path) {
  std::cout << "Processing image: " << path << "..." << std::endl;
}

void ReportImageLoaded(size_t feature_count) {
  std::cout << "Image loaded (" << feature_count << " features).\n";
  std::cout << "Now type your question about the image.\n";
}

void ReportSessionReset() { std::cout << "Session reset.\n"; }

void ReportContext(const ChatContextUsage& usage) {
  std::cout << "Context: " << usage.used_tokens << "/" << kCacheLen
            << " tokens, remaining=" << usage.remaining_tokens
            << ", turns=" << usage.turns << std::endl;
}

void ReportTurn(const ChatTurnResult& turn) {
  if (turn.prompt_too_long) {
    std::cout << "Error: the current prompt uses " << turn.prompt_tokens
              << " tokens, exceeding the " << kCacheLen
              << "-token KV cache." << std::endl;
    return;
  }
  std::cout << "\n"
            << "[" << std::fixed << std::setprecision(1) << turn.elapsed_ms
            << " ms, " << turn.streamed_tokens << " tokens, "
            << turn.tokens_per_sec << " tok/s]\n\n";
}

}  // namespace gemma4::cli
