/** @file cli.cpp
 * @brief gflags parsing and machine-readable RESULT emission.
 */
#include "cli.hpp"

#include <iostream>

#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

DEFINE_string(model_path, "../../model/s600", "Directory containing the verified S600 model files");
DEFINE_string(prompt, "What is 1+1? Give a short answer.", "First UTF-8 user prompt");
DEFINE_string(follow_up, "", "Optional second prompt using the same conversation");
DEFINE_int32(max_new_tokens, 128, "Maximum generated tokens per request, from 1 to 4096");

namespace minicpm5 {

bool parse_cli(int argc, char* argv[], CliOptions& options) {
  gflags::SetUsageMessage("MiniCPM5-2B text generation on RDK S600");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::cerr << "Unexpected positional argument; use --prompt or --follow_up\n";
    return false;
  }
  options.model_path = FLAGS_model_path;
  options.prompt = FLAGS_prompt;
  options.follow_up = FLAGS_follow_up;
  options.max_new_tokens = FLAGS_max_new_tokens;
  return true;
}

void emit_result(int turn, const Result& result) {
  const nlohmann::json output = {
      {"turn", turn},          {"text", result.text},
      {"token_ids", result.tokens}, {"status", result.status},
      {"ttft_ms", result.ttft_ms},  {"decode_tps", result.decode_tps},
      {"e2e_ms", result.e2e_ms}};
  std::cout << "RESULT " << output.dump() << std::endl;
}

int fail(const std::exception& error) {
  std::cerr << "ERROR: " << error.what() << '\n';
  return 1;
}
}  // namespace minicpm5
