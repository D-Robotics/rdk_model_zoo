/** @file main.cc
 * @brief Run one MiniCPM5 prompt and an optional follow-up on RDK S600.
 */
#include <iostream>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>
#include "minicpm5.hpp"

DEFINE_string(model_path, "../../model/s600", "Directory containing the verified S600 model files");
DEFINE_string(prompt, "What is 1+1? Give a short answer.", "First UTF-8 user prompt");
DEFINE_string(follow_up, "", "Optional second prompt using the same conversation");
DEFINE_int32(max_new_tokens, 128, "Maximum generated tokens per request, from 1 to 4096");

/** @brief Parse CLI options, generate text and emit machine-readable results.
 * @param argc Argument count.
 * @param argv Argument values.
 * @return Zero for completed requests (including configured limits), nonzero for errors.
 */
int main(int argc, char** argv) {
  gflags::SetUsageMessage("MiniCPM5-2B text generation on RDK S600");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc != 1) {
    std::cerr << "Unexpected positional argument; use --prompt or --follow_up\n";
    return 2;
  }
  try {
    minicpm5::Config config;
    config.model_path = FLAGS_model_path;
    config.max_new_tokens = FLAGS_max_new_tokens;
    minicpm5::MiniCPM5 model(config);
    for (int turn = 0; turn < (FLAGS_follow_up.empty() ? 1 : 2); ++turn) {
      const auto result = model.Generate(turn == 0 ? FLAGS_prompt : FLAGS_follow_up, turn == 0);
      const nlohmann::json output = {
          {"turn", turn + 1}, {"text", result.text}, {"token_ids", result.tokens},
          {"status", result.status}, {"ttft_ms", result.ttft_ms},
          {"decode_tps", result.decode_tps}, {"e2e_ms", result.e2e_ms}};
      std::cout << "RESULT " << output.dump() << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
