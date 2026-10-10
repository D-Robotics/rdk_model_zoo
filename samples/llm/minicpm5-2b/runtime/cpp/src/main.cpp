/** @file main.cpp
 * @brief Run one MiniCPM5 prompt and an optional follow-up on RDK S600.
 *
 * The CLI owns flags and RESULT emission; the named model owns runtime
 * initialization and the preprocess -> infer -> postprocess chain.
 */
#include <exception>

#include "cli.hpp"
#include "minicpm5.hpp"

/** @brief Parse CLI options, generate text per turn and emit results.
 * @param argc Argument count.
 * @param argv Argument values.
 * @return Zero for completed requests (including configured limits), 2 for a
 * positional argument, 1 for a reported error.
 */
int main(int argc, char** argv) {
  minicpm5::CliOptions options;
  if (!minicpm5::parse_cli(argc, argv, options)) return 2;
  try {
    minicpm5::Config config;
    config.model_path = options.model_path;
    config.max_new_tokens = options.max_new_tokens;
    minicpm5::MiniCPM5 model(config);
    const int turns = options.follow_up.empty() ? 1 : 2;
    for (int turn = 0; turn < turns; ++turn) {
      const auto result = model.predict(
          turn == 0 ? options.prompt : options.follow_up, turn == 0);
      minicpm5::emit_result(turn + 1, result);
    }
  } catch (const std::exception& error) {
    return minicpm5::fail(error);
  }
  return 0;
}
