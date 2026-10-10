/** @file main.cpp Command-line entry point for S100/S100P text generation.
 *
 * The CLI owns argument parsing, the stdout sink and RESULT rendering; the
 * named model owns SDK initialization and the single request.
 */
#include "cli.hpp"
#include "minicpm5.hpp"
/** Parse explicit paths supplied by run.sh and propagate runtime failures. */
int main(int argc, char** argv) {
  CliOptions options;
  try {
    if (!parse_cli(argc, argv, options)) return 0;  // --help
    MiniCPM5Config config;
    config.model_path = options.model_path;
    config.tokenizer_path = options.tokenizer_path;
    config.template_path = options.template_path;
    config.prompt = options.prompt;
    // Console presentation lives here: stream tokens to stdout via the sink.
    config.text_sink = stdout_sink();
    MiniCPM5 model(config);
    const RequestOutcome outcome = model.predict();
    emit_result(outcome);
    return outcome.exit_code();
  } catch (const std::exception& error) {
    return fail(error);
  }
}
