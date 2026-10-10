/** @file cli.cpp Argument parsing, stdout sink and RESULT rendering. */
#include "cli.hpp"

#include <iostream>
#include <stdexcept>

bool parse_cli(int argc, char* argv[], CliOptions& options) {
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help") {
      std::cout << "main --model-path FILE --tokenizer-path DIR "
                   "--template-path FILE [--prompt TEXT]\n";
      return false;
    }
    if (i + 1 == argc) throw std::runtime_error("Missing value for " + option);
    const std::string value = argv[++i];
    if (option == "--model-path")
      options.model_path = value;
    else if (option == "--tokenizer-path")
      options.tokenizer_path = value;
    else if (option == "--template-path")
      options.template_path = value;
    else if (option == "--prompt")
      options.prompt = value;
    else
      throw std::runtime_error("Unknown option: " + option);
  }
  if (options.model_path.empty() || options.tokenizer_path.empty() ||
      options.template_path.empty() || options.prompt.empty())
    throw std::runtime_error(
        "Model, tokenizer, template and prompt must be nonempty; use run.sh "
        "for defaults");
  return true;
}

std::function<void(const char*)> stdout_sink() {
  return [](const char* chunk) { std::cout << chunk << std::flush; };
}

void emit_result(const RequestOutcome& outcome) {
  if (outcome.stream_error)
    std::cerr << "WARNING: streaming consumer failed mid-request\n";
  std::cout << "\nRESULT status=" << outcome.sdk_status
            << " ended=" << outcome.ended << " failed=" << outcome.failed
            << " destroy=" << outcome.destroy_status << '\n';
}

int fail(const std::exception& error) {
  std::cerr << "ERROR: " << error.what() << '\n';
  return 1;
}
