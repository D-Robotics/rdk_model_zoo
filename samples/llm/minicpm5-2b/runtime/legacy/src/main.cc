/** @file main.cc Command-line entry point for S100/S100P text generation. */
#include "minicpm5.hpp"
#include <iostream>
#include <stdexcept>
/** Parse explicit paths supplied by run.sh and propagate runtime failures. */
int main(int argc, char** argv) {
  MiniCPM5Config config;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        std::cout << "main --model-path FILE --tokenizer-path DIR --template-path FILE [--prompt TEXT]\n";
        return 0;
      }
      if (i + 1 == argc) throw std::runtime_error("Missing value for " + option);
      const std::string value = argv[++i];
      if (option == "--model-path") config.model_path = value;
      else if (option == "--tokenizer-path") config.tokenizer_path = value;
      else if (option == "--template-path") config.template_path = value;
      else if (option == "--prompt") config.prompt = value;
      else throw std::runtime_error("Unknown option: " + option);
    }
    if (config.model_path.empty() || config.tokenizer_path.empty() || config.template_path.empty() || config.prompt.empty())
      throw std::runtime_error("Model, tokenizer, template and prompt must be nonempty; use run.sh for defaults");
    MiniCPM5 model(config);
    model.init();
    return model.predict();
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
