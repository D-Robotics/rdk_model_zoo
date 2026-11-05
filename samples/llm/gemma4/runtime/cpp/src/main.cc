// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#include "gemma4.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  gemma4::Config config;
  std::string prompt = "What is 2 + 2? Answer with the number only.", follow_up;
  bool interactive = false;
  try {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        std::cout << "Usage: main --config_path PATH [--prompt TEXT] [--follow_up TEXT] "
                     "[--max_new_tokens 128] [--interactive]\n";
        return 0;
      }
      if (arg == "--interactive") { interactive = true; continue; }
      const auto equal = arg.find('=');
      const auto key = arg.substr(0, equal);
      if (key != "--config_path" && key != "--prompt" && key != "--follow_up" && key != "--max_new_tokens")
        throw std::runtime_error("Unknown argument: " + key);
      if (equal == std::string::npos && ++i >= argc) throw std::runtime_error("Missing value: " + key);
      const std::string value = equal == std::string::npos ? argv[i] : arg.substr(equal + 1);
      if (key == "--config_path") config.config_path = value;
      else if (key == "--prompt") prompt = value;
      else if (key == "--follow_up") follow_up = value;
      else {
        std::size_t used = 0;
        config.max_new_tokens = std::stoi(value, &used);
        if (used != value.size()) throw std::runtime_error("Invalid max_new_tokens");
      }
    }
    if (config.config_path.empty()) throw std::runtime_error("--config_path is required");
    gemma4::Gemma4 model(config);
    if (interactive) {
      std::string line;
      while (std::cout << "gemma4> " && std::getline(std::cin, line)) {
        if (line == "/quit") break;
        if (line == "/reset") { model.Reset(); continue; }
        if (!line.empty()) std::cout << "RESULT " << model.Generate(line).dump() << '\n';
      }
    } else {
      std::cout << "RESULT " << model.Generate(prompt).dump() << '\n';
      if (!follow_up.empty()) std::cout << "RESULT " << model.Generate(follow_up).dump() << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
