// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    auto options = yoloe::parse_cli(argc, argv);
    if (options.help) {
      std::cout << yoloe::cli_help();
      return 0;
    }
    auto gate = yoloe::make_preflight(options.model_sha256, options.label_path);
    gate(options.model); // identity and bytes before image/output or SDK work
    auto inputs = yoloe::load_cli_inputs(options);
    yoloe::create_output_directory(options.output);
    yoloe::YOLOE task(options.model, options.config, gate);
    auto result = task.predict(inputs.image);
    yoloe::save_cli_outputs(options, inputs, result);
    std::cout << "Saved " << result.size() << " instances to " << options.output
              << "/report.json\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
