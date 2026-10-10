// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// LaneNet S100 entry: parse options, run predict on one image, save the
// artifacts. Options, image loading and all artifact/report IO live in the
// CLI helpers; model lifecycle and stage math live in segment.cpp.
#include "cli.hpp"
#include "segment.hpp"
#include <iostream>
#include <vector>
int main(int argc, char **argv) {
  try {
    const auto options = lanenet::parse_options(
        std::vector<std::string>(argv + 1, argv + argc));
    if (options.help) {
      lanenet::print_help(argv[0]);
      return 0;
    }
    lanenet::validate_output_paths(options);
    const auto image = lanenet::load_image(options.image_path);
    // Construction loads the runtime and enforces the S100 board identity.
    lanenet::LaneNet model(options.model_path);
    const auto result = model.predict(image);
    lanenet::save_results(options, model, result);
    std::cout << "Saved embedding, binary labels and raw outputs to "
              << options.output_directory << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
