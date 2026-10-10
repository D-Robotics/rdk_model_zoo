// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// Thin entry: parse options, construct the named model, run one predict,
// hand the result to the CLI module for artifact/report IO.
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli.hpp"
#include "depth.hpp"

namespace fs = std::filesystem;
using namespace yolo26_depth;
int main(int argc, char **argv) {
  try {
    const auto options =
        parse_options(std::vector<std::string>(argv + 1, argv + argc));
    if (options.help) {
      print_help();
      return 0;
    }
    if (fs::exists(options.output_directory))
      throw std::invalid_argument("Output directory must be new");
    const auto image = load_image(options.image_path);
    Yolo26Depth model(options.model_path, DepthOptions{options.warmup});
    const auto result = model.predict(image);
    save_results(options, image, result, model.model_name());
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}
