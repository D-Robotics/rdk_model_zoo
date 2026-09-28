// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli_io.hpp"
#include <iostream>
int main(int argc, char **argv) {
  try {
    auto options = himloco::parse_cli(argc, argv);
    if (options.help) {
      std::cout << himloco::cli_help();
      return 0;
    }
    himloco::execute(options);
    std::cout << "completed report=" << options.report << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "HIMLoco: " << e.what() << '\n';
    return 2;
  }
}
