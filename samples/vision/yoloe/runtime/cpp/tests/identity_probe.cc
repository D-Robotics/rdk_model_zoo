// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "platform_identity.h"
#include "sha256.h"
#include <iostream>
int main(int argc, char **argv) {
  if (argc == 3 && std::string(argv[1]) == "sha256") {
    auto h = rdk::sha256_file(argv[2]);
    std::cout << h << '\n';
    return h.empty() ? 2 : 0;
  }
  if (argc != 5)
    return 2;
  std::cout << rdk::identify_target({argv[1], argv[2], argv[3], argv[4]})
            << '\n';
}
