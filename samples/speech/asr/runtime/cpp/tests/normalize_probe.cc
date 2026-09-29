// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "contract.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  try {
    std::ifstream input(argv[1], std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (!input || bytes.empty() || bytes.size() % sizeof(float))
      throw std::invalid_argument("Invalid fixture bytes");
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    auto output = asr::normalize_and_pad(values);
    std::ofstream file(argv[2], std::ios::binary);
    file.write(reinterpret_cast<const char *>(output.values.data()),
               output.values.size() * sizeof(float));
    if (!file)
      return 2;
    std::cout << output.valid_samples << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
