// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Host verification utility, not a board inference executable.
#include "e26_decode.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument(
          "usage: decode_probe OUTPUT_TENSOR_DIRECTORY SINGLE_LABEL_0_OR_1");
    std::string mode = argv[2];
    if (mode != "0" && mode != "1")
      throw std::invalid_argument("mode must be 0 or 1");
    std::array<std::vector<float>, 10> tensors;
    for (int i = 0; i < 10; ++i) {
      int grid = i == 9 ? 160 : 80 >> (i / 3),
          channels = i == 9 ? 32 : (i % 3 == 0 ? 4585 : (i % 3 == 1 ? 4 : 32));
      auto &data = tensors[i];
      data.resize(static_cast<size_t>(grid) * grid * channels);
      std::ifstream stream(std::string(argv[1]) + "/" + std::to_string(i) +
                               ".f32",
                           std::ios::binary);
      stream.read(reinterpret_cast<char *>(data.data()),
                  data.size() * sizeof(float));
      if (!stream || stream.peek() != std::char_traits<char>::eof())
        throw std::invalid_argument("Missing, short or oversized float tensor");
    }
    auto result = yoloe::decode_e26(tensors, 0.25f, 300, mode == "1");
    std::cout << std::setprecision(9) << "[";
    for (size_t i = 0; i < result.size(); ++i) {
      const auto &row = result[i];
      if (i)
        std::cout << ",";
      std::cout << "{\"label\":" << row.label << ",\"score\":" << row.score
                << ",\"box\":[";
      for (int k = 0; k < 4; ++k) {
        if (k)
          std::cout << ",";
        std::cout << row.box[k];
      }
      std::cout << "],\"coefficients\":[";
      for (int k = 0; k < 32; ++k) {
        if (k)
          std::cout << ",";
        std::cout << row.coefficients[k];
      }
      std::cout << "]}";
    }
    std::cout << "]\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
