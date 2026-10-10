// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "cli.hpp"
#include "frontend.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  try {
    asr::AudioReader reader(argv[1]);
    asr::AudioChunk chunk;
    while (reader.next(chunk)) {
      const auto result = asr::prepare_chunk(chunk);
      std::ofstream out(std::string(argv[2]) + std::to_string(chunk.index) +
                            ".bin",
                        std::ios::binary);
      out.write(reinterpret_cast<const char *>(result.values.data()),
                result.values.size() * sizeof(float));
      if (!out)
        throw std::runtime_error("Could not write probe output");
      std::cout << chunk.index << ' ' << chunk.source_start << ' '
                << chunk.samples.size() / chunk.channels << ' '
                << result.valid_samples << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
