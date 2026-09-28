// Host numerical comparison driver only; this is not a model/SDK executable.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef LEGACY_CIF
#include "source_cif.h"
#else
#include "contract.h"
#endif
std::vector<float> read_values(const std::string &path, size_t count) {
  std::ifstream input(path, std::ios::binary);
  std::vector<float> values(count);
  input.read(reinterpret_cast<char *>(values.data()), std::streamsize(count * sizeof(float)));
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Input byte count mismatch");
  return values;
}
int main(int argc, char **argv) {
  try {
    if (argc < 5) throw std::runtime_error("mode input output count [vocabulary]");
    const int count = std::stoi(argv[4]);
    std::ofstream output(argv[3], std::ios::binary);
    if (!output) throw std::runtime_error("Output unavailable");
    if (std::string(argv[1]) == "cif") {
      auto input = read_values(argv[2], 401 + 401 * 512);
      std::vector<float> weights(input.begin(), input.begin() + 401);
      std::vector<float> hidden(input.begin() + 401, input.end());
#ifdef LEGACY_CIF
      std::vector<float> acoustic(100 * 512);
      int32_t tokens = 0;
      cif_numpy(weights.data(), hidden.data(), count, acoustic.data(), &tokens);
#else
      auto result = paraformer::cif(weights, hidden, count);
      const auto &acoustic = result.acoustic;
      int32_t tokens = result.token_count;
#endif
      output.write(reinterpret_cast<const char *>(&tokens), sizeof(tokens));
      output.write(reinterpret_cast<const char *>(acoustic.data()), acoustic.size() * sizeof(float));
    } else if (std::string(argv[1]) == "decode" && argc == 6) {
#ifndef LEGACY_CIF
      auto logits = read_values(argv[2], 100 * 8404);
      std::ifstream file(argv[5]);
      std::vector<std::string> vocabulary;
      for (std::string line; std::getline(file, line);) vocabulary.push_back(line);
      const auto decoded = paraformer::decode(logits, count, vocabulary);
      output << decoded.text;
#else
      throw std::runtime_error("Legacy fixture is CIF only");
#endif
    } else throw std::runtime_error("Unknown mode");
    if (!output) throw std::runtime_error("Output write failed");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
