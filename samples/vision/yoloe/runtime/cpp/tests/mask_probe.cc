// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Host comparison utility: compact proto + candidate boxes/coefficients to ROI
// bytes.
#include "image_ops.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
std::vector<float> read_floats(const std::string &path, size_t expected = 0) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  auto length = input.tellg();
  if (length < 0 || length % 4)
    throw std::invalid_argument("Invalid float input length");
  auto count = static_cast<size_t>(length) / 4;
  if (expected && count != expected)
    throw std::invalid_argument("Wrong float input size");
  std::vector<float> data(count);
  input.seekg(0);
  input.read(reinterpret_cast<char *>(data.data()), length);
  if (!input)
    throw std::invalid_argument("Failed to read float input");
  return data;
}
int main(int argc, char **argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "usage: mask_probe INPUT_DIR WIDTH HEIGHT NEW_OUTPUT_DIR");
    std::filesystem::path input = argv[1], output = argv[4];
    auto geometry = yoloe::make_geometry(std::stoi(argv[2]), std::stoi(argv[3]),
                                         yoloe::Protocol::E26);
    auto proto = read_floats((input / "9.f32").string(), 160 * 160 * 32);
    auto values = read_floats((input / "mask-candidates.f32").string());
    if (values.size() % 36)
      throw std::invalid_argument("Expected boxes4 + coefficients32");
    std::vector<yoloe::RawDetection> candidates;
    for (size_t start = 0; start < values.size(); start += 36) {
      yoloe::RawDetection candidate;
      std::copy_n(values.data() + start, 4, candidate.box.begin());
      std::copy_n(values.data() + start + 4, 32,
                  candidate.coefficients.begin());
      candidates.push_back(candidate);
    }
    auto masks = yoloe::restore_e26_masks(candidates, proto, geometry);
    if (!std::filesystem::create_directory(output))
      throw std::invalid_argument("Use a new output directory");
    std::cout << std::setprecision(9) << "[";
    for (size_t i = 0; i < masks.size(); ++i) {
      const auto &item = masks[i];
      const auto &mask = item.mask;
      std::string file = std::to_string(i) + ".u8";
      std::ofstream stream(output / file, std::ios::binary);
      if (!mask.empty())
        stream.write(reinterpret_cast<const char *>(mask.data), mask.total());
      if (!stream)
        throw std::runtime_error("Failed to write mask");
      if (i)
        std::cout << ",";
      std::cout << "{\"file\":\"" << file << "\",\"height\":" << mask.rows
                << ",\"width\":" << mask.cols << ",\"box\":[";
      for (int j = 0; j < 4; ++j) {
        if (j)
          std::cout << ",";
        std::cout << item.box[j];
      }
      std::cout << "]}";
    }
    std::cout << "]\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
