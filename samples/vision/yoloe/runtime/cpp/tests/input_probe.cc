// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
// Host-only preparation probe. Backend inference deliberately cannot run.
#include "detect.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
class UnusedRunner : public yoloe::Runner {
public:
  explicit UnusedRunner(yoloe::Protocol p) : protocol_(p) {}
  yoloe::Protocol protocol() const override { return protocol_; }
  yoloe::Heads infer(const yoloe::Nv12Input &) override {
    throw std::logic_error("Inference is not part of this preparation probe");
  }

private:
  yoloe::Protocol protocol_;
};
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::invalid_argument("usage: input_probe BGR_FILE WIDTH HEIGHT "
                                  "11_OR_26 RESIZE_TYPE NEW_OUTPUT_DIR");
    int width = std::stoi(argv[2]), height = std::stoi(argv[3]);
    std::string family = argv[4];
    if (family != "11" && family != "26")
      throw std::invalid_argument("Unknown family");
    yoloe::Config cfg;
    cfg.protocol = family == "11" ? yoloe::Protocol::E11 : yoloe::Protocol::E26;
    cfg.resize_type = std::stoi(argv[5]);
    yoloe::make_geometry(width, height, cfg.protocol, cfg.resize_type);
    cv::Mat image(height, width, CV_8UC3);
    std::ifstream stream(argv[1], std::ios::binary);
    stream.read(reinterpret_cast<char *>(image.data), image.total() * 3);
    if (!stream || stream.peek() != std::char_traits<char>::eof())
      throw std::invalid_argument("BGR file size mismatch");
    yoloe::YOLOE task(cfg, std::make_unique<UnusedRunner>(cfg.protocol));
    auto prepared = task.preprocess(image);
    std::filesystem::path output = argv[6];
    if (!std::filesystem::create_directory(output))
      throw std::invalid_argument("Use new output directory");
    for (auto plane : {std::make_pair("y.u8", &prepared.input().y),
                       std::make_pair("uv.u8", &prepared.input().uv)}) {
      std::ofstream file(output / plane.first, std::ios::binary);
      file.write(reinterpret_cast<const char *>(plane.second->data()),
                 plane.second->size());
      if (!file)
        throw std::runtime_error("Write failed");
    }
    const auto &g = prepared.geometry();
    std::cout << "{\"resized\":[" << g.resized_h << "," << g.resized_w
              << "],\"padding\":[" << g.left << "," << g.top << "," << g.right
              << "," << g.bottom << "]}\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}
