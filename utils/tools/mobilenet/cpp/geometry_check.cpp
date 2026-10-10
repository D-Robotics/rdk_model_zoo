// SPDX-License-Identifier: Apache-2.0
// Host-independent geometry probe; compares against Pillow fixtures externally.
#include "geometry.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("Usage: geometry_check IMAGE_LIST OUTPUT_DIRECTORY");
    std::ifstream list(argv[1]); std::string path; int index=0;
    while (std::getline(list,path)) {
      const auto crop=mobilenet::center_crop(cv::imread(path));
      if (!cv::imwrite(std::string(argv[2])+"/crop-"+std::to_string(index++)+".png",crop)) throw std::runtime_error("Cannot save crop");
    }
    if (!index) throw std::runtime_error("No images");
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
