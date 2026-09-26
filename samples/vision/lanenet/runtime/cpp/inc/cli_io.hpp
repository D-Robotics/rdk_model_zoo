// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "tensor_contract.hpp"
namespace lanenet {
struct RuntimeOptions {
  std::string model_path, image_path, instance_path, binary_path;
  std::string output_directory = "outputs/lanenet_cpp", target = "s100";
  bool help = false;
};
RuntimeOptions parse_options(const std::vector<std::string> &args);
void validate_output_paths(const RuntimeOptions &options);
std::string json_quote(const std::string &value);
std::string dtype_descriptor(ScalarType type);
void write_npy(const std::string &path, ScalarType type,
               const std::vector<std::size_t> &shape,
               const std::vector<unsigned char> &bytes);
} // namespace lanenet
