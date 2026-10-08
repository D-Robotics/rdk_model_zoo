// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "platform_identity.h"
#include <fstream>
#include <iterator>
namespace rdk {
namespace {
std::string read(const char *path) {
  std::ifstream file(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(file), {});
}
} // namespace
NativeIdentity read_native_identity() {
  return {read("/sys/class/boardinfo/soc_name"),
          read("/sys/class/boardinfo/board_type"),
          read("/sys/class/socinfo/soc_name"), read("/proc/device-tree/model")};
}
} // namespace rdk
