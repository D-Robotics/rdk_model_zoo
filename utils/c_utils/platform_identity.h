// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cctype>
#include <string>
namespace rdk {
struct NativeIdentity {
  std::string soc, board, socinfo, device_tree;
};
inline std::string trim_identity(std::string value, bool lower = true) {
  auto whitespace = [](unsigned char c) {
    return c == 0 || c == ' ' || c == '\t' || c == '\r' || c == '\n';
  };
  auto first = std::find_if_not(value.begin(), value.end(), whitespace);
  auto last = std::find_if_not(value.rbegin(), value.rend(), whitespace).base();
  if (first >= last)
    return {};
  value = std::string(first, last);
  if (lower)
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  return value;
}
// Exact aliases and fallback precedence from docs/release/platforms.json and
// utils/py_utils/platforms.py. Registry parity is exercised by the host probe.
inline std::string identify_target(const NativeIdentity &raw) {
  const auto soc = trim_identity(raw.soc), board = trim_identity(raw.board);
  if (!soc.empty()) {
    if (soc == "s100" && (board == "s100p" || board == "rdk s100p"))
      return "s100p";
    if (soc == "x5" || soc == "s100" || soc == "s100p" || soc == "s600")
      return soc;
    return {};
  }
  const auto socinfo = trim_identity(raw.socinfo);
  if (!socinfo.empty())
    return (socinfo == "x5u" || socinfo == "x5h" || socinfo == "x5m") ? "x5"
                                                                      : "";
  return trim_identity(raw.device_tree, false) == "D-Robotics RDK X5 V1.0"
             ? "x5"
             : "";
}
NativeIdentity read_native_identity();
} // namespace rdk
