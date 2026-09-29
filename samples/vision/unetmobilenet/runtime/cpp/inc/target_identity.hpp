// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace unetmobilenet {
inline std::string normalize_identity(std::string value) {
    const auto begin=value.find_first_not_of(" \t\r\n");
    if(begin==std::string::npos)return {};
    value=value.substr(begin,value.find_last_not_of(" \t\r\n")-begin+1);
    std::transform(value.begin(),value.end(),value.begin(),[](unsigned char c){return std::tolower(c);});
    return value;
}
// Exact S aliases from docs/release/platforms.json; no unknown→S100 fallback.
inline std::string match_native_target(std::string soc, std::string board) {
    soc=normalize_identity(soc);board=normalize_identity(board);
    if(soc=="s100" && (board=="s100p" || board=="rdk s100p"))return "s100p";
    if(soc=="s100" || soc=="s100p" || soc=="s600")return soc;
    return {};
}
inline std::string read_identity(const char* path) {
    std::ifstream stream(path);
    return std::string(std::istreambuf_iterator<char>(stream),{});
}
inline void require_native_target(const std::string& requested, const std::string& built) {
    if(requested!=built || (requested!="s100" && requested!="s600"))
        throw std::invalid_argument("Requested target does not match the S100/S600 build");
    const auto actual=match_native_target(read_identity("/sys/class/boardinfo/soc_name"),
                                          read_identity("/sys/class/boardinfo/board_type"));
    if(actual!=requested)throw std::invalid_argument("Local board identity does not match requested target");
}
}  // namespace unetmobilenet
