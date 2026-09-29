// Copyright (c) 2026 D-Robotics. SPDX-License-Identifier: Apache-2.0
#include "feature_io.h"
#include "sha256.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
namespace paraformer {
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}
std::string sha(std::string value) {
  require(value.size() == 64 && std::all_of(value.begin(), value.end(),
                                            [](unsigned char c) {
                                              return (c >= '0' && c <= '9') ||
                                                     (c >= 'a' && c <= 'f') ||
                                                     (c >= 'A' && c <= 'F');
                                            }),
          "Expected feature SHA-256");
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}
int positive_integer(const nlohmann::json &value) {
  require(value.is_number_integer(), "Frame counts must be integers");
  const auto number = value.get<int64_t>();
  require(number > 0 && number <= std::numeric_limits<int>::max(),
          "Frame count out of range");
  return int(number);
}
bool whitespace(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' ||
         c == '\f';
}
// Small data-only parser for the NPY scalar float array header grammar. It
// never evaluates Python and accepts key order/quote style independently.
class Header {
  const std::string &s;
  size_t p = 0;
  void spaces() {
    while (p < s.size() && whitespace(s[p]))
      ++p;
  }
  char peek() {
    spaces();
    return p < s.size() ? s[p] : '\0';
  }
  void take(char c) {
    require(peek() == c, "Malformed NPY header");
    ++p;
  }
  std::string string() {
    const char q = peek();
    require(q == '\'' || q == '"', "Expected NPY string");
    ++p;
    const size_t start = p;
    while (p < s.size() && s[p] != q) {
      require(s[p] != '\\' && s[p] != '\0' && s[p] != '\n',
              "Unsupported NPY string escape");
      ++p;
    }
    require(p < s.size(), "Unclosed NPY string");
    auto out = s.substr(start, p - start);
    ++p;
    return out;
  }
  int integer() {
    spaces();
    const size_t start = p;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9')
      ++p;
    require(p > start && p - start <= 9, "Invalid NPY dimension");
    return std::stoi(s.substr(start, p - start));
  }

public:
  explicit Header(const std::string &text) : s(text) {}
  std::string parse() {
    std::set<std::string> keys;
    std::string dtype;
    std::vector<int> shape;
    take('{');
    while (peek() != '}') {
      const auto key = string();
      require(keys.insert(key).second, "Duplicate NPY key");
      take(':');
      if (key == "descr")
        dtype = string();
      else if (key == "fortran_order") {
        spaces();
        require(s.compare(p, 5, "False") == 0,
                "Only C-order features supported");
        p += 5;
      } else if (key == "shape") {
        take('(');
        while (peek() != ')') {
          shape.push_back(integer());
          require(shape.size() <= 3, "Expected three NPY dimensions");
          if (peek() == ')')
            break;
          take(',');
        }
        take(')');
      } else
        throw std::invalid_argument("Unknown NPY header key");
      if (peek() == '}')
        break;
      take(',');
    }
    take('}');
    spaces();
    require(p == s.size(), "Trailing NPY header syntax");
    require(keys == std::set<std::string>{"descr", "fortran_order", "shape"},
            "Incomplete NPY header");
    require(shape == std::vector<int>{1, 400, 560},
            "Expected feature shape [1,400,560]");
    require(dtype == "<f4" || dtype == ">f4",
            "Expected explicit-endian float32 features");
    return dtype;
  }
};
} // namespace
std::vector<FeatureItem> load_prepared_manifest(const std::string &path,
                                                size_t max_utts) {
  std::ifstream file(path, std::ios::binary);
  require(bool(file), "Cannot open prepared manifest");
  auto entries = nlohmann::json::parse(file);
  require(entries.is_array() && !entries.empty(),
          "Prepared manifest must be a nonempty JSON list");
  std::vector<FeatureItem> result;
  std::set<std::string> ids;
  for (const auto &entry : entries) {
    require(entry.is_object(), "Prepared entry must be an object");
    FeatureItem item;
    item.utt_id = entry.at("utt_id").get<std::string>();
    require(!item.utt_id.empty() && item.utt_id != "." && item.utt_id != ".." &&
                !whitespace(item.utt_id.front()) &&
                !whitespace(item.utt_id.back()) &&
                item.utt_id.find_first_of("/\\") == std::string::npos &&
                item.utt_id.find('\0') == std::string::npos,
            "Invalid utterance ID");
    require(ids.insert(item.utt_id).second, "Duplicate utterance ID");
    item.valid_frames = positive_integer(entry.at("feat_length"));
    item.original_frames = positive_integer(entry.at("original_frames"));
    require(entry.at("truncated").is_boolean(), "truncated must be boolean");
    item.truncated = entry.at("truncated").get<bool>();
    require(item.valid_frames == std::min(item.original_frames, 400) &&
                item.truncated == (item.original_frames > 400),
            "Inconsistent original/valid frames or truncation");
    item.sha256 = sha(entry.at("feature_sha256").get<std::string>());
    auto feature = entry.at("feature_file").get<std::string>();
    require(!feature.empty() && feature.find('\0') == std::string::npos,
            "Invalid feature path");
    item.path = (std::filesystem::absolute(path).parent_path() / feature)
                    .lexically_normal()
                    .string();
    if (entry.contains("text"))
      item.reference_text = entry.at("text").get<std::string>();
    item.original_record_json = entry.dump();
    result.push_back(std::move(item));
  }
  if (max_utts && max_utts < result.size())
    result.resize(max_utts);
  return result;
}
std::vector<float> load_features(const FeatureItem &item) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                "Requires IEEE float32");
  require(item.valid_frames >= 1 && item.valid_frames <= 400 &&
              item.original_frames >= item.valid_frames &&
              item.valid_frames == std::min(item.original_frames, 400) &&
              item.truncated == (item.original_frames > 400),
          "Invalid feature frame metadata");
  const auto expected = sha(item.sha256);
  require(std::filesystem::is_regular_file(item.path),
          "Feature file must be regular");
  const auto size = std::filesystem::file_size(item.path);
  constexpr size_t payload = 400 * 560 * 4;
  require(size >= payload + 10 && size <= payload + 65536 + 12,
          "Invalid feature NPY size");
  std::vector<unsigned char> bytes(size);
  std::ifstream file(item.path, std::ios::binary);
  file.read(reinterpret_cast<char *>(bytes.data()),
            std::streamsize(bytes.size()));
  require(file.gcount() == std::streamsize(bytes.size()) &&
              file.peek() == std::ifstream::traits_type::eof() && !file.bad(),
          "Incomplete or changed feature file");
  require(rdk::sha256_hex(bytes.data(), bytes.size()) == expected,
          "Feature SHA-256 mismatch");
  require(std::memcmp(bytes.data(), "\x93NUMPY", 6) == 0, "Invalid NPY magic");
  const int major = bytes[6], minor = bytes[7];
  require((major == 1 || major == 2 || major == 3) && minor == 0,
          "Unsupported NPY version");
  const size_t width = major == 1 ? 2 : 4, begin = 8 + width;
  uint32_t length = 0;
  for (size_t i = 0; i < width; ++i)
    length |= uint32_t(bytes[8 + i]) << (8 * i);
  require(length > 0 && length <= 65536 &&
              begin + length + payload == bytes.size(),
          "Invalid NPY header/payload length");
  const std::string header(reinterpret_cast<const char *>(bytes.data() + begin),
                           length);
  require(header.back() == '\n', "NPY header must end in newline");
  const auto dtype = Header(header).parse();
  const bool little = dtype == "<f4";
  std::vector<float> values(400 * 560);
  const auto *data = bytes.data() + begin + length;
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t bits = 0;
    for (size_t j = 0; j < 4; ++j)
      bits |= uint32_t(data[4 * i + j]) << (8 * (little ? j : 3 - j));
    std::memcpy(&values[i], &bits, 4);
    require(std::isfinite(values[i]), "Features must be finite");
  }
  return values;
}
} // namespace paraformer
