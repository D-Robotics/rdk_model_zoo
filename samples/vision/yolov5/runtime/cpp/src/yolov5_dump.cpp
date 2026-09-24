// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0

#include "yolov5_dump.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace yolov5 {
namespace {

// ---------------------------------------------------------------- SHA-256 ---
// Local, dependency-free SHA-256 so the dump can bind a run to its model and
// input bytes on a board image that has no crypto library.

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t rotr(std::uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(const unsigned char* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
      buffer_[buffer_size_++] = data[i];
      if (buffer_size_ == 64) {
        compress(buffer_.data());
        bit_length_ += 512;
        buffer_size_ = 0;
      }
    }
  }

  std::string hex() {
    const std::uint64_t bits = bit_length_ + buffer_size_ * 8U;
    const unsigned char pad = 0x80U;
    update(&pad, 1);
    const unsigned char zero = 0x00U;
    while (buffer_size_ != 56) update(&zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; ++i)
      length[i] = static_cast<unsigned char>((bits >> (56 - 8 * i)) & 0xffU);
    update(length, 8);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t word : state_) out << std::setw(8) << word;
    return out.str();
  }

 private:
  void compress(const unsigned char* block) {
    std::uint32_t words[64];
    for (int i = 0; i < 16; ++i)
      words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                 (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                 (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                 static_cast<std::uint32_t>(block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(words[i - 15], 7) ^ rotr(words[i - 15], 18) ^
                               (words[i - 15] >> 3);
      const std::uint32_t s1 = rotr(words[i - 2], 17) ^ rotr(words[i - 2], 19) ^
                               (words[i - 2] >> 10);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + words[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g; g = f; f = e; e = d + temp1;
      d = c; c = b; b = a; a = temp1 + temp2;
    }
    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
  }

  std::uint32_t state_[8];
  std::array<unsigned char, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_length_ = 0;
};

// ------------------------------------------------------------------ JSON ----

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (const char raw : value) {
    const unsigned char ch = static_cast<unsigned char>(raw);
    switch (ch) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (ch < 0x20) {
          char buffer[8];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
          out += buffer;
        } else {
          out += static_cast<char>(ch);
        }
    }
  }
  return out;
}

std::string quote(const std::string& value) { return "\"" + json_escape(value) + "\""; }

std::string number(long long value) { return std::to_string(value); }
std::string number(int value) { return std::to_string(value); }

std::string number(double value) {
  std::ostringstream out;
  out << std::setprecision(9) << value;
  return out.str();
}

std::string shape_json(const std::vector<long long>& shape) {
  std::string out = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += number(shape[i]);
  }
  return out + "]";
}

// A -1 entry means "not reported by this SDK" and serializes as null so a
// consumer cannot mistake it for a real stride or dimension.
std::string opt_number(long long value) {
  return value < 0 ? std::string("null") : number(value);
}

std::string opt_array_json(const long long (&values)[4]) {
  std::string out = "[";
  for (int i = 0; i < 4; ++i) {
    if (i) out += ", ";
    out += opt_number(values[i]);
  }
  return out + "]";
}

std::string double_array_json(const std::vector<double>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += number(values[i]);
  }
  return out + "]";
}

std::string long_array_json(const std::vector<long long>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += number(values[i]);
  }
  return out + "]";
}

std::string tensor_info_json(const DumpTensorInfo& info) {
  std::string block = "      {\n";
  block += "        \"name\": " + quote(info.name) + ",\n";
  block += "        \"dtype\": " + quote(info.dtype) + ",\n";
  block += "        \"shape\": " + shape_json(info.shape) + ",\n";
  block += "        \"quanti\": " + quote(info.quanti) + ",\n";
  block += "        \"scale_len\": " + number(info.scale_len) + ",\n";
  block += "        \"aligned_byte_size\": " + opt_number(info.aligned_byte_size) + ",\n";
  block += "        \"stride\": " + opt_array_json(info.stride) + ",\n";
  block += "        \"aligned\": " + opt_array_json(info.aligned) + ",\n";
  block += "        \"quantize_axis\": " + opt_number(info.quantize_axis) + ",\n";
  block += "        \"scale_values\": " + double_array_json(info.scale_values) + ",\n";
  block += "        \"zero_point_values\": " + long_array_json(info.zero_point_values) + "\n";
  block += "      }";
  return block;
}

std::string string_array_json(const std::vector<std::string>& values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += quote(values[i]);
  }
  return out + "]";
}

}  // namespace

std::string sha256_hex(const void* data, std::size_t size) {
  Sha256 hasher;
  hasher.update(static_cast<const unsigned char*>(data), size);
  return hasher.hex();
}

DumpTensorInfo dump_tensor_info(const std::string& name, const TensorMeta& meta,
                                const float* scale_data,
                                const std::int32_t* zero_point_data) {
  DumpTensorInfo info;
  info.name = name;
  info.dtype = dtype_name(meta.dtype);
  info.shape.clear();
  for (int i = 0; i < meta.num_dimensions && i < 4; ++i)
    info.shape.push_back(meta.valid[i]);
  info.quanti = quanti_name(meta.quanti_type);
  info.scale_len = meta.scale_len;
  info.aligned_byte_size = meta.aligned_byte_size;
  // A non-positive entry means the projection never saw a reported value
  // (TensorMeta zero-initializes; the X5 gate treats aligned 0 the same way),
  // so it is recorded as unreported rather than as a fake zero stride.
  for (int i = 0; i < 4; ++i) info.stride[i] = meta.stride[i] > 0 ? meta.stride[i] : -1;
  for (int i = 0; i < 4; ++i) info.aligned[i] = meta.aligned[i] > 0 ? meta.aligned[i] : -1;
  info.quantize_axis = meta.quantize_axis;
  if (meta.quanti_type == kQuantiScale) {
    if (scale_data == nullptr || meta.scale_len <= 0)
      throw std::invalid_argument("dump_tensor_info: SCALE tensor without a readable "
                                  "scale descriptor: " + name);
    if (meta.scale_len > kMaxQuantValues)
      throw std::invalid_argument("dump_tensor_info: scale descriptor exceeds the "
                                  "recorded bound (" + std::to_string(meta.scale_len) +
                                  ")");
    info.scale_values.assign(scale_data, scale_data + meta.scale_len);
    if (meta.zero_point_len > 0) {
      if (zero_point_data == nullptr)
        throw std::invalid_argument("dump_tensor_info: zero-point length without a "
                                    "buffer: " + name);
      if (meta.zero_point_len > kMaxQuantValues)
        throw std::invalid_argument("dump_tensor_info: zero-point descriptor exceeds "
                                    "the recorded bound");
      info.zero_point_values.assign(zero_point_data,
                                    zero_point_data + meta.zero_point_len);
    }
  }
  return info;
}

std::string sha256_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  Sha256 hasher;
  std::array<char, 1 << 16> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = input.gcount();
    if (got > 0)
      hasher.update(reinterpret_cast<const unsigned char*>(buffer.data()),
                    static_cast<std::size_t>(got));
  }
  return hasher.hex();
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

bool write_dump(const DumpRecord& record, std::string* error) {
  const auto fail = [error](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (record.dir.empty()) return fail("dump directory is empty");

  std::error_code code;
  std::filesystem::create_directories(record.dir, code);
  if (code) return fail("cannot create dump directory: " + code.message());

  // Each stage writes into its own subdirectory: the raw and transformed
  // payloads of one output share neither a file nor bytes, so a later stage
  // can never overwrite the original evidence of an earlier one.
  const auto write_tensor = [&](const DumpTensor& tensor, const std::string& category,
                                std::size_t index) -> std::string {
    const std::string filename = category + "/" + std::to_string(index) + "-" +
                                 (tensor.name.empty() ? "tensor" : tensor.name) + ".bin";
    const std::filesystem::path path = std::filesystem::path(record.dir) / filename;
    if (path.has_parent_path()) {
      std::filesystem::create_directories(path.parent_path(), code);
      if (code) return {};
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return {};
    if (!tensor.bytes.empty())
      out.write(reinterpret_cast<const char*>(tensor.bytes.data()),
                static_cast<std::streamsize>(tensor.bytes.size()));
    if (!out) return {};
    return filename;
  };
  const auto hash_of = [](const DumpTensor& tensor) {
    return tensor.bytes.empty() ? std::string()
                                : sha256_hex(tensor.bytes.data(), tensor.bytes.size());
  };

  std::vector<std::string> input_files;
  std::vector<std::string> input_hashes;
  for (std::size_t i = 0; i < record.input_tensors.size(); ++i) {
    const std::string filename = write_tensor(record.input_tensors[i], "input", i);
    if (filename.empty()) return fail("cannot write input tensor file");
    input_files.push_back(filename);
    input_hashes.push_back(hash_of(record.input_tensors[i]));
  }
  std::vector<std::string> raw_files;
  std::vector<std::string> raw_hashes;
  for (std::size_t i = 0; i < record.raw_tensors.size(); ++i) {
    const std::string filename = write_tensor(record.raw_tensors[i], "raw", i);
    if (filename.empty()) return fail("cannot write raw tensor file");
    raw_files.push_back(filename);
    raw_hashes.push_back(hash_of(record.raw_tensors[i]));
  }
  std::vector<std::string> transformed_files;
  std::vector<std::string> transformed_hashes;
  for (std::size_t i = 0; i < record.transformed_tensors.size(); ++i) {
    const std::string filename = write_tensor(record.transformed_tensors[i], "transformed", i);
    if (filename.empty()) return fail("cannot write transformed tensor file");
    transformed_files.push_back(filename);
    transformed_hashes.push_back(hash_of(record.transformed_tensors[i]));
  }

  const auto tensor_list_json = [](const std::vector<DumpTensor>& tensors,
                                   const std::vector<std::string>& files,
                                   const std::vector<std::string>& hashes) {
    std::string list = "[\n";
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = tensors[i];
      list += "    {\"name\": " + quote(tensor.name) +
              ", \"dtype\": " + quote(tensor.dtype) +
              ", \"shape\": " + shape_json(tensor.shape) +
              ", \"bytes\": " + number(static_cast<long long>(tensor.bytes.size())) +
              ", \"file\": " + quote(files[i]) +
              ", \"sha256\": " + quote(hashes[i]) + "}";
      list += (i + 1 == tensors.size()) ? "\n" : ",\n";
    }
    return list + "  ]";
  };

  std::string json = "{\n";
  json += "  \"schema\": \"rdk-model-zoo/yolov5-cpp-dump/v2\",\n";
  json += "  \"utc\": " + quote(record.utc) + ",\n";
  json += "  \"target\": " + quote(record.target) + ",\n";
  json += "  \"build_target\": " + quote(record.build_target) + ",\n";
  json += "  \"asset_id\": " + quote(record.asset_id) + ",\n";
  json += "  \"model_path\": " + quote(record.model_path) + ",\n";
  json += "  \"model_sha256\": " + quote(sha256_file(record.model_path)) + ",\n";
  json += "  \"binary_path\": " + quote(record.binary_path) + ",\n";
  json += "  \"binary_sha256\": " +
          (record.binary_path.empty()
               ? std::string("null")
               : quote(sha256_file(record.binary_path))) +
          ",\n";
  json += "  \"image_path\": " + quote(record.image_path) + ",\n";
  json += "  \"image_sha256\": " + quote(sha256_file(record.image_path)) + ",\n";
  json += "  \"cwd\": " + quote(record.cwd) + ",\n";
  json += "  \"argv\": " + string_array_json(record.argv) + ",\n";
  json += "  \"return_code\": " + number(record.return_code) + ",\n";
  json += "  \"error\": " + quote(record.error) + ",\n";
  json += "  \"notes\": " + string_array_json(record.notes) + ",\n";

  json += "  \"parameters\": {\n";
  for (std::size_t i = 0; i < record.options.size(); ++i) {
    json += "    " + quote(record.options[i].first) + ": " +
            quote(record.options[i].second);
    json += (i + 1 == record.options.size()) ? "\n" : ",\n";
  }
  json += "  },\n";

  json += "  \"inputs\": [\n";
  for (std::size_t i = 0; i < record.inputs.size(); ++i) {
    json += tensor_info_json(record.inputs[i]);
    json += (i + 1 == record.inputs.size()) ? "\n" : ",\n";
  }
  json += "  ],\n";

  json += "  \"outputs\": [\n";
  for (std::size_t i = 0; i < record.outputs.size(); ++i) {
    json += tensor_info_json(record.outputs[i]);
    json += (i + 1 == record.outputs.size()) ? "\n" : ",\n";
  }
  json += "  ],\n";

  json += "  \"input_tensors\": " +
          (record.input_tensors.empty()
               ? std::string("[]")
               : tensor_list_json(record.input_tensors, input_files, input_hashes)) +
          ",\n";
  json += "  \"raw_tensors\": " +
          (record.raw_tensors.empty()
               ? std::string("[]")
               : tensor_list_json(record.raw_tensors, raw_files, raw_hashes)) +
          ",\n";
  json += "  \"transformed_tensors\": " +
          (record.transformed_tensors.empty()
               ? std::string("[]")
               : tensor_list_json(record.transformed_tensors, transformed_files,
                                  transformed_hashes)) +
          ",\n";

  const auto detections_block = [](const std::vector<Detection>& list) {
    std::string block = "[\n";
    for (std::size_t i = 0; i < list.size(); ++i) {
      const auto& det = list[i];
      block += "    {\"x1\": " + number(static_cast<double>(det.x1)) +
               ", \"y1\": " + number(static_cast<double>(det.y1)) +
               ", \"x2\": " + number(static_cast<double>(det.x2)) +
               ", \"y2\": " + number(static_cast<double>(det.y2)) +
               ", \"score\": " + number(static_cast<double>(det.score)) +
               ", \"class_id\": " + number(det.class_id) + "}";
      block += (i + 1 == list.size()) ? "\n" : ",\n";
    }
    return block + "  ]";
  };
  json += "  \"detections\": " + detections_block(record.detections) + ",\n";
  json += "  \"detections_original\": " + detections_block(record.detections_original) +
          "\n}\n";

  std::ofstream manifest(std::filesystem::path(record.dir) / "manifest.json",
                         std::ios::binary | std::ios::trunc);
  if (!manifest) return fail("cannot open dump manifest for writing");
  manifest << json;
  if (!manifest) return fail("cannot write dump manifest");
  return true;
}

std::string current_binary_path(const std::string& argv0) {
#if defined(__linux__)
  std::error_code code;
  const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", code);
  if (!code && !self.empty() && self.is_absolute()) return self.string();
#endif
  if (argv0.empty()) return {};
  std::error_code resolve_error;
  const std::filesystem::path resolved = std::filesystem::absolute(argv0, resolve_error);
  if (!resolve_error) return resolved.string();
  return {};
}

std::vector<Detection> map_to_original(const std::vector<Detection>& detections,
                                       int image_cols, int image_rows, int model_size) {
  // Mirrors render_detections: uniform letterbox scale to the square model
  // input, symmetric padding, then the inverse mapping per coordinate. The
  // results are unclamped floats — they describe the detection, not the
  // pixels the renderer ends up drawing.
  std::vector<Detection> mapped;
  if (image_cols <= 0 || image_rows <= 0 || model_size <= 0) return mapped;
  const double scale = std::min(static_cast<double>(model_size) / image_cols,
                                static_cast<double>(model_size) / image_rows);
  const double pad_x = (model_size - image_cols * scale) / 2.0;
  const double pad_y = (model_size - image_rows * scale) / 2.0;
  mapped.reserve(detections.size());
  for (const auto& detection : detections) {
    Detection out;
    out.x1 = static_cast<float>((detection.x1 - pad_x) / scale);
    out.y1 = static_cast<float>((detection.y1 - pad_y) / scale);
    out.x2 = static_cast<float>((detection.x2 - pad_x) / scale);
    out.y2 = static_cast<float>((detection.y2 - pad_y) / scale);
    out.score = detection.score;
    out.class_id = detection.class_id;
    mapped.push_back(out);
  }
  return mapped;
}

}  // namespace yolov5
