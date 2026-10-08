// Copyright (c) 2026 D-Robotics Corporation
// SPDX-License-Identifier: Apache-2.0
// Shared dependency-free streaming SHA-256, extracted from YOLOv5 run evidence.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
namespace rdk {
namespace sha256_detail {
constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

inline std::uint32_t rotr(std::uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
public:
  Sha256()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void update(const unsigned char *data, std::size_t size) {
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
    while (buffer_size_ != 56)
      update(&zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; ++i)
      length[i] = static_cast<unsigned char>((bits >> (56 - 8 * i)) & 0xffU);
    update(length, 8);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::uint32_t word : state_)
      out << std::setw(8) << word;
    return out.str();
  }

private:
  void compress(const unsigned char *block) {
    std::uint32_t words[64];
    for (int i = 0; i < 16; ++i)
      words[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                 (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                 (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                 static_cast<std::uint32_t>(block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(words[i - 15], 7) ^
                               rotr(words[i - 15], 18) ^ (words[i - 15] >> 3);
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
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::uint32_t state_[8];
  std::array<unsigned char, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_length_ = 0;
};

} // namespace sha256_detail
inline std::string sha256_hex(const void *data, std::size_t size) {
  sha256_detail::Sha256 hasher;
  hasher.update(static_cast<const unsigned char *>(data), size);
  return hasher.hex();
}
// Empty string means open/read failure; the digest of an empty file is
// nonempty.
inline std::string sha256_file(const std::string &path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error) || error)
    return {};
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  sha256_detail::Sha256 hasher;
  std::array<char, 1 << 16> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize got = input.gcount();
    if (got > 0)
      hasher.update(reinterpret_cast<const unsigned char *>(buffer.data()),
                    static_cast<std::size_t>(got));
  }
  if (input.bad())
    return {};
  return hasher.hex();
}
} // namespace rdk
