/**
 * @file gemma4_config.hpp
 * @brief Compile-time constants for Gemma4-E2B (model dims, special token IDs,
 *        quantization scales, KV-cache flush indices).
 *
 * Centralizes every "magic number" tied to the pre-compiled HBM models so
 * the runtime code stays free of literals. If you re-compile with different
 * `chunk_size` / `cache_len`, update kChunkSize / kCacheLen here too.
 */
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace gemma4 {

inline bool RuntimeDebugEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("GEMMA4_DEBUG");
    if (value == nullptr) {
      return false;
    }
    const std::string_view setting(value);
    return !setting.empty() && setting != "0" && setting != "false" &&
           setting != "FALSE";
  }();
  return enabled;
}

constexpr int kChunkSize = 256;
constexpr int kCacheLen = 4096;
constexpr int kSlidingWindow = 512;
constexpr int kHiddenSize = 1536;
constexpr int kVocabSize = 262144;
constexpr int kNumKvLayers = 15;
constexpr int kPadTokenId = 0;
constexpr int kEosTokenId = 1;
constexpr int kTurnEndTokenId = 106;

constexpr int kVisionSoftTokens = 280;
constexpr int kBoiTokenId = 255999;
constexpr int kImageTokenId = 249560;  // 🖼 (U+1F5BC) — the per-patch soft token
constexpr int kEoiTokenId = 258882;
constexpr int kVisionPatches = 2520;
constexpr int kVisionPatchDim = 768;
constexpr int kImageHeight = 672;
constexpr int kImageWidth = 960;
constexpr int kPatchSize = 16;

constexpr float kMaskValue = -32768.0f;
constexpr float kLogitScale = 0.00455029f;

constexpr int kHeadDims[kNumKvLayers] = {
    256, 256, 256, 256, 512, 256, 256, 256, 256, 512,
    256, 256, 256, 256, 512,
};

// Input tensor indices that need cache flush before each BPU inference.
// The runtime rolls newly produced KV rows into inputs 5..34 on CPU, so all
// text inputs must be cleaned before the BPU reads them again.
constexpr int kKvInputStart = 5;  // first KV cache input tensor index
constexpr int kLogitsOutputIndex = 0;

inline std::vector<int> TextInputFlushIndices() {
  std::vector<int> indices;
  indices.reserve(kKvInputStart + 2 * kNumKvLayers);
  for (int index = 0; index < kKvInputStart + 2 * kNumKvLayers; ++index) {
    indices.push_back(index);
  }
  return indices;
}

inline std::vector<int> PrefillFlushIndices() {
  return TextInputFlushIndices();
}

inline std::vector<int> DecodeFlushIndices() {
  return TextInputFlushIndices();
}

}  // namespace gemma4
