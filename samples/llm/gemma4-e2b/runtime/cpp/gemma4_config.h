#pragma once

#include <cstdint>
#include <vector>

namespace gemma4 {

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
// KV cache tensors (indices 5..34) are BPU-owned and don't need CPU flush.
constexpr int kKvInputStart = 5;  // first KV cache input tensor index
constexpr int kLogitsOutputIndex = 0;

// Indices to flush: embedding, token_ids, position_ids, full_mask, sliding_mask
inline std::vector<int> PrefillFlushIndices() {
  return {0, 1, 2, 3, 4};
}

inline std::vector<int> DecodeFlushIndices() {
  return {0, 1, 2, 3, 4};
}

}  // namespace gemma4
