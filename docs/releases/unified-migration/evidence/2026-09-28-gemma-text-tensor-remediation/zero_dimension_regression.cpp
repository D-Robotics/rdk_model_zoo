// GEMMA-TEXT-R1 regression driver: zero dimensions must be rejected before
// any arithmetic. Host buffers only; no SDK, model, or inference.
//
// Expected behavior after the fix: every zero-dimension case throws
// std::invalid_argument, singleton-adjacent positive layouts still validate.
// Before the fix, cases marked R1 reach INT64_MAX / 0 in
// FlattenedElements (UBSan integer-divide-by-zero, SIGABRT under
// -fno-sanitize-recover).
#include "gemma4_text_tensor.hpp"
#include "gemma4_config.hpp"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

// A [d0 x d1] F32/S16/S8 descriptor with contiguous strides.
hbDNNTensorProperties Matrix(int32_t type, int element, int32_t d0, int32_t d1) {
  hbDNNTensorProperties properties{};
  properties.tensorType = type;
  properties.quantiType = NONE;
  properties.validShape.numDimensions = 2;
  properties.validShape.dimensionSize[0] = d0;
  properties.validShape.dimensionSize[1] = d1;
  properties.stride[1] = element;
  properties.stride[0] = static_cast<int64_t>(d1) * element;
  properties.alignedByteSize = properties.stride[0] * d0;
  return properties;
}

// A realistic SDK may report a positive allocation independent of the
// (broken) dimensions; zero-dimension cases use this so the rejection must
// come from shape handling, not the allocation-size check.
void WithPositiveAllocation(hbDNNTensorProperties &p, int64_t bytes) {
  p.alignedByteSize = bytes;
}

void ExpectReject(const std::string &what, const hbDNNTensorProperties &p,
                  gemma4::TextTensorRole role, int seq_len, int layer = 0) {
  try {
    gemma4::ValidateTextTensor(p, role, seq_len, layer);
  } catch (const std::exception &error) {
    std::printf("ok   rejected %-34s (%s)\n", what.c_str(), error.what());
    return;
  }
  std::printf("FAIL accepted %s\n", what.c_str());
  ++g_failures;
}

void ExpectAccept(const std::string &what, const hbDNNTensorProperties &p,
                  gemma4::TextTensorRole role, int seq_len, int layer = 0) {
  try {
    gemma4::ValidateTextTensor(p, role, seq_len, layer);
  } catch (const std::exception &error) {
    std::printf("FAIL rejected %s (%s)\n", what.c_str(), error.what());
    ++g_failures;
    return;
  }
  std::printf("ok   accepted %-34s\n", what.c_str());
}

}  // namespace

int main() {
  using gemma4::TextTensorRole;

  // R1 exact: reviewer descriptor — F32/NONE [0,1536], strides [6144,4],
  // positive allocation.
  hbDNNTensorProperties r1 =
      Matrix(HB_DNN_TENSOR_TYPE_F32, 4, 0, gemma4::kHiddenSize);
  WithPositiveAllocation(r1, r1.stride[0]);
  ExpectReject("embeds zero leading [0,1536]", r1,
               TextTensorRole::kInputsEmbeds, gemma4::kChunkSize);
  // Decode side of the same shape.
  hbDNNTensorProperties r1_decode =
      Matrix(HB_DNN_TENSOR_TYPE_F32, 4, 0, gemma4::kHiddenSize);
  WithPositiveAllocation(r1_decode, r1_decode.stride[0]);
  ExpectReject("embeds decode zero [0,1536]", r1_decode,
               TextTensorRole::kInputsEmbeds, 1);
  // Singleton-adjacent zero: [1,0,1536] canonicalizes to [0,1536].
  hbDNNTensorProperties adjacent =
      Matrix(HB_DNN_TENSOR_TYPE_F32, 4, 0, gemma4::kHiddenSize);
  adjacent.validShape.numDimensions = 3;
  adjacent.validShape.dimensionSize[0] = 1;
  adjacent.validShape.dimensionSize[1] = 0;
  adjacent.stride[2] = 4;
  adjacent.stride[1] = static_cast<int64_t>(gemma4::kHiddenSize) * 4;
  adjacent.stride[0] = adjacent.stride[1];
  WithPositiveAllocation(adjacent, adjacent.stride[1]);
  ExpectReject("embeds [1,0,1536] singleton-adjacent", adjacent,
               TextTensorRole::kInputsEmbeds, gemma4::kChunkSize);
  // Masks and logits with a zero leading row count.
  hbDNNTensorProperties zero_mask =
      Matrix(HB_DNN_TENSOR_TYPE_S16, 2, 0, gemma4::kCacheLen);
  WithPositiveAllocation(zero_mask, zero_mask.stride[0]);
  ExpectReject("mask zero leading [0,4096]", zero_mask,
               TextTensorRole::kFullMask, gemma4::kChunkSize);
  hbDNNTensorProperties zero_logits =
      Matrix(HB_DNN_TENSOR_TYPE_S16, 2, 0, gemma4::kVocabSize);
  WithPositiveAllocation(zero_logits, zero_logits.stride[0]);
  ExpectReject("logits zero leading [0,vocab]", zero_logits,
               TextTensorRole::kLogits, gemma4::kChunkSize);
  // Zero position count.
  hbDNNTensorProperties zero_positions =
      Matrix(HB_DNN_TENSOR_TYPE_S32, 4, 0, 1);
  WithPositiveAllocation(zero_positions, 4);
  ExpectReject("positions zero [0]", zero_positions,
               TextTensorRole::kPositionIds, gemma4::kChunkSize);
  // Zero trailing dimension is rejected without arithmetic.
  hbDNNTensorProperties zero_trailing =
      Matrix(HB_DNN_TENSOR_TYPE_F32, 4, gemma4::kChunkSize, 0);
  WithPositiveAllocation(zero_trailing, 4);
  ExpectReject("embeds zero trailing [256,0]", zero_trailing,
               TextTensorRole::kInputsEmbeds, gemma4::kChunkSize);
  // Internal zero inside a rank-3 declaration.
  hbDNNTensorProperties internal_zero =
      Matrix(HB_DNN_TENSOR_TYPE_F32, 4, gemma4::kChunkSize, gemma4::kHiddenSize);
  internal_zero.validShape.numDimensions = 3;
  internal_zero.validShape.dimensionSize[0] = gemma4::kChunkSize;
  internal_zero.validShape.dimensionSize[1] = 0;
  internal_zero.validShape.dimensionSize[2] = gemma4::kHiddenSize;
  ExpectReject("embeds [256,0,1536] internal zero", internal_zero,
               TextTensorRole::kInputsEmbeds, gemma4::kChunkSize);
  // KV cache input with a zero leading axis ([0,1,head_dim]).
  hbDNNTensorProperties zero_cache =
      Matrix(HB_DNN_TENSOR_TYPE_S8, 1, 0, gemma4::kHeadDims[0]);
  zero_cache.validShape.numDimensions = 3;
  zero_cache.validShape.dimensionSize[0] = 0;
  zero_cache.validShape.dimensionSize[1] = 1;
  zero_cache.validShape.dimensionSize[2] = gemma4::kHeadDims[0];
  zero_cache.stride[2] = 1;
  zero_cache.stride[1] = gemma4::kHeadDims[0];
  zero_cache.stride[0] = gemma4::kHeadDims[0];
  WithPositiveAllocation(zero_cache, gemma4::kHeadDims[0]);
  ExpectReject("kv input [0,1,256] zero", zero_cache,
               TextTensorRole::kKvInput, gemma4::kChunkSize, 0);
  // KV output with a zero row count.
  hbDNNTensorProperties zero_rows =
      Matrix(HB_DNN_TENSOR_TYPE_S8, 1, 0, gemma4::kHeadDims[0]);
  WithPositiveAllocation(zero_rows, gemma4::kHeadDims[0]);
  ExpectReject("kv output zero rows [0,256]", zero_rows,
               TextTensorRole::kKvOutput, gemma4::kChunkSize, 0);

  // Singleton-adjacent positive layouts must keep validating.
  hbDNNTensorProperties positive =
      Matrix(HB_DNN_TENSOR_TYPE_F32, 4, gemma4::kChunkSize, gemma4::kHiddenSize);
  positive.validShape.numDimensions = 4;
  positive.validShape.dimensionSize[0] = 1;
  positive.validShape.dimensionSize[1] = gemma4::kChunkSize;
  positive.validShape.dimensionSize[2] = 1;
  positive.validShape.dimensionSize[3] = gemma4::kHiddenSize;
  positive.stride[3] = 4;
  positive.stride[2] = 4;
  positive.stride[1] = static_cast<int64_t>(gemma4::kHiddenSize) * 4;
  positive.stride[0] = positive.stride[1];
  ExpectAccept("embeds [1,256,1,1536]", positive,
               TextTensorRole::kInputsEmbeds, gemma4::kChunkSize);
  hbDNNTensorProperties kv_positive = zero_cache;
  kv_positive.validShape.dimensionSize[0] = gemma4::kCacheLen;
  kv_positive.alignedByteSize =
      static_cast<int64_t>(gemma4::kHeadDims[0]) * gemma4::kCacheLen + 64;
  ExpectAccept("kv input [4096,1,256]", kv_positive,
               TextTensorRole::kKvInput, gemma4::kChunkSize, 0);

  if (g_failures != 0) {
    std::printf("%d unexpected outcomes\n", g_failures);
    return 1;
  }
  std::printf("zero-dimension regression: all cases behaved as specified\n");
  return 0;
}
