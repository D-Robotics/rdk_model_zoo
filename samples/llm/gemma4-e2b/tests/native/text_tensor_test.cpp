// Contract tests for the fixed Text export tensor helpers: host buffers
// only, no SDK, model, or inference.
#include "gemma4_text_tensor.hpp"
#include "gemma4_config.hpp"
#include "hb_utils.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using namespace gemma4;

namespace {

template <class F> void Rejects(F action) {
  bool threw = false;
  try {
    action();
  } catch (const std::exception &) {
    threw = true;
  }
  assert(threw);
}

// Row-padded [rows, cols] descriptor with an optional leading singleton.
hbDNNTensorProperties MatrixProps(int type, int32_t rows, int32_t cols,
                                  int64_t row_pad, bool lead_singleton) {
  hbDNNTensorProperties properties{};
  properties.tensorType = type;
  properties.quantiType = NONE;
  const int element = ElementSize(type);
  if (lead_singleton) {
    properties.validShape.numDimensions = 3;
    properties.validShape.dimensionSize[0] = 1;
    properties.validShape.dimensionSize[1] = rows;
    properties.validShape.dimensionSize[2] = cols;
    properties.stride[2] = element;
    properties.stride[1] = static_cast<int64_t>(cols) * element + row_pad;
    properties.stride[0] = properties.stride[1] * rows;
    properties.alignedByteSize = properties.stride[1] * rows;
  } else {
    properties.validShape.numDimensions = 2;
    properties.validShape.dimensionSize[0] = rows;
    properties.validShape.dimensionSize[1] = cols;
    properties.stride[1] = element;
    properties.stride[0] = static_cast<int64_t>(cols) * element + row_pad;
    properties.alignedByteSize = properties.stride[0] * rows;
  }
  return properties;
}

// Source graph declares token ids as [1, seq].
hbDNNTensorProperties VectorProps(int type, int32_t size) {
  hbDNNTensorProperties properties{};
  properties.tensorType = type;
  properties.quantiType = NONE;
  properties.validShape.numDimensions = 2;
  properties.validShape.dimensionSize[0] = 1;
  properties.validShape.dimensionSize[1] = size;
  properties.stride[1] = ElementSize(type);
  properties.stride[0] = static_cast<int64_t>(size) * properties.stride[1];
  properties.alignedByteSize = properties.stride[0];
  return properties;
}

// Cache declarations: [cache_len, head_dim], or [cache_len, 1, head_dim] as
// the source graph declares.
hbDNNTensorProperties CacheProps(int32_t rows, int32_t cols, bool middle_one) {
  hbDNNTensorProperties properties{};
  properties.tensorType = HB_DNN_TENSOR_TYPE_S8;
  properties.quantiType = NONE;
  properties.validShape.numDimensions = middle_one ? 3 : 2;
  properties.validShape.dimensionSize[0] = rows;
  if (middle_one) {
    properties.validShape.dimensionSize[1] = 1;
    properties.validShape.dimensionSize[2] = cols;
    // Contiguous [rows, 1, cols]: the singleton axis contributes nothing.
    properties.stride[2] = 1;
    properties.stride[1] = cols;
    properties.stride[0] = cols;
  } else {
    properties.validShape.dimensionSize[1] = cols;
    properties.stride[1] = 1;
    properties.stride[0] = cols;
  }
  properties.alignedByteSize = static_cast<int64_t>(cols) * rows + 64;
  return properties;
}

void StoreInt16(std::vector<unsigned char> &memory, int64_t offset,
                int16_t value) {
  std::memcpy(memory.data() + offset, &value, sizeof(value));
}

}  // namespace

int main() {
  using namespace gemma4;

  // ---- inputs_embeds: F32 [256,1536], row padding, leading singleton ----
  auto embed_props =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kHiddenSize, 32, true);
  std::vector<unsigned char> embed_memory(embed_props.alignedByteSize, 0xcd);
  hbDNNTensor embeds{};
  embeds.properties = embed_props;
  embeds.sysMem.virAddr = embed_memory.data();
  std::vector<float> hidden(static_cast<size_t>(kChunkSize) * kHiddenSize);
  for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = static_cast<float>(i);
  WriteTextInput(embeds, hidden.data(), hidden.size(),
                 TextTensorRole::kInputsEmbeds, kChunkSize,
                 embed_props.alignedByteSize);
  float written = -1.f;
  std::memcpy(&written, embed_memory.data(), 4);
  assert(written == 0.f);
  std::memcpy(&written, embed_memory.data() + embed_props.stride[2], 4);
  assert(written == 1.f);
  const int64_t row1 = embed_props.stride[1];
  std::memcpy(&written, embed_memory.data() + row1 + kHiddenSize * 4 - 4, 4);
  // Row 1's last element is hidden[2*kHiddenSize - 1].
  assert(written == static_cast<float>(2 * kHiddenSize - 1));
  // Inter-row padding is zeroed, never left with stale 0xcd bytes.
  for (int64_t pad = row1 - 32; pad < row1; ++pad)
    assert(embed_memory[static_cast<size_t>(pad)] == 0);
  Rejects([&] { WriteTextInput(embeds, hidden.data(), hidden.size() - 1,
                               TextTensorRole::kInputsEmbeds, kChunkSize,
                               embed_props.alignedByteSize); });
  Rejects([&] { WriteTextInput(embeds, hidden.data(), hidden.size(),
                               TextTensorRole::kInputsEmbeds, kChunkSize,
                               embed_props.alignedByteSize - 1); });
  embeds.sysMem.virAddr = nullptr;
  Rejects([&] { WriteTextInput(embeds, hidden.data(), hidden.size(),
                               TextTensorRole::kInputsEmbeds, kChunkSize,
                               embed_props.alignedByteSize); });
  // Element gaps cannot be honored by the compact innermost copy.
  auto gapped_embeds =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kHiddenSize, 0, false);
  gapped_embeds.stride[1] = 8;
  gapped_embeds.stride[0] = static_cast<int64_t>(kHiddenSize) * 8;
  gapped_embeds.alignedByteSize = gapped_embeds.stride[0] * kChunkSize;
  hbDNNTensor gapped{};
  gapped.properties = gapped_embeds;
  gapped.sysMem.virAddr = embed_memory.data();
  Rejects([&] { WriteTextInput(gapped, hidden.data(), hidden.size(),
                               TextTensorRole::kInputsEmbeds, kChunkSize,
                               gapped_embeds.alignedByteSize); });

  // ---- token ids: S64 [1,256]; position ids: S32 [256] ----
  std::vector<int64_t> tokens(static_cast<size_t>(kChunkSize), 0);
  tokens[7] = kImageTokenId;
  hbDNNTensor token_tensor{};
  token_tensor.properties = VectorProps(HB_DNN_TENSOR_TYPE_S64, kChunkSize);
  std::vector<unsigned char> token_memory(
      token_tensor.properties.alignedByteSize, 0xcd);
  token_tensor.sysMem.virAddr = token_memory.data();
  WriteTextInput(token_tensor, tokens.data(), tokens.size(),
                 TextTensorRole::kTokenIds, kChunkSize,
                 token_tensor.properties.alignedByteSize);
  int64_t token_read = 0;
  std::memcpy(&token_read, token_memory.data() + 7 * 8, 8);
  assert(token_read == kImageTokenId);
  assert(token_memory[8 * 8] == 0);
  auto position_props = VectorProps(HB_DNN_TENSOR_TYPE_S32, kChunkSize);
  position_props.validShape.numDimensions = 1;
  position_props.validShape.dimensionSize[0] = kChunkSize;
  position_props.stride[0] = 4;
  position_props.alignedByteSize = static_cast<int64_t>(kChunkSize) * 4;
  ValidateTextTensor(position_props, TextTensorRole::kPositionIds, kChunkSize);
  auto tokens_as_s32 = VectorProps(HB_DNN_TENSOR_TYPE_S32, kChunkSize);
  Rejects([&] {
    ValidateTextTensor(tokens_as_s32, TextTensorRole::kTokenIds, kChunkSize);
  });
  auto tokens_short = VectorProps(HB_DNN_TENSOR_TYPE_S64, kChunkSize - 1);
  Rejects([&] {
    ValidateTextTensor(tokens_short, TextTensorRole::kTokenIds, kChunkSize);
  });

  // ---- masks: S16 [256,4096], row padding allowed ----
  auto mask_props =
      MatrixProps(HB_DNN_TENSOR_TYPE_S16, kChunkSize, kCacheLen, 16, false);
  hbDNNTensor mask_tensor{};
  mask_tensor.properties = mask_props;
  std::vector<unsigned char> mask_memory(mask_props.alignedByteSize, 0xcd);
  mask_tensor.sysMem.virAddr = mask_memory.data();
  std::vector<int16_t> mask(static_cast<size_t>(kChunkSize) * kCacheLen,
                            static_cast<int16_t>(kMaskValue));
  mask[0] = 0;
  mask[kCacheLen - 1] = 0;
  WriteTextInput(mask_tensor, mask.data(), mask.size(),
                 TextTensorRole::kFullMask, kChunkSize,
                 mask_props.alignedByteSize);
  int16_t mask_read = 0;
  std::memcpy(&mask_read, mask_memory.data(), 2);
  assert(mask_read == 0);
  std::memcpy(&mask_read, mask_memory.data() + (kCacheLen - 1) * 2, 2);
  assert(mask_read == 0);
  std::memcpy(&mask_read, mask_memory.data() + mask_props.stride[0], 2);
  assert(mask_read == static_cast<int16_t>(kMaskValue));  // Row 1, column 0.
  for (int64_t pad = kCacheLen * 2; pad < mask_props.stride[0]; ++pad)
    assert(mask_memory[static_cast<size_t>(pad)] == 0);  // Row padding zeroed.
  auto float_mask =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kCacheLen, 0, false);
  Rejects([&] {
    ValidateTextTensor(float_mask, TextTensorRole::kFullMask, kChunkSize);
  });

  // ---- logits: S16 [256,262144]; argmax honors row and column strides ----
  auto logit_props =
      MatrixProps(HB_DNN_TENSOR_TYPE_S16, kChunkSize, kVocabSize, 64, true);
  std::vector<unsigned char> logit_memory(logit_props.alignedByteSize, 0);
  StoreInt16(logit_memory, 128 * 2, 5);                        // row 0
  StoreInt16(logit_memory, logit_props.stride[1] + 5120 * 2, 9);  // row 1
  StoreInt16(logit_memory, 2 * logit_props.stride[1] + 5120 * 2, 9);  // row 2
  StoreInt16(logit_memory, 4 * logit_props.stride[1] + 3 * 2, -2);    // row 4
  // One row carries a genuine tie: the first maximum must win.
  StoreInt16(logit_memory, 5 * logit_props.stride[1] + 100 * 2, 7);
  StoreInt16(logit_memory, 5 * logit_props.stride[1] + 200 * 2, 7);
  hbDNNTensor logits{};
  logits.properties = logit_props;
  logits.sysMem.virAddr = logit_memory.data();
  assert(ArgmaxTextLogits(logits, 0, kChunkSize, logit_props.alignedByteSize) ==
         128);
  assert(ArgmaxTextLogits(logits, 1, kChunkSize, logit_props.alignedByteSize) ==
         5120);
  assert(ArgmaxTextLogits(logits, 2, kChunkSize, logit_props.alignedByteSize) ==
         5120);
  assert(ArgmaxTextLogits(logits, 3, kChunkSize, logit_props.alignedByteSize) ==
         0);  // All zero: the first index wins.
  assert(ArgmaxTextLogits(logits, 4, kChunkSize, logit_props.alignedByteSize) ==
         0);  // Negative storage loses against zero.
  assert(ArgmaxTextLogits(logits, 5, kChunkSize, logit_props.alignedByteSize) ==
         100);
  Rejects([&] {
    ArgmaxTextLogits(logits, kChunkSize, kChunkSize,
                     logit_props.alignedByteSize);
  });
  auto vocab_short =
      MatrixProps(HB_DNN_TENSOR_TYPE_S16, kChunkSize, kVocabSize / 2, 0, false);
  Rejects([&] {
    ValidateTextTensor(vocab_short, TextTensorRole::kLogits, kChunkSize);
  });
  auto vocab_padded = MatrixProps(HB_DNN_TENSOR_TYPE_S16, kChunkSize,
                                  kVocabSize + 64, 0, false);
  Rejects([&] {
    ValidateTextTensor(vocab_padded, TextTensorRole::kLogits, kChunkSize);
  });
  // Column gaps are read through the descriptor, not assumed contiguous.
  auto wide_columns = logit_props;
  wide_columns.stride[2] = 4;
  wide_columns.stride[1] = static_cast<int64_t>(kVocabSize) * 4;
  wide_columns.alignedByteSize = wide_columns.stride[1] * kChunkSize;
  std::vector<unsigned char> wide_memory(wide_columns.alignedByteSize, 0);
  StoreInt16(wide_memory, 40 * 4, 6);
  hbDNNTensor wide{};
  wide.properties = wide_columns;
  wide.sysMem.virAddr = wide_memory.data();
  assert(ArgmaxTextLogits(wide, 0, kChunkSize,
                          wide_columns.alignedByteSize) == 40);

  // ---- KV inputs: S8 [4096, head_dim], fully contiguous ----
  for (int layer = 0; layer < kNumKvLayers; ++layer) {
    auto cache = CacheProps(kCacheLen, kHeadDims[layer], layer % 2 == 0);
    ValidateTextTensor(cache, TextTensorRole::kKvInput, kChunkSize, layer);
    // The cache input spans the whole physical cache, never one chunk.
    auto chunk_shaped = CacheProps(kChunkSize, kHeadDims[layer], false);
    Rejects([&] {
      ValidateTextTensor(chunk_shaped, TextTensorRole::kKvInput, kChunkSize,
                         layer);
    });
    auto padded_rows = CacheProps(kCacheLen, kHeadDims[layer], false);
    padded_rows.stride[0] += 8;  // Internal row padding breaks raw rolling.
    Rejects([&] {
      ValidateTextTensor(padded_rows, TextTensorRole::kKvInput, kChunkSize,
                         layer);
    });
  }
  auto wrong_head = CacheProps(kCacheLen, 128, false);
  Rejects([&] {
    ValidateTextTensor(wrong_head, TextTensorRole::kKvInput, kChunkSize, 0);
  });
  auto kv_input_f16 = CacheProps(kCacheLen, kHeadDims[0], false);
  kv_input_f16.tensorType = HB_DNN_TENSOR_TYPE_F16;
  kv_input_f16.stride[1] = 2;
  kv_input_f16.stride[0] = static_cast<int64_t>(kHeadDims[0]) * 2;
  kv_input_f16.alignedByteSize = kv_input_f16.stride[0] * kCacheLen + 64;
  Rejects([&] {
    ValidateTextTensor(kv_input_f16, TextTensorRole::kKvInput, kChunkSize, 0);
  });

  // ---- KV outputs: S8 [rows, head_dim], row padding allowed ----
  auto kv_out =
      MatrixProps(HB_DNN_TENSOR_TYPE_S8, kChunkSize, kHeadDims[3], 8, false);
  std::vector<unsigned char> kv_memory(kv_out.alignedByteSize, 0);
  for (int row = 0; row < 5; ++row)
    for (int col = 0; col < kHeadDims[3]; ++col)
      kv_memory[static_cast<size_t>(row * kv_out.stride[0] + col)] =
          static_cast<unsigned char>(row * 31 + col);
  hbDNNTensor kv_output{};
  kv_output.properties = kv_out;
  kv_output.sysMem.virAddr = kv_memory.data();
  const TextKvRows rows3 =
      TextKvOutputRows(kv_output, 3, kChunkSize, kv_out.alignedByteSize);
  assert(rows3.row_stride == kv_out.stride[0]);
  assert(rows3.data[2 * kv_out.stride[0]] == 62);  // Row 2 read via the stride.
  // A one-row decode binding must not accept the 256-row prefill shape.
  Rejects([&] { TextKvOutputRows(kv_output, 3, 1, kv_out.alignedByteSize); });
  Rejects([&] {
    TextKvOutputRows(kv_output, 3, kChunkSize, kv_out.alignedByteSize - 1);
  });
  auto kv_out_s32 =
      MatrixProps(HB_DNN_TENSOR_TYPE_S32, kChunkSize, kHeadDims[3], 0, false);
  Rejects([&] {
    ValidateTextTensor(kv_out_s32, TextTensorRole::kKvOutput, kChunkSize, 3);
  });

  // ---- output/KV bindings are never CPU-written ----
  hbDNNTensor writable{};
  writable.properties = logit_props;
  writable.sysMem.virAddr = logit_memory.data();
  Rejects([&] { WriteTextInput(writable, logit_memory.data(), kVocabSize,
                               TextTensorRole::kLogits, kChunkSize,
                               logit_props.alignedByteSize); });

  // ---- sequence length, unknown types, quantization flags, overlap ----
  auto short_seq = MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize / 2,
                               kHiddenSize, 0, false);
  Rejects([&] {
    ValidateTextTensor(short_seq, TextTensorRole::kInputsEmbeds, kChunkSize);
  });
  auto unknown_type =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kHiddenSize, 0, false);
  unknown_type.tensorType = 99;  // An out-of-contract storage type.
  Rejects([&] {
    ValidateTextTensor(unknown_type, TextTensorRole::kInputsEmbeds, kChunkSize);
  });
  auto scaled =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kHiddenSize, 0, false);
  scaled.quantiType = SCALE;
  Rejects([&] {
    ValidateTextTensor(scaled, TextTensorRole::kInputsEmbeds, kChunkSize);
  });
  auto overlapping =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kHiddenSize, 0, false);
  overlapping.stride[1] = 2;  // Columns overlap within a row.
  Rejects([&] {
    ValidateTextTensor(overlapping, TextTensorRole::kInputsEmbeds, kChunkSize);
  });
  auto oversized_stride = logit_props;
  oversized_stride.stride[1] = std::numeric_limits<int64_t>::max() / 4;
  Rejects([&] {
    ValidateTextTensor(oversized_stride, TextTensorRole::kLogits, kChunkSize);
  });

  // ---- zero dimensions are rejected before any arithmetic (R1) ----
  // A positive allocation with a zero dimension must fail on shape handling,
  // not on the allocation-size check, and must never reach the element-count
  // division.
  auto zero_leading = MatrixProps(HB_DNN_TENSOR_TYPE_F32, 0, kHiddenSize, 0,
                                  false);
  zero_leading.alignedByteSize = zero_leading.stride[0];
  Rejects([&] {
    ValidateTextTensor(zero_leading, TextTensorRole::kInputsEmbeds, kChunkSize);
  });
  Rejects([&] {
    ValidateTextTensor(zero_leading, TextTensorRole::kInputsEmbeds, 1);
  });
  auto zero_adjacent = zero_leading;
  zero_adjacent.validShape.numDimensions = 3;
  zero_adjacent.validShape.dimensionSize[0] = 1;  // Singleton beside the zero.
  zero_adjacent.validShape.dimensionSize[1] = 0;
  zero_adjacent.validShape.dimensionSize[2] = kHiddenSize;
  zero_adjacent.stride[2] = 4;
  zero_adjacent.stride[1] = static_cast<int64_t>(kHiddenSize) * 4;
  zero_adjacent.stride[0] = zero_adjacent.stride[1];
  Rejects([&] {
    ValidateTextTensor(zero_adjacent, TextTensorRole::kInputsEmbeds,
                       kChunkSize);
  });
  auto zero_mask = MatrixProps(HB_DNN_TENSOR_TYPE_S16, 0, kCacheLen, 0, false);
  zero_mask.alignedByteSize = zero_mask.stride[0];
  Rejects([&] {
    ValidateTextTensor(zero_mask, TextTensorRole::kFullMask, kChunkSize);
  });
  auto zero_logits = MatrixProps(HB_DNN_TENSOR_TYPE_S16, 0, kVocabSize, 0, true);
  zero_logits.alignedByteSize = zero_logits.stride[1];
  Rejects([&] {
    ValidateTextTensor(zero_logits, TextTensorRole::kLogits, kChunkSize);
  });
  auto zero_trailing = MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, 0, 0,
                                   false);
  zero_trailing.alignedByteSize = 4;
  Rejects([&] {
    ValidateTextTensor(zero_trailing, TextTensorRole::kInputsEmbeds, kChunkSize);
  });
  auto zero_cache = CacheProps(0, kHeadDims[0], true);
  zero_cache.alignedByteSize = kHeadDims[0];
  Rejects([&] {
    ValidateTextTensor(zero_cache, TextTensorRole::kKvInput, kChunkSize, 0);
  });
  // Singleton-adjacent positive layouts keep validating (and writing).
  auto adjacent_positive =
      MatrixProps(HB_DNN_TENSOR_TYPE_F32, kChunkSize, kHiddenSize, 0, false);
  adjacent_positive.validShape.numDimensions = 4;
  adjacent_positive.validShape.dimensionSize[0] = 1;
  adjacent_positive.validShape.dimensionSize[1] = kChunkSize;
  adjacent_positive.validShape.dimensionSize[2] = 1;
  adjacent_positive.validShape.dimensionSize[3] = kHiddenSize;
  adjacent_positive.stride[3] = 4;
  adjacent_positive.stride[2] = 4;
  adjacent_positive.stride[1] = static_cast<int64_t>(kHiddenSize) * 4;
  adjacent_positive.stride[0] = adjacent_positive.stride[1];
  adjacent_positive.alignedByteSize = adjacent_positive.stride[1] * kChunkSize;
  ValidateTextTensor(adjacent_positive, TextTensorRole::kInputsEmbeds,
                     kChunkSize);

  std::cout << "text tensor dtype/shape/stride/capacity/argmax checks passed\n";
  return 0;
}
